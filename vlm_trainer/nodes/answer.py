"""Answer Design — 정답 Text 생성·검증·누설 차단. 설계 문서 06."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from ..answer.schema import AnswerSchema, _safe_eval
from ..core.node import Node, NodeDoc, NodeError, NodeKind, Port, RunCtx
from ..core.registry import register
from ..core.types import ANY, BaseKind, PortType, regions, simple, text


@dataclass
class EvidenceParams:
    rules: str = "schemas/evidence.yaml"
    precision: int = 2


@register(
    type="answer.evidence_rules",
    version="1.0.0",
    category="Answer Design",
    kind=NodeKind.PROCESSING,
    inputs={
        "stats": Port(simple(BaseKind.TABLE), "수치 통계"),
        "regions": Port(regions(domain=ANY, frame=ANY).as_optional(), "전문가 지목(선택)"),
    },
    outputs={"evidence": Port(text("evidence").as_list(1, 32), "단계별 근거 문장")},
    params=EvidenceParams,
    recipe_overridable=["rules"],
    preview="text",
    doc=NodeDoc(
        summary="수치에서 단계별 근거 문장을 규칙으로 생성한다. 결정적이다.",
        scenario="LLM 윤문은 옵션이며 기본 경로에 두지 않는다 — 재현 불가능한 데이터셋이 조용히 만들어지는 것을 막는다.",
    ),
)
class EvidenceRules(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        with open(ctx.path(params.rules), "r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh) or {}
        scope = dict(inputs["stats"])
        rs = inputs.get("regions") or []
        scope["region_count"] = len(rs)
        scope["region_top_score"] = max((float(r.get("score", 0)) for r in rs), default=0.0)

        out: List[str] = []
        for step_id, rules in (spec.get("steps") or {}).items():
            sentence = ""
            for rule in rules:
                if "default" in rule:
                    sentence = sentence or _fill(str(rule["default"]), scope, params.precision)
                    continue
                if _safe_eval(str(rule["when"]), scope):
                    sentence = _fill(str(rule["then"]), scope, params.precision)
                    break
            out.append(f"{step_id}: {sentence}" if sentence else f"{step_id}: ")
        return {"evidence": out}


def _fill(t: str, scope: Dict[str, Any], precision: int) -> str:
    def rep(m: re.Match) -> str:
        key = m.group(1)
        v = scope.get(key, "")
        return f"{v:.{precision}f}" if isinstance(v, float) else str(v)

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", rep, t)


@dataclass
class StepwiseParams:
    render: str = "tagged"
    mapping: dict = field(default_factory=dict)  # step_id -> stats에 대한 식


@register(
    type="answer.stepwise",
    version="1.0.0",
    category="Answer Design",
    kind=NodeKind.PROCESSING,
    inputs={
        "schema": Port(simple(BaseKind.SCHEMA), "정답 스키마"),
        "fields": Port(simple(BaseKind.TABLE), "단계 값을 뽑을 수치"),
        "evidence": Port(text("evidence").as_list(1, 32).as_optional(), "근거 문장(선택)"),
    },
    outputs={"answer": Port(text("answer"), "단계 구조를 가진 정답 Text")},
    params=StepwiseParams,
    recipe_overridable=["render"],
    preview="answer_render",
    doc=NodeDoc(
        summary="스키마가 선언한 단계 순서대로 정답 Text를 만든다.",
        scenario="렌더러와 파서가 같은 스키마에서 생성되므로 학습과 추론이 어긋날 수 없다.",
    ),
)
class Stepwise(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        schema: AnswerSchema = inputs["schema"]
        scope = dict(inputs["fields"])
        values: Dict[str, Any] = {}
        for s in schema.steps:
            expr = (params.mapping or {}).get(s.id)
            if expr is None:
                values[s.id] = scope.get(s.id)
                continue
            if isinstance(expr, dict):  # object 단계: 필드별 식
                values[s.id] = {k: _eval_value(v, scope) for k, v in expr.items()}
            else:
                values[s.id] = _eval_value(expr, scope)

        ev: Dict[str, str] = {}
        for line in inputs.get("evidence") or []:
            if ":" in line:
                k, v = line.split(":", 1)
                ev[k.strip()] = v.strip()
        return {"answer": schema.render_answer(values, ev)}


def _eval_value(expr: Any, scope: Dict[str, Any]) -> Any:
    if not isinstance(expr, str):
        return expr
    try:
        return eval(expr, {"__builtins__": {}}, dict(scope))  # noqa: S307 - 스펙 파일의 일부
    except Exception:
        return expr


@dataclass
class ValidateParams:
    on_violation: str = "quarantine"  # quarantine | drop | fail
    max_tokens: int = 320


@register(
    type="answer.validate",
    version="1.0.0",
    category="Answer Design",
    kind=NodeKind.PROCESSING,
    inputs={
        "answer": Port(text("answer"), "정답 Text"),
        "schema": Port(simple(BaseKind.SCHEMA), "정답 스키마"),
    },
    outputs={
        "answer": Port(text("answer", "validated"), "검증을 통과한 정답 Text"),
        "report": Port(simple(BaseKind.REPORT), "위반 규칙 목록"),
    },
    params=ValidateParams,
    recipe_overridable=["on_violation", "max_tokens"],
    preview="answer_validated",
    doc=NodeDoc(
        summary="정답 Text가 스키마를 지키는지 검사한다.",
        scenario="위반 샘플은 격리된다. 위반율이 임계를 넘으면 dry-run이 학습 전에 멈춘다.",
    ),
)
class Validate(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        schema: AnswerSchema = inputs["schema"]
        violations = schema.validate(str(inputs["answer"]))
        if violations and params.on_violation == "fail":
            raise NodeError(ctx.node_id, "스키마 위반: " + ", ".join(violations), sample_key=ctx.sample_key)
        if violations and params.on_violation in ("quarantine", "drop"):
            raise NodeError(
                ctx.node_id,
                "스키마 위반: " + ", ".join(violations),
                sample_key=ctx.sample_key,
                hint="answer.stepwise의 mapping이나 스키마 규칙을 확인하라",
            )
        return {
            "answer": str(inputs["answer"]),
            "report": {"violations": violations, "schema": schema.id, "ok": not violations},
        }


@dataclass
class GuardParams:
    mode: str = "normalized"  # exact | normalized
    on_leak: str = "fail"  # fail | mask
    exclude_sections: tuple = ()
    extra_terms: tuple = ()


@register(
    type="answer.leakage_guard",
    version="1.0.0",
    category="Answer Design",
    kind=NodeKind.PROCESSING,
    inputs={
        "prompt": Port(text("prompt"), "프롬프트"),
        "answer": Port(text("answer"), "정답 Text"),
        "schema": Port(simple(BaseKind.SCHEMA).as_optional(), "닫힌 어휘 출처(선택)"),
    },
    outputs={
        "prompt": Port(text("prompt"), "누설 검사를 통과한 프롬프트"),
        "report": Port(simple(BaseKind.REPORT), "검사 결과"),
    },
    params=GuardParams,
    recipe_overridable=["mode", "on_leak"],
    clears_taint=["label", "answer"],
    preview="leak_report",
    doc=NodeDoc(
        summary="정답 어휘가 프롬프트에 섞였는지 검사한다. taint를 제거하는 유일한 노드.",
        scenario="정적 taint 검사(G2)가 경로를 막고, 이 노드가 실제 문자열을 본다.",
    ),
)
class LeakageGuard(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        prompt = str(inputs["prompt"])
        schema: Optional[AnswerSchema] = inputs.get("schema")
        terms = list(params.extra_terms or ())
        if schema is not None:
            terms += schema.vocabulary_terms()

        haystack = prompt
        for sec in params.exclude_sections or ():
            haystack = haystack.replace(sec, " ")
        if params.mode == "normalized":
            haystack = re.sub(r"\s+", " ", haystack).lower()
            terms = [t.lower() for t in terms]

        hits = sorted({t for t in terms if t and t in haystack})
        if hits and params.on_leak == "fail":
            raise NodeError(
                ctx.node_id,
                f"정답 어휘가 프롬프트에 있다: {hits}",
                sample_key=ctx.sample_key,
                hint="템플릿에서 해당 표현을 빼거나 exclude_sections로 지식 블록을 제외하라",
            )
        if hits and params.on_leak == "mask":
            for t in hits:
                prompt = re.sub(re.escape(t), "[MASKED]", prompt, flags=re.I)
        return {"prompt": prompt, "report": {"leak_terms": hits, "ok": not hits}}
