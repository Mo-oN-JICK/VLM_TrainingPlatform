"""정답 Text 스키마 — 렌더러와 파서를 같은 정의에서 만든다.

학습 때 문자열을 만든 규칙이 추론 때 문자열을 읽는 규칙과 달라질 수 없어야 한다.
설계 문서 06.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class Step:
    id: str
    label: str = ""
    type: str = "enum"  # enum | object | bool | float | int | text
    values: Tuple[str, ...] = ()
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evidence: str = "required"  # required | optional | none
    depends_on: Tuple[str, ...] = ()
    rules: Tuple[str, ...] = ()


@dataclass
class AnswerSchema:
    id: str
    version: str = "1.0.0"
    render: str = "tagged"  # tagged | json
    locale: str = "ko"
    steps: List[Step] = field(default_factory=list)
    max_tokens: int = 320
    step_order: str = "strict"
    vocabulary: str = "closed"
    source_path: str = ""

    # ── 로드 ────────────────────────────────────────────────────────────
    @staticmethod
    def load(path: str) -> "AnswerSchema":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if data.get("kind") != "AnswerSchema":
            raise ValueError(f"{path}: kind는 AnswerSchema여야 한다")
        c = data.get("constraints") or {}
        steps = []
        for s in data.get("steps") or []:
            steps.append(
                Step(
                    id=str(s["id"]),
                    label=str(s.get("label", s["id"])),
                    type=str(s.get("type", "enum")),
                    values=tuple(s.get("values") or ()),
                    fields=dict(s.get("fields") or {}),
                    evidence=str(s.get("evidence", "required")),
                    depends_on=tuple(s.get("depends_on") or ()),
                    rules=tuple(s.get("rules") or ()),
                )
            )
        return AnswerSchema(
            id=str(data.get("id", "answer")),
            version=str(data.get("version", "1.0.0")),
            render=str(data.get("render", "tagged")),
            locale=str(data.get("locale", "ko")),
            steps=steps,
            max_tokens=int(c.get("max_tokens", 320)),
            step_order=str(c.get("step_order", "strict")),
            vocabulary=str(c.get("vocabulary", "closed")),
            source_path=path,
        )

    def digest(self) -> Dict[str, Any]:
        """캐시 키와 추론 계약에 들어가는 정규 표현."""
        return {
            "id": self.id,
            "version": self.version,
            "render": self.render,
            "max_tokens": self.max_tokens,
            "steps": [
                {
                    "id": s.id,
                    "type": s.type,
                    "values": list(s.values),
                    "fields": s.fields,
                    "evidence": s.evidence,
                    "rules": list(s.rules),
                }
                for s in self.steps
            ],
        }

    # ── 렌더 ────────────────────────────────────────────────────────────
    def render_answer(self, values: Dict[str, Any], evidence: Dict[str, str]) -> str:
        parts = []
        for s in self.steps:
            body = self._render_value(s, values.get(s.id))
            ev = evidence.get(s.id, "")
            inner = f"{body} — {ev}" if ev else body
            parts.append(f"<{s.id}>{inner}</{s.id}>")
        return "\n".join(parts)

    def _render_value(self, s: Step, v: Any) -> str:
        if s.type == "object" and isinstance(v, dict):
            return ", ".join(f"{k}={_fmt(v.get(k))}" for k in s.fields)
        return _fmt(v)

    # ── 파싱 ────────────────────────────────────────────────────────────
    def parse(self, text: str) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
        """렌더의 역연산. (values, evidence, 파싱 오류 목록)"""
        values: Dict[str, Any] = {}
        evidence: Dict[str, str] = {}
        errors: List[str] = []
        order_seen: List[str] = []

        for s in self.steps:
            m = re.search(rf"<{s.id}>(.*?)</{s.id}>", text, re.S)
            if not m:
                errors.append(f"step_missing:{s.id}")
                continue
            order_seen.append(s.id)
            body = m.group(1).strip()
            if " — " in body:
                body, ev = body.split(" — ", 1)
                evidence[s.id] = ev.strip()
            body = body.strip()
            if s.type == "object":
                obj: Dict[str, Any] = {}
                for kv in body.split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        obj[k.strip()] = _coerce(v.strip())
                values[s.id] = obj
            else:
                values[s.id] = _coerce(body)

        if self.step_order == "strict" and order_seen != [s.id for s in self.steps if s.id in order_seen]:
            errors.append("step_order")
        return values, evidence, errors

    # ── 검증 ────────────────────────────────────────────────────────────
    def validate(self, text: str) -> List[str]:
        """위반 규칙 id 목록. 비어 있으면 통과."""
        values, evidence, errors = self.parse(text)
        v = list(errors)

        for s in self.steps:
            if s.id not in values:
                continue
            val = values[s.id]
            if s.type == "enum" and s.values and val not in s.values:
                v.append(f"enum_domain:{s.id}")
            if s.type == "object":
                if not isinstance(val, dict):
                    v.append(f"object_shape:{s.id}")
                    continue
                for fname, spec in s.fields.items():
                    req_if = spec.get("required_if")
                    present = fname in val and val[fname] is not None
                    if req_if and _safe_eval(req_if, val) and not present:
                        v.append(f"required_if:{s.id}.{fname}")
                    rng = spec.get("range")
                    if present and rng and isinstance(val[fname], (int, float)):
                        if not (rng[0] <= float(val[fname]) <= rng[1]):
                            v.append(f"range:{s.id}.{fname}")
            if s.evidence == "required" and len(evidence.get(s.id, "")) < 8:
                v.append(f"evidence_missing:{s.id}")

        flat = _flatten(values)
        for s in self.steps:
            for i, rule in enumerate(s.rules):
                try:
                    if not _safe_eval(rule, flat):
                        v.append(f"rule:{s.id}#{i}")
                except Exception:
                    v.append(f"rule_error:{s.id}#{i}")

        if len(text) > self.max_tokens * 4:  # 대략적 상한. 정확한 토큰 수는 G4가 잰다
            v.append("max_tokens")
        return v

    def vocabulary_terms(self) -> List[str]:
        """누설 검사에 쓰는 닫힌 어휘."""
        out: List[str] = []
        for s in self.steps:
            out.extend(str(x) for x in s.values)
        return out


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.2f}"
    return "" if v is None else str(v)


def _coerce(s: str) -> Any:
    t = s.strip()
    if t in ("true", "false"):
        return t == "true"
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def _flatten(values: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in values.items():
        if isinstance(v, dict):
            out[k] = _Obj(v)
            for fk, fv in v.items():
                out[f"{k}_{fk}"] = fv
        else:
            out[k] = v
    return out


class _Obj(dict):
    """규칙 문자열에서 `spike.found` 처럼 점 접근을 허용하기 위한 얇은 래퍼."""

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


def _safe_eval(expr: str, scope: Dict[str, Any]) -> bool:
    """스키마 파일 안의 규칙 표현식만 평가한다(사용자 입력이 아니라 우리 스펙의 일부).

    `=>`는 함의로 취급한다: A => B 는 (not A) or B.
    """
    e = expr.strip()
    if "=>" in e:
        left, right = e.split("=>", 1)
        return (not _safe_eval(left, scope)) or _safe_eval(right, scope)
    return bool(eval(e, {"__builtins__": {}}, dict(scope)))  # noqa: S307
