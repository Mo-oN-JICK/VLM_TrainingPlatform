"""Data Processing과 File — 샘플 조립, 리스트 연산, 그리고 Output 노드들.

Output은 부작용을 일으키는 유일한 분류다. 캐시하지 않고, 미리보기하지 않는다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict

import numpy as np

from ..core.node import Node, NodeDoc, NodeError, NodeKind, Port, RunCtx
from ..core.registry import register
from ..core.types import (
    ANY,
    BaseKind,
    ListSpec,
    PortType,
    Var,
    image,
    simple,
    text,
)
from ..core.unify import unify_ports


@dataclass
class WrapParams:
    pass


@register(
    type="list.wrap",
    version="1.0.0",
    category="Data Processing",
    kind=NodeKind.PROCESSING,
    inputs={"item": Port(PortType(base=Var("T")), "단일 값")},
    outputs={"items": Port(PortType(base=Var("T"), list_of=ListSpec(1, 1)), "원소가 하나인 리스트")},
    params=WrapParams,
    doc=NodeDoc(label="단일값 → 리스트",
            hint="값 하나를 원소가 하나인 리스트로 만듭니다.", summary="단일 값을 리스트로 감싼다. 자동 승격이 없으므로 명시적으로 필요하다."),
)
class ListWrap(Node):
    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        src = inputs.get("item")
        if src is None:
            return {"items": PortType(base=Var("T"), list_of=ListSpec(1, 1))}
        return {"items": replace(src, list_of=ListSpec(1, 1))}

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"items": [inputs["item"]]}


@dataclass
class ConcatParams:
    max_n: int = 8
    on_overflow: str = "fail"  # fail | trim


@register(
    type="list.concat",
    version="1.0.0",
    category="Data Processing",
    kind=NodeKind.PROCESSING,
    inputs={
        "a": Port(PortType(base=Var("T"), list_of=ListSpec(1, 1 << 30)), "앞 리스트"),
        "b": Port(PortType(base=Var("U"), list_of=ListSpec(1, 1 << 30)), "뒤 리스트"),
    },
    outputs={"out": Port(PortType(base=Var("T"), list_of=ListSpec(1, 1 << 30)), "이어붙인 리스트")},
    params=ConcatParams,
    recipe_overridable=["max_n"],
    doc=NodeDoc(label="리스트 병합", hint="같은 타입의 리스트 둘을 이어 붙입니다.", summary="같은 타입의 두 리스트를 이어붙인다."),
)
class ListConcat(Node):
    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        a, b = inputs.get("a"), inputs.get("b")
        if a is None or b is None:
            return {"out": PortType(base=Var("T"), list_of=ListSpec(1, int(params.max_n)))}
        # 리스트에 담기는 이미지들의 출처 태그는 서로 달라도 된다(합집합으로 전파된다).
        # 나머지 필드는 정확히 같아야 한다.
        bare_a = replace(a, list_of=None, semantic=frozenset())
        bare_b = replace(b, list_of=None, semantic=frozenset())
        res = unify_ports(bare_b, bare_a)
        if not res.ok:
            from ..core.errors import PolicyError

            raise PolicyError(
                "list.concat: 두 리스트의 원소 타입이 다르다.\n"
                f"  a: {a}\n  b: {b}\n"
                "  불일치: " + ", ".join(str(m) for m in res.mismatches) + "\n"
                "  이 검사가 없었다면: 서로 다른 전처리를 거친 이미지가 한 샘플에 섞여 "
                "학습은 완주하고 성능만 나빠진다."
            )
        return {
            "out": replace(
                a,
                list_of=ListSpec(1, int(params.max_n)),
                semantic=frozenset(set(a.semantic) | set(b.semantic)),
            )
        }

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        out = list(inputs["a"]) + list(inputs["b"])
        if len(out) > int(params.max_n):
            if params.on_overflow == "fail":
                raise NodeError(
                    ctx.node_id,
                    f"리스트 길이 {len(out)}가 상한 {params.max_n}을 넘는다",
                    sample_key=ctx.sample_key,
                    hint="max_n을 올리거나 상류의 topk를 줄여라",
                )
            out = out[: int(params.max_n)]
        return {"out": out}


@dataclass
class MapParams:
    node: str = ""
    params: dict = field(default_factory=dict)


@register(
    type="list.map",
    version="1.0.0",
    category="Data Processing",
    kind=NodeKind.PROCESSING,
    inputs={"items": Port(PortType(base=Var("T"), list_of=ListSpec(1, 1 << 30)), "입력 리스트")},
    outputs={"items": Port(PortType(base=Var("U"), list_of=ListSpec(1, 1 << 30)), "변환된 리스트")},
    params=MapParams,
    recipe_overridable=["params"],
    doc=NodeDoc(
        label="리스트 일괄 처리",
            hint="리스트의 원소마다 같은 노드를 적용합니다.",
        summary="리스트의 원소마다 노드 하나를 적용한다.",
        scenario="crop 여러 장을 한꺼번에 리사이즈할 때. 브로드캐스트가 없으므로 명시적으로 필요하다.",
    ),
)
class ListMap(Node):
    """지금은 단일 노드만 매핑한다. 서브 Procedure 매핑은 엔진이 서브그래프를 실행할 수 있게 되면 붙인다."""

    def _inner(self, params: Any):
        from ..core.registry import resolve as resolve_node

        if not params.node:
            from ..core.errors import PolicyError

            raise PolicyError("list.map: 적용할 node 파라미터가 비어 있다")
        return resolve_node(params.node)

    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        src = inputs.get("items")
        d = self._inner(params)
        if len(d.inputs) != 1 or len(d.outputs) != 1:
            from ..core.errors import PolicyError

            raise PolicyError(f"list.map: {d.ref}는 입력·출력이 각각 하나여야 한다")
        in_port = next(iter(d.inputs))
        out_port = next(iter(d.outputs))
        if src is None:
            return {"items": PortType(base=Var("U"), list_of=ListSpec(1, 1 << 30))}
        elem = replace(src, list_of=None)
        inner_params = d.build_params({**d.default_params(), **(params.params or {})})
        out_elem = (d.impl() if d.impl else None).infer_types({in_port: elem}, inner_params)[out_port]
        return {"items": replace(out_elem, list_of=src.list_of)}

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        d = self._inner(params)
        impl = d.impl()
        inner_params = d.build_params({**d.default_params(), **(params.params or {})})
        in_port = next(iter(d.inputs))
        out_port = next(iter(d.outputs))
        return {"items": [impl.run(ctx, inner_params, **{in_port: x})[out_port] for x in inputs["items"]]}


@dataclass
class AssembleParams:
    id_from: str = "sample_id"
    require_validated: bool = True
    placeholder: str = "<image>"


@register(
    type="sample.assemble",
    version="1.0.0",
    category="Data Processing",
    kind=NodeKind.PROCESSING,
    inputs={
        "images": Port(image(frame=ANY).as_list(1, 16).as_optional(), "학습에 들어갈 이미지들"),
        "prompt": Port(text("prompt"), "프롬프트"),
        "answer": Port(text("answer", "validated"), "검증된 정답 Text"),
        "meta": Port(simple(BaseKind.TABLE).as_optional(), "함께 남길 메타데이터"),
    },
    outputs={"sample": Port(simple(BaseKind.SAMPLE), "학습 샘플 한 건")},
    params=AssembleParams,
    recipe_overridable=["require_validated"],
    preview="sample_card",
    doc=NodeDoc(
        label="학습 데이터 맵핑",
        hint="이미지·질문·정답을 학습 데이터 한 건으로 묶습니다.",
        summary="이미지·프롬프트·정답을 학습 샘플 한 건으로 묶는다.",
        scenario="자리표시자 개수와 실제 이미지 개수를 실측으로 대조한다 — "
        "어긋나면 이미지가 조용히 무시된 채 텍스트만 학습된다.",
    ),
)
class SampleAssemble(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        images = list(inputs.get("images") or [])
        prompt = str(inputs["prompt"])
        n_slots = prompt.count(params.placeholder)
        if n_slots != len(images):
            raise NodeError(
                ctx.node_id,
                f"이미지 자리표시자 {n_slots}개, 실제 이미지 {len(images)}개로 개수가 다르다",
                sample_key=ctx.sample_key,
                hint="prompt.image_slots가 같은 ImageList를 받고 있는지 확인하라",
            )
        return {
            "sample": {
                "id": str(ctx.sample.get(params.id_from, ctx.sample_key)),
                "images": images,
                "prompt": prompt,
                "answer": str(inputs["answer"]),
                "meta": dict(inputs.get("meta") or {}),
            }
        }


@dataclass
class ExportParams:
    out_dir: str = "runs/{run_id}/dataset"
    format: str = "jsonl"
    image_encoding: str = "png"
    write_images: bool = True


@register(
    type="io.dataset_export",
    version="1.0.0",
    category="File",
    kind=NodeKind.OUTPUT,
    inputs={"sample": Port(simple(BaseKind.SAMPLE), "학습 샘플")},
    params=ExportParams,
    recipe_overridable=["out_dir", "image_encoding"],
    preview="dry_summary",
    doc=NodeDoc(
        label="데이터셋 내보내기",
            hint="생성된 학습 데이터를 디스크에 저장합니다.",
        summary="샘플을 디스크로 내보낸다. 부작용을 일으키는 Output 노드.",
        scenario="물질화 경계 뒤의 학습은 이 산출물만 읽는다.",
    ),
)
class DatasetExport(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        from PIL import Image as PILImage

        s = inputs["sample"]
        out_dir = os.path.abspath(params.out_dir.replace("{run_id}", ctx.run_id))
        os.makedirs(out_dir, exist_ok=True)
        rec: Dict[str, Any] = {"id": s["id"], "prompt": s["prompt"], "answer": s["answer"], "meta": s["meta"]}

        if params.write_images and s.get("images"):
            img_dir = os.path.join(out_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            paths = []
            for i, arr in enumerate(s["images"]):
                p = os.path.join(img_dir, f"{s['id']}_{i}.{params.image_encoding}")
                PILImage.fromarray(np.asarray(arr, dtype=np.uint8)).save(p)
                paths.append(os.path.relpath(p, out_dir))
            rec["images"] = paths

        # 원자적 append: 같은 run 안에서만 열고 닫는다
        with open(os.path.join(out_dir, "samples.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {}


@dataclass
class PromptExportParams:
    out_dir: str = "runs/{run_id}/infer"
    write_images: bool = False


@register(
    type="io.prompt_export",
    version="1.0.0",
    category="File",
    kind=NodeKind.OUTPUT,
    inputs={
        "prompt": Port(text("prompt"), "최종 프롬프트"),
        "images": Port(image(frame=ANY).as_list(1, 16).as_optional(), "함께 넣을 이미지들"),
    },
    params=PromptExportParams,
    recipe_overridable=["out_dir"],
    preview="dry_summary",
    doc=NodeDoc(
        label="프롬프트 내보내기",
            hint="프롬프트를 파일로 저장합니다. 추론 그래프의 종결점입니다.",
        summary="프롬프트를 그대로 내보낸다. 추론 그래프의 종결점.",
        scenario="추론 계약이 실제로 재현되는지 확인하는 수단 — 학습 때의 프롬프트와 바이트 단위로 대조한다.",
    ),
)
class PromptExport(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        out_dir = os.path.abspath(params.out_dir.replace("{run_id}", ctx.run_id))
        os.makedirs(out_dir, exist_ok=True)
        rec = {
            "id": ctx.sample_key,
            "prompt": str(inputs["prompt"]),
            "n_images": len(inputs.get("images") or []),
        }
        with open(os.path.join(out_dir, "prompts.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {}


@dataclass
class TrainerParams:
    config_path: str = "trainer.yaml"  # project.yaml 기준. 설계 문서 07의 선언형 스키마
    out_dir: str = "runs/{run_id}/train"


@register(
    type="train.vlm_trainer",
    version="0.1.0",
    category="Training",
    kind=NodeKind.OUTPUT,
    inputs={
        "sample": Port(simple(BaseKind.SAMPLE), "학습 샘플"),
        "schema": Port(simple(BaseKind.SCHEMA), "정답 스키마"),
    },
    # 학습은 체크포인트를 디스크에 쓴다 — 부작용이므로 Output 이 맞다. 그러면서
    # **산출물이 어디 있는지**를 내보내 `모델 추론` 이 받는다. Output 이 그래프의 끝이어야
    # 한다는 규칙을 푼 이유가 이것이다(설계 문서와 다른 지점. README 표 참고).
    outputs={"model": Port(simple(BaseKind.MODEL), "학습된 모델")},
    params=TrainerParams,
    recipe_overridable=["config_path", "out_dir"],
    per_sample=False,  # 샘플마다가 아니라 물질화된 데이터셋 전체에 한 번
    preview="budget_table",
    doc=NodeDoc(
        label="모델 학습",
        hint="준비된 학습 데이터로 모델을 파인튜닝합니다.",
        summary="학습 실행. 산출물이 어디 있는지를 내보내 추론이 받는다.",
        scenario="굽기가 끝난 뒤에 한 번 돈다. 샘플마다가 아니다.",
    ),
)
class VlmTrainer(Node):
    """Phase 5 전까지의 스텁.

    실제 학습 루프 대신 학습 계획과 샘플 통계를 기록한다. 그래프가 Output으로
    끝나야 한다는 계약을 만족시키고, 예산 게이트가 붙을 자리를 잡아 둔다.
    """

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        from ..train.config import TrainerConfig

        cfg = TrainerConfig.load(ctx.asset(params.config_path))
        out_dir = os.path.abspath(params.out_dir.replace("{run_id}", ctx.run_id))
        os.makedirs(out_dir, exist_ok=True)
        handle = {"model": {"dir": out_dir, "backbone": cfg.backbone}}

        # 산출물이 **어디 있는지**는 샘플과 무관하다. 그래서 입력이 없어도 답할 수 있고,
        # 실행 엔진이 이 값을 한 번만 만들어 모든 샘플에 얹는다. 아래 통계는 덤이다.
        s = inputs.get("sample")
        if s is None:
            return handle

        plan_path = os.path.join(out_dir, "train_plan.json")
        plan = {
            "backbone": cfg.backbone,
            "stages": [st.name for st in cfg.stages],
            "config": cfg.digest(),
            "samples": 0,
            "chars": 0,
        }
        if os.path.exists(plan_path):
            with open(plan_path, "r", encoding="utf-8") as fh:
                plan = json.load(fh)
        plan["samples"] = int(plan.get("samples", 0)) + 1
        plan["chars"] = int(plan.get("chars", 0)) + len(s["prompt"]) + len(s["answer"])
        if inputs.get("schema") is not None:
            plan["schema"] = inputs["schema"].id
        tmp = plan_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(plan, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, plan_path)  # Windows에서도 원자적
        # 가중치를 값으로 흘려보내지 않는다 — 그러면 샘플마다 수 GB 가 캐시에 복사된다.
        return handle
