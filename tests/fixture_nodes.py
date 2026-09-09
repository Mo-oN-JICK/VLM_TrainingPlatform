"""테스트용 노드.

Phase 2의 실제 노드 카탈로그가 들어오기 전까지 컴파일러를 검증하는 최소 세트.
모두 모듈 최상위에 정의된다 — spawn 워커가 임포트할 수 있어야 하기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from vlm_trainer.core.node import Node, NodeKind, Port, RunCtx
from vlm_trainer.core.registry import register
from vlm_trainer.core.types import (
    ANY,
    BaseKind,
    Color,
    DType,
    DimVar,
    Frame,
    Layout,
    Norm,
    PortType,
    Range,
    Var,
    image,
    regions,
    text,
    timeseries,
)

RAW_IMAGE = image().with_sem("raw_image")
NORM_IMAGE = image(
    dtype=DType.F32,
    layout=Layout.CHW,
    value_range=Range.UNIT_0_1,
    norm=Norm.IMAGENET,
    shape=(3, DimVar("H"), DimVar("W")),
)


@dataclass
class SourceParams:
    column: str = "image_path"
    color_space: str = "RGB"


@register(
    type="test.image_source",
    version="1.0.0",
    category="Data Acquisition",
    kind=NodeKind.INPUT,
    outputs={"image": Port(RAW_IMAGE)},
    params=SourceParams,
    recipe_overridable=["column"],
    preview="image",
)
class ImageSource(Node):
    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        return f"dummy:{params.column}"

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"image": None}


@dataclass
class TsSourceParams:
    column: str = "ecg_path"
    hz: float = 500.0


@register(
    type="test.ts_source",
    version="1.0.0",
    category="Data Acquisition",
    kind=NodeKind.INPUT,
    outputs={"series": Port(timeseries(hz=500.0).with_sem("raw_signal"))},
    params=TsSourceParams,
)
class TsSource(Node):
    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        return f"dummy:{params.column}"

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"series": None}


@dataclass
class LabelParams:
    column: str = "label"


@register(
    type="test.label_source",
    version="1.0.0",
    category="Data Acquisition",
    kind=NodeKind.INPUT,
    outputs={"value": Port(text("label"))},
    params=LabelParams,
)
class LabelSource(Node):
    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        return f"dummy:{params.column}"

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"value": "abnormal"}


@dataclass
class TextAssetParams:
    path: str = "knowledge/rules.md"
    max_chars: int = 2400


@register(
    type="test.text_asset",
    version="1.0.0",
    category="Data Acquisition",
    kind=NodeKind.INPUT,
    outputs={"text": Port(text("domain_knowledge"))},
    params=TextAssetParams,
    recipe_overridable=["path"],
)
class TextAsset(Node):
    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        return f"dummy:{params.path}"

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"text": "규칙 텍스트"}


@dataclass
class NormParams:
    scheme: str = "imagenet"


@register(
    type="test.norm",
    version="1.0.0",
    category="Adapters",
    kind=NodeKind.PROCESSING,
    inputs={"image": Port(image())},
    outputs={"image": Port(NORM_IMAGE)},
    params=NormParams,
    recipe_overridable=["scheme"],
)
class NormAdapter(Node):
    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        """HWC -> CHW. 출력 차원은 입력에서 그대로 옮겨온다."""
        src = inputs.get("image")
        if src is None or src.shape is None:
            return {name: p.type for name, p in self.definition.outputs.items()}
        h, w, c = src.shape
        return {
            "image": image(
                dtype=DType.F32,
                layout=Layout.CHW,
                value_range=Range.UNIT_0_1,
                norm=Norm.IMAGENET,
                shape=(c, h, w),
            )
        }

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"image": inputs["image"]}


@dataclass
class ExpertParams:
    plugin: str = "dummy@0.1.0"
    topk: int = 3
    score_threshold: float = 0.3


@register(
    type="test.expert",
    version="1.0.0",
    category="Expert Models",
    kind=NodeKind.PROCESSING,
    inputs={"subject": Port(PortType(base=Var("T")))},
    outputs={"regions": Port(regions(domain=Var("D"), frame=Var("F")).with_sem("expert_hint"))},
    params=ExpertParams,
    recipe_overridable=["topk", "score_threshold"],
    external_call=True,
    preview="regions_overlay",
)
class Expert(Node):
    """제네릭 노드. 출력 타입은 입력의 종류에서 확정된다."""

    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        src = inputs.get("subject")
        if src is None:  # 미연결이면 확정할 수 없다 -> G2가 잡는다
            return {name: p.type for name, p in self.definition.outputs.items()}
        domain = "image2d" if src.base is BaseKind.IMAGE else "series1d"
        return {"regions": regions(domain=domain, frame=src.frame).with_sem("expert_hint")}

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"regions": []}


@dataclass
class CropParams:
    padding_ratio: float = 0.15
    max_n: int = 3


@register(
    type="test.crop",
    version="1.0.0",
    category="2D General Processing",
    kind=NodeKind.PROCESSING,
    inputs={
        "image": Port(image()),
        "regions": Port(regions(domain="image2d", frame=Frame.ORIG_PX)),
    },
    outputs={"crops": Port(image(frame=Frame.CROP_PX).as_list(1, 16).with_sem("roi_crop"))},
    params=CropParams,
    recipe_overridable=["padding_ratio", "max_n"],
    preview="image_grid",
)
class Crop(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"crops": []}


@dataclass
class PromptParams:
    template: str = "{task}"


@register(
    type="test.prompt",
    version="1.0.0",
    category="Prompt Assembly",
    kind=NodeKind.PROCESSING,
    inputs={"text": Port(text())},
    outputs={"text": Port(text("prompt"))},
    params=PromptParams,
    recipe_overridable=["template"],
    preview="prompt_render",
)
class Prompt(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"text": params.template}


@dataclass
class GuardParams:
    mode: str = "normalized"
    on_leak: str = "fail"


@register(
    type="test.leakage_guard",
    version="1.0.0",
    category="Answer Design",
    kind=NodeKind.PROCESSING,
    inputs={"prompt": Port(text("prompt")), "answer": Port(text("answer"))},
    outputs={"prompt": Port(text("prompt"))},
    params=GuardParams,
    clears_taint=["label", "answer"],
    preview="leak_report",
)
class LeakageGuard(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"prompt": inputs["prompt"]}


@dataclass
class AnswerParams:
    render: str = "tagged"


@register(
    type="test.answer",
    version="1.0.0",
    category="Answer Design",
    kind=NodeKind.PROCESSING,
    inputs={"value": Port(text("label"))},
    outputs={"answer": Port(text("answer"))},
    params=AnswerParams,
)
class Answer(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"answer": "<verdict>abnormal</verdict>"}


@dataclass
class SinkParams:
    out_dir: str = "runs/{run_id}/out"


@register(
    type="test.sink",
    version="1.0.0",
    category="File",
    kind=NodeKind.OUTPUT,
    inputs={"text": Port(text()), "extra": Port(text().as_optional())},
    params=SinkParams,
    preview="dry_summary",
)
class Sink(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {}


@dataclass
class ImgSinkParams:
    out_dir: str = "runs/{run_id}/img"


@register(
    type="test.image_sink",
    version="1.0.0",
    category="File",
    kind=NodeKind.OUTPUT,
    inputs={"images": Port(image(frame=Frame.CROP_PX).as_list(1, 16))},
    params=ImgSinkParams,
)
class ImageSink(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {}


# ── 엔진 검증용 (Phase 2) ───────────────────────────────────────────────


@dataclass
class CrashParams:
    mode: str = "hard"  # hard = 프로세스 즉시 종료


@register(
    type="test.crasher",
    version="1.0.0",
    category="Expert Models",
    kind=NodeKind.PROCESSING,
    inputs={"text": Port(text())},
    outputs={"text": Port(text())},
    params=CrashParams,
    external_call=True,  # 격리 실행 대상
)
class Crasher(Node):
    """워커 프로세스를 강제로 죽인다. 엔진이 살아남고 이 노드만 failed가 되어야 한다."""

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        import os

        if params.mode == "hard":
            os._exit(9)
        return {"text": inputs["text"]}


@dataclass
class NonDetParams:
    pass


@register(
    type="test.nondet",
    version="1.0.0",
    category="Adapters",
    kind=NodeKind.PROCESSING,
    inputs={"text": Port(text())},
    outputs={"text": Port(text())},
    params=NonDetParams,
)
class NonDeterministic(Node):
    """결정성을 선언해 놓고 지키지 않는 노드. 결정성 감사가 잡아내야 한다."""

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        import random

        return {"text": f"{inputs['text']}-{random.random()}"}
