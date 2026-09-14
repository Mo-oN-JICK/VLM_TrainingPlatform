"""Prompt Assembly.

프롬프트 조립 경로는 정답 경로와 분리되어 있어야 한다. semantic 태그가 그 경계를 표시하고,
컴파일러의 taint 검사가 이를 강제한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..core.node import Node, NodeDoc, NodeKind, Port, RunCtx
from ..core.registry import register
from ..core.types import ANY, BaseKind, PortType, image, simple, text


def _fill(template: str, values: Dict[str, Any]) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{" + str(k) + "}", _fmt(v))
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


@dataclass
class TemplateParams:
    template: str = "{task}"
    strip: bool = True
    max_chars: int = 8000


@register(
    type="text.template",
    version="1.0.0",
    category="Prompt Assembly",
    kind=NodeKind.PROCESSING,
    inputs={"context": Port(simple(BaseKind.TABLE), "슬롯에 채울 값")},
    outputs={"text": Port(text("prompt"), "조립된 프롬프트")},
    params=TemplateParams,
    recipe_overridable=["template", "max_chars"],
    preview="prompt_render",
    doc=NodeDoc(
        label="프롬프트 템플릿",
            hint="표의 값을 템플릿 슬롯에 채워 프롬프트를 만듭니다.",
        summary="표의 값을 템플릿 슬롯에 채워 프롬프트를 만든다.",
        scenario="프롬프트 문구만 바꾸는 실험은 Parameter Recipe로 template만 덮어쓰면 된다.",
    ),
)
class TextTemplate(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        s = _fill(params.template, dict(inputs["context"]))
        if params.strip:
            s = s.strip()
        return {"text": s[: int(params.max_chars)]}


@dataclass
class KnowledgeParams:
    position: str = "before"  # before | after
    header: str = "## 도메인 규칙"
    max_chars: int = 2400
    dedup: bool = True


@register(
    type="prompt.knowledge_inject",
    version="1.0.0",
    category="Prompt Assembly",
    kind=NodeKind.PROCESSING,
    inputs={
        "text": Port(text("prompt"), "프롬프트"),
        "knowledge": Port(text("domain_knowledge"), "도메인 지식 텍스트"),
    },
    outputs={"text": Port(text("prompt"), "지식이 주입된 프롬프트")},
    params=KnowledgeParams,
    recipe_overridable=["position", "max_chars", "header"],
    preview="prompt_render",
    doc=NodeDoc(label="참조 문서 삽입",
            hint="도메인 지식 문서를 프롬프트 앞뒤에 붙입니다.", summary="도메인 지식 블록을 프롬프트 앞이나 뒤에 붙인다."),
)
class KnowledgeInject(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        kb = str(inputs["knowledge"])[: int(params.max_chars)].strip()
        base = str(inputs["text"])
        if params.dedup and kb and kb in base:
            return {"text": base}
        block = f"{params.header}\n{kb}" if params.header else kb
        return {"text": f"{block}\n\n{base}" if params.position == "before" else f"{base}\n\n{block}"}


@dataclass
class ImageSlotParams:
    placeholder: str = "<image>"
    policy: str = "prepend"  # prepend | append


@register(
    type="prompt.image_slots",
    version="1.0.0",
    category="Prompt Assembly",
    kind=NodeKind.PROCESSING,
    inputs={
        "text": Port(text("prompt"), "프롬프트"),
        "images": Port(image(frame=ANY).as_list(1, 16), "함께 넣을 이미지들"),
    },
    outputs={"text": Port(text("prompt"), "이미지 자리표시자가 붙은 프롬프트")},
    params=ImageSlotParams,
    recipe_overridable=["placeholder", "policy"],
    preview="prompt_render",
    doc=NodeDoc(
        label="이미지 슬롯 삽입",
            hint="이미지 개수만큼 자리표시자를 프롬프트에 넣습니다.",
        summary="이미지 개수만큼 자리표시자를 프롬프트에 넣는다.",
        scenario="자리표시자 개수와 실제 이미지 개수가 어긋나면 이미지가 조용히 무시된다. "
        "sample.assemble이 그 개수를 실측으로 대조한다.",
    ),
)
class ImageSlots(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        n = len(inputs["images"])
        slots = " ".join([params.placeholder] * n)
        base = str(inputs["text"])
        return {"text": f"{slots}\n{base}" if params.policy == "prepend" else f"{base}\n{slots}"}
