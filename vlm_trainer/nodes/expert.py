"""Expert Models — 외부 전문가 모델 호출.

노드 타입은 하나다. 이미지 영역 지목과 시계열 구간 지목이 같은 인터페이스로 들어오고,
도메인 차이는 플러그인 매니페스트가 선언하는 타입에서만 드러난다. 설계 문서 05.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..core.errors import PolicyError
from ..core.node import Node, NodeDoc, NodeKind, Port, RunCtx
from ..core.registry import register
from ..core.types import PortType, Var
from ..core.unify import unify_ports
from ..plugins.base import resolve_expert


@dataclass
class ProposeParams:
    plugin: str = ""
    topk: int = 3
    score_threshold: float = 0.0
    device: str = "cpu"
    seed: int = 0


@register(
    type="expert.propose",
    version="1.0.0",
    category="Expert Models",
    kind=NodeKind.PROCESSING,
    inputs={"subject": Port(PortType(base=Var("T")), "이미지 또는 시계열")},
    outputs={"regions": Port(PortType(base=Var("R")), "지목된 영역 또는 구간")},
    params=ProposeParams,
    recipe_overridable=["plugin", "topk", "score_threshold", "seed"],
    type_affecting=["plugin"],
    external_call=True,
    preview="regions_overlay",
    doc=NodeDoc(
        label="전문가 모델 추론",
            hint="외부 모델을 호출해 관심 영역이나 구간을 지목받습니다.",
        summary="외부 전문가 모델이 의심 영역(이미지) 또는 의심 구간(시계열)을 지목한다.",
        scenario="플러그인 문자열만 바꾸면 도메인이 바뀐다. 출력 타입은 플러그인 매니페스트에서 확정된다.",
        ports="subject: 제네릭. 플러그인의 accepts와 단일화된다 / regions: 플러그인의 produces",
    ),
)
class ExpertPropose(Node):
    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        if not params.plugin:
            raise PolicyError(
                "expert.propose: plugin 파라미터가 비어 있어 출력 타입을 확정할 수 없다. "
                "플러그인 id@semver를 지정하라."
            )
        m = resolve_expert(params.plugin).manifest
        src = inputs.get("subject")
        if src is not None:
            res = unify_ports(src, m.accepts)
            if not res.ok:
                raise PolicyError(
                    f"expert.propose: 플러그인 {m.ref}는 {m.accepts}를 받는데 "
                    f"{src}가 연결되었다.\n"
                    "  불일치: " + ", ".join(str(x) for x in res.mismatches) + "\n"
                    "  이 검사가 없었다면: 물질화 도중 전문가 모델이 엉뚱한 배열을 받아 "
                    "실패하거나, 조용히 무의미한 지목을 만든다."
                )
        return {"regions": m.produces}

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        plugin = resolve_expert(params.plugin)()
        plugin.load()
        try:
            out = plugin.propose(inputs["subject"], params, ctx)
        finally:
            plugin.unload()
        keep = [r for r in out if float(r.get("score", 1.0)) >= float(params.score_threshold)]
        keep.sort(key=lambda r: -float(r.get("score", 0.0)))
        return {"regions": keep[: int(params.topk)]}
