"""직접 만든 노드의 예제 하나 — 사진을 정사각으로 채운다.

`adapt.image_resize` 는 448x448 로 맞춰 준다. 그런데 원본이 640x480 이면 **가로가
눌린다.** 우리 정답 스키마는 `orientation`(가로/세로/정사각)을 맞히라고 하는데, 눌린
사진에서는 그 정보가 사라진 뒤다. 모델을 탓하게 되는 자리다.

그래서 리사이즈 **앞에** 이 노드를 넣는다. 긴 변에 맞춰 짧은 변을 채우면 가로세로 비가
보존되고, 그다음 리사이즈는 찌그러뜨리지 않는다.

이 파일은 `vlmt new-node` 가 만든 뼈대에서 시작해 내용만 채운 것이다.
`vlmt check-node examples.custom_node.pad_to_square` 로 검사한다.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from vlm_trainer.core.node import Node, NodeDoc, NodeError, NodeKind, Port, RunCtx
from vlm_trainer.core.registry import register
from vlm_trainer.core.types import DYN, image


@dataclass
class PadToSquareParams:
    """파라미터는 **dataclass 여야 한다**(계약 C3). 앱이 이 선언을 보고 설정 화면을
    자동으로 만든다 — 선언이 없으면 화면에 아무것도 안 뜬다."""

    fill: List[int] = field(default_factory=lambda: [0, 0, 0])


@register(
    type="example.pad_to_square",
    version="1.0.0",
    category="2D General Processing",
    kind=NodeKind.PROCESSING,
    # 들어오는 크기는 샘플마다 다르다(`DYN`). 나가는 것도 크기는 모르지만 **정사각**이다 —
    # 그 사실은 타입에 실을 수 없으므로 여기서는 모양만 같게 두고, 리사이즈가 뒤를 맡는다.
    inputs={"image": Port(image(shape=(DYN, DYN, 3)), "임의 크기 사진")},
    outputs={"image": Port(image(shape=(DYN, DYN, 3)), "정사각으로 채운 사진")},
    params=PadToSquareParams,
    recipe_overridable=["fill"],
    preview="image",
    doc=NodeDoc(
        label="정사각으로 채우기",
        hint="긴 변에 맞춰 짧은 변을 채웁니다. 리사이즈해도 사진이 눌리지 않습니다.",
        summary="가로세로 비를 보존한 채 정사각으로 패딩한다.",
        scenario=(
            "정답에 화면 방향(가로/세로)이 들어가는데 리사이즈가 그것을 지워 버리는 경우. "
            "리사이즈 **앞에** 둔다."
        ),
    ),
)
class PadToSquare(Node):
    """**구현은 모듈 최상위에 둔다.** 함수 안에 정의하면 Windows spawn 워커가
    임포트할 수 없고, 등록 시점에 거부된다.

    순수 함수다 — 파일·네트워크·전역 난수·시계에 손대지 않는다. 같은 사진에는 언제나
    같은 결과를 낸다. 그래야 캐시가 거짓말을 하지 않는다.
    """

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        arr = np.asarray(inputs["image"])
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise NodeError(
                ctx.node_id,
                cause=f"HWC 3채널 이미지가 와야 하는데 모양이 {arr.shape} 다",
                sample_key=ctx.sample_key,
                hint=(
                    "  안 잡혔다면: 채널이 뒤섞인 배열이 그대로 학습에 들어가고,\n"
                    "  손실은 내려가는데 모델은 엉뚱한 것을 본다.\n"
                    "  추정 낭비: 학습 시간 전부.\n"
                    "  앞에 `adapt.image_layout` 을 넣어 HWC 로 맞춰라."
                ),
            )
        fill = list(params.fill or [0, 0, 0])
        if len(fill) != 3:
            raise NodeError(
                ctx.node_id,
                cause=f"fill 은 RGB 세 값이어야 하는데 {len(fill)}개다",
                hint="  예: fill: [0, 0, 0] (검정) 또는 [255, 255, 255] (흰색).",
            )

        h, w = arr.shape[0], arr.shape[1]
        side = max(h, w)
        if side == h == w:
            return {"image": arr}

        out = np.empty((side, side, 3), dtype=arr.dtype)
        out[:, :] = np.asarray(fill, dtype=arr.dtype)
        top, left = (side - h) // 2, (side - w) // 2
        out[top : top + h, left : left + w] = arr
        return {"image": out}
