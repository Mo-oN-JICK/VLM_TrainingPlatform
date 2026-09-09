"""2D General Processing와 Adapters.

암묵 변환 금지 정책의 실행 수단이 여기 있다. 리사이즈·좌표계 변환은 전부 명시적 노드다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.node import Node, NodeDoc, NodeKind, Port, RunCtx
from ..core.registry import register
from ..core.types import (
    ANY,
    DYN,
    BaseKind,
    Color,
    DType,
    Frame,
    Layout,
    Norm,
    PortType,
    Range,
    Var,
    image,
    regions,
    simple,
)


@dataclass
class ResizeParams:
    size: tuple = (448, 448)
    mode: str = "bilinear"
    keep_aspect: bool = True
    pad_value: int = 0


@register(
    type="adapt.image_resize",
    version="1.0.0",
    category="Adapters",
    kind=NodeKind.PROCESSING,
    inputs={"image": Port(image(shape=(DYN, DYN, 3)), "임의 크기 이미지")},
    outputs={"image": Port(image(shape=(448, 448, 3)), "고정 크기 이미지")},
    params=ResizeParams,
    recipe_overridable=["size", "mode", "keep_aspect"],
    type_affecting=["size"],
    preview="image",
    doc=NodeDoc(
        summary="가변 크기를 고정 크기로 바꾼다.",
        scenario="배치 조립과 자원 예산 산정은 고정 크기를 요구한다. 가변(dyn) 차원은 여기서만 사라진다.",
    ),
)
class ImageResize(Node):
    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        src = inputs.get("image")
        h, w = int(params.size[0]), int(params.size[1])
        if src is None:
            return {"image": image(shape=(h, w, 3))}
        c = src.shape[2] if src.shape and len(src.shape) == 3 else 3
        return {"image": replace(src, shape=(h, w, c))}

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        from PIL import Image as PILImage

        arr = np.asarray(inputs["image"])
        h, w = int(params.size[0]), int(params.size[1])
        resample = {"bilinear": PILImage.BILINEAR, "bicubic": PILImage.BICUBIC, "nearest": PILImage.NEAREST}[
            params.mode
        ]
        im = PILImage.fromarray(arr)
        if params.keep_aspect:
            im.thumbnail((w, h), resample)
            canvas = PILImage.new("RGB", (w, h), (params.pad_value,) * 3)
            canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
            im = canvas
        else:
            im = im.resize((w, h), resample)
        return {"image": np.asarray(im, dtype=np.uint8)}


@dataclass
class FrameParams:
    to: str = "orig_px"
    clip_to_bounds: bool = True
    ref_width: int = 0
    ref_height: int = 0


@register(
    type="adapt.frame",
    version="1.0.0",
    category="Adapters",
    kind=NodeKind.PROCESSING,
    inputs={"regions": Port(regions(domain="image2d", frame=ANY), "임의 좌표계의 영역")},
    outputs={"regions": Port(regions(domain="image2d", frame=Frame.ORIG_PX), "지정 좌표계의 영역")},
    params=FrameParams,
    recipe_overridable=["to"],
    type_affecting=["to"],
    preview="regions_overlay",
    doc=NodeDoc(
        summary="영역의 좌표계 기준을 바꾼다.",
        scenario="좌표계 불일치는 조용히 엉뚱한 영역을 잘라낸다. 변환은 반드시 명시적이어야 한다.",
    ),
)
class FrameAdapter(Node):
    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        return {"regions": regions(domain="image2d", frame=Frame(params.to)).with_sem("expert_hint")}

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        w, h = float(params.ref_width or 1), float(params.ref_height or 1)
        out = []
        for r in inputs["regions"]:
            x0, y0, x1, y1 = r["extent"]
            if params.to == "norm01":
                x0, x1, y0, y1 = x0 / w, x1 / w, y0 / h, y1 / h
            elif params.to == "orig_px" and max(x0, x1, y0, y1) <= 1.0:
                x0, x1, y0, y1 = x0 * w, x1 * w, y0 * h, y1 * h
            if params.clip_to_bounds and params.to == "orig_px":
                x0, x1 = max(0.0, x0), min(w, x1)
                y0, y1 = max(0.0, y0), min(h, y1)
            out.append({**r, "extent": (x0, y0, x1, y1)})
        return {"regions": out}


@dataclass
class ImageFrameParams:
    to: str = "orig_px"


@register(
    type="adapt.image_frame",
    version="1.0.0",
    category="Adapters",
    kind=NodeKind.PROCESSING,
    inputs={"image": Port(image(frame=ANY), "임의 좌표 기준의 이미지")},
    outputs={"image": Port(image(frame=Frame.ORIG_PX), "기준을 다시 선언한 이미지")},
    params=ImageFrameParams,
    recipe_overridable=["to"],
    type_affecting=["to"],
    preview="image",
    doc=NodeDoc(
        summary="이미지 픽셀 격자의 좌표 기준을 다시 선언한다. 픽셀은 그대로 두고 타입만 바꾼다.",
        scenario="crop을 그 자체로 하나의 이미지로 취급해 원본과 한 리스트에 담을 때. "
        "암묵 변환을 금지했으므로 이 선언도 노드로 남아 스펙에 기록된다.",
    ),
)
class ImageFrameAdapter(Node):
    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        src = inputs.get("image")
        f = Frame(params.to)
        return {"image": image(frame=f) if src is None else replace(src, frame=f)}

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"image": inputs["image"]}


@dataclass
class CropParams:
    padding_ratio: float = 0.15
    square_pad: bool = True
    max_n: int = 4
    min_side_px: int = 16


@register(
    type="image.crop_by_regions",
    version="1.0.0",
    category="2D General Processing",
    kind=NodeKind.PROCESSING,
    inputs={
        "image": Port(image(), "원본 이미지"),
        "regions": Port(regions(domain="image2d", frame=Frame.ORIG_PX), "잘라낼 영역"),
    },
    outputs={
        "crops": Port(image(frame=Frame.CROP_PX).as_list(1, 16).with_sem("roi_crop"), "영역별 crop")
    },
    params=CropParams,
    recipe_overridable=["padding_ratio", "max_n", "square_pad"],
    preview="image_grid",
    doc=NodeDoc(summary="지목된 영역을 원본에서 잘라낸다."),
)
class CropByRegions(Node):
    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        src = inputs.get("image")
        base = image(frame=Frame.CROP_PX) if src is None else replace(src, frame=Frame.CROP_PX, shape=(DYN, DYN, 3))
        return {"crops": base.with_sem("roi_crop").as_list(1, max(1, int(params.max_n)))}

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        arr = np.asarray(inputs["image"])
        h, w = arr.shape[0], arr.shape[1]
        crops: List[np.ndarray] = []
        for r in list(inputs["regions"])[: int(params.max_n)]:
            x0, y0, x1, y1 = (float(v) for v in r["extent"])
            pw, ph = (x1 - x0) * params.padding_ratio, (y1 - y0) * params.padding_ratio
            x0, x1, y0, y1 = x0 - pw, x1 + pw, y0 - ph, y1 + ph
            if params.square_pad:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                side = max(x1 - x0, y1 - y0) / 2
                x0, x1, y0, y1 = cx - side, cx + side, cy - side, cy + side
            xi0, yi0 = max(0, int(round(x0))), max(0, int(round(y0)))
            xi1, yi1 = min(w, int(round(x1))), min(h, int(round(y1)))
            if xi1 - xi0 < params.min_side_px or yi1 - yi0 < params.min_side_px:
                continue
            crops.append(arr[yi0:yi1, xi0:xi1].copy())
        if not crops:  # 리스트 하한이 1이므로 빈 결과는 계약 위반이다
            crops = [arr.copy()]
        return {"crops": crops}
