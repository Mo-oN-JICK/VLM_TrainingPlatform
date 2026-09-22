"""Time Series Processing.

Mech-Vision의 3D 카테고리 자리에 우리의 두 번째 모달이 들어간다.
ts.plot이 시계열을 VLM에 넣기 위한 표준 경로다 — 렌더 결과가 Image 타입이 되므로
이후 어댑터 체인과 그대로 결합된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..core.node import Node, NodeDoc, NodeKind, Port, RunCtx
from ..core.registry import register
from ..core.types import (
    ANY,
    BaseKind,
    PortType,
    image,
    regions,
    simple,
    timeseries,
)


@dataclass
class StatsParams:
    z_thresh: float = 3.0
    channel: int = 0


@register(
    type="ts.stats",
    version="1.0.0",
    category="Time Series Processing",
    kind=NodeKind.PROCESSING,
    inputs={"series": Port(timeseries(hz=ANY), "센서 시계열")},
    outputs={"stats": Port(simple(BaseKind.TABLE), "추세·주기·스파이크 통계")},
    params=StatsParams,
    recipe_overridable=["z_thresh", "channel"],
    preview="table",
    doc=NodeDoc(
        label="시계열 특징 추출",
            hint="추세·주기·이상값 개수를 계산합니다.",
        summary="추세 기울기, 자기상관 주기, 스파이크 개수를 계산한다.",
        scenario="정답 Text의 단계별 값과 근거 문장이 전부 이 수치에서 나온다.",
    ),
)
class TsStats(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        arr = np.asarray(inputs["series"], dtype=np.float64)
        ch = arr[:, int(params.channel)]
        n = len(ch)
        t = np.arange(n, dtype=np.float64)
        slope = float(np.polyfit(t, ch, 1)[0]) if n > 2 else 0.0

        c = ch - ch.mean()
        denom = float((c * c).sum()) or 1.0
        acf = np.correlate(c, c, mode="full")[n - 1 :] / denom
        lo = max(2, n // 100)
        seg = acf[lo : max(lo + 1, n // 2)]
        peak_i = int(np.argmax(seg)) + lo if len(seg) else 0
        acf_peak = float(seg.max()) if len(seg) else 0.0

        sd = float(ch.std()) or 1.0
        z = np.abs((ch - ch.mean()) / sd)
        spikes = z > float(params.z_thresh)
        # 인접 표본을 하나의 스파이크로 묶는다
        count = int(np.count_nonzero(spikes & ~np.concatenate(([False], spikes[:-1]))))

        return {
            "stats": {
                "n": n,
                "trend_slope": round(slope * 1000, 4),  # 1000 표본당 변화량
                "acf_peak": round(acf_peak, 4),
                "acf_period_n": peak_i,
                "spike_count": count,
                "spike_max_z": round(float(z.max()) if n else 0.0, 4),
                "z_thresh": float(params.z_thresh),
                "mean": round(float(ch.mean()), 4),
                "std": round(sd, 4),
            }
        }


@dataclass
class PlotParams:
    size: tuple = (640, 320)
    channel: int = 0
    mark_regions: bool = True
    line_width: int = 1
    hz: float = 0.0  # 구간(초)을 x축에 매핑하려면 필요. 0이면 구간을 그리지 않는다


@register(
    type="ts.plot",
    version="1.0.0",
    category="Time Series Processing",
    kind=NodeKind.PROCESSING,
    inputs={
        "series": Port(timeseries(hz=ANY), "센서 시계열"),
        "regions": Port(regions(domain="series1d", frame=ANY).as_optional(), "표시할 구간(선택)"),
    },
    outputs={"plot": Port(image(shape=(320, 640, 3)).with_sem("raw_image"), "렌더된 파형 이미지")},
    params=PlotParams,
    recipe_overridable=["size", "channel", "mark_regions", "hz"],
    type_affecting=["size"],
    preview="image",
    doc=NodeDoc(
        label="시계열 렌더링",
            hint="시계열을 이미지로 그려 VLM 입력으로 만듭니다.",
        summary="시계열을 이미지로 렌더한다. VLM에 시계열을 넣는 표준 경로.",
        scenario="렌더 결과가 Image 타입이므로 이후 crop·resize 체인과 그대로 결합된다.",
    ),
)
class TsPlot(Node):
    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        w, h = int(params.size[0]), int(params.size[1])
        return {"plot": image(shape=(h, w, 3)).with_sem("raw_image")}

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        from PIL import Image as PILImage
        from PIL import ImageDraw

        arr = np.asarray(inputs["series"], dtype=np.float64)
        ch = arr[:, int(params.channel)]
        w, h = int(params.size[0]), int(params.size[1])
        im = PILImage.new("RGB", (w, h), (255, 255, 255))
        d = ImageDraw.Draw(im)

        lo, hi = float(ch.min()), float(ch.max())
        span = (hi - lo) or 1.0
        pad = 8
        n = len(ch)
        xs = np.linspace(pad, w - pad, num=min(n, w * 2))
        idx = np.linspace(0, n - 1, num=len(xs)).astype(int)
        ys = (h - pad) - ((ch[idx] - lo) / span) * (h - 2 * pad)

        if params.mark_regions and params.hz and inputs.get("regions"):
            duration = n / float(params.hz)
            for r in inputs["regions"]:
                t0, t1 = (float(x) for x in r["extent"])
                x0 = pad + (t0 / duration) * (w - 2 * pad)
                x1 = pad + (t1 / duration) * (w - 2 * pad)
                d.rectangle([x0, pad, max(x0 + 1, x1), h - pad], fill=(255, 236, 236))

        d.line([(0, h // 2), (w, h // 2)], fill=(220, 220, 220), width=1)
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=(20, 60, 90), width=int(params.line_width))
        return {"plot": np.asarray(im, dtype=np.uint8)}
