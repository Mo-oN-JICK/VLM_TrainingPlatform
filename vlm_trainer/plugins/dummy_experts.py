"""더미 전문가 플러그인.

실제 모델 가중치가 없는 동안 배선과 타입을 검증하기 위한 것. 결정적이다 —
같은 입력이면 같은 지목을 낸다(입력 바이트에서 시드를 뽑는다).
실물 모델이 생기면 같은 인터페이스로 교체된다. 그래프는 바뀌지 않는다.
"""

from __future__ import annotations

import random
from hashlib import blake2b
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import Frame, image, regions, timeseries
from .base import ExpertManifest, ExpertPlugin, ResourceCost, register_expert


def _seed_of(arr: np.ndarray) -> int:
    h = blake2b(np.ascontiguousarray(arr).tobytes()[:65536], digest_size=8).digest()
    return int.from_bytes(h, "little")


@register_expert
class DummyRegionProposer(ExpertPlugin):
    """이미지에서 의심 영역을 지목한다."""

    manifest = ExpertManifest(
        id="dummy_region_proposer",
        version="0.1.0",
        domain="image2d",
        accepts=image(),
        produces=regions(domain="image2d", frame=Frame.ORIG_PX).with_sem("expert_hint"),
        deterministic=True,
        vram_mb=0,
    )

    def propose(self, subject: Any, params: Any, ctx: Any) -> List[Dict[str, Any]]:
        arr = np.asarray(subject)
        h, w = arr.shape[0], arr.shape[1]
        rng = random.Random(_seed_of(arr))
        out: List[Dict[str, Any]] = []
        for i in range(max(1, int(params.topk))):
            bw = rng.uniform(0.15, 0.35) * w
            bh = rng.uniform(0.2, 0.5) * h
            x0 = rng.uniform(0, w - bw)
            y0 = rng.uniform(0, h - bh)
            out.append(
                {
                    "extent": (x0, y0, x0 + bw, y0 + bh),
                    "score": round(1.0 - 0.17 * i, 4),
                    "label": "suspect",
                }
            )
        return out

    def cost(self, shape: Optional[Tuple], params: Any) -> ResourceCost:
        return ResourceCost(vram_mb=0, ms_per_sample=1.0, out_bytes_per_sample=256)


@register_expert
class DummyIntervalProposer(ExpertPlugin):
    """시계열에서 의심 구간을 지목한다. 이미지 쪽과 같은 인터페이스."""

    manifest = ExpertManifest(
        id="dummy_interval_proposer",
        version="0.1.0",
        domain="series1d",
        accepts=timeseries(),
        produces=regions(domain="series1d", frame=Frame.TS_SECONDS).with_sem("expert_hint"),
        deterministic=True,
    )

    def propose(self, subject: Any, params: Any, ctx: Any) -> List[Dict[str, Any]]:
        arr = np.asarray(subject)
        n = arr.shape[0]
        # 결정적 규칙: 채널 0의 |z| 최대 지점 주변을 구간으로 잡는다
        ch = arr[:, 0].astype(np.float64)
        mu, sd = float(ch.mean()), float(ch.std()) or 1.0
        z = np.abs((ch - mu) / sd)
        idx = np.argsort(-z)[: max(1, int(params.topk)) * 8]
        picked: List[int] = []
        for i in sorted(int(x) for x in idx):
            if all(abs(i - p) > n * 0.08 for p in picked):
                picked.append(i)
            if len(picked) >= int(params.topk):
                break
        hz = float(getattr(params, "hz", 0) or 0) or 1.0
        half = max(1, int(n * 0.03))
        return [
            {
                "extent": (max(0, i - half) / hz, min(n, i + half) / hz),
                "score": round(float(z[i]) / (float(z.max()) or 1.0), 4),
                "label": "spike",
            }
            for i in picked
        ]
