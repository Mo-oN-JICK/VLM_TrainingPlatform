"""합성 더미 데이터 생성기.

실데이터가 없는 동안 배선·타입·엔진을 검증하기 위한 것. 성질을 심어서 만든다 —
추세, 주기, 스파이크를 파라미터로 넣으므로 정답 Text가 맞게 생성되는지 확인할 수 있다.
이미지 크기는 샘플마다 다르게 만든다. 가변(dyn) 차원 경로를 실제로 태우기 위해서다.

    python tools/make_dummy_dataset.py --n 24
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_OUT = os.path.join("solutions", "dummy_ecg", "data", "dummy")


def make_series(rng: random.Random, n: int, hz: float, channels: int) -> Dict[str, Any]:
    trend = rng.choice([0.0, 0.0, 0.6, -0.6])
    period_s = rng.choice([0.0, 0.8, 1.2, 2.0])
    n_spikes = rng.choice([0, 0, 1, 2, 3])
    t = np.arange(n, dtype=np.float64) / hz

    base = trend * t
    if period_s:
        base = base + 1.5 * np.sin(2 * math.pi * t / period_s)
    cols = []
    for c in range(channels):
        noise = np.asarray([rng.gauss(0, 0.25) for _ in range(n)])
        cols.append(base * (1.0 - 0.15 * c) + noise)
    arr = np.stack(cols, axis=1)

    spike_at: List[int] = []
    for _ in range(n_spikes):
        i = rng.randrange(int(n * 0.1), int(n * 0.9))
        spike_at.append(i)
        arr[i : i + 2, :] += rng.choice([-1, 1]) * rng.uniform(6.0, 9.0)

    return {
        "array": arr.astype(np.float32),
        "truth": {
            "trend": "rising" if trend > 0.05 else ("falling" if trend < -0.05 else "flat"),
            "period_s": period_s,
            "n_spikes": n_spikes,
            "spike_at": spike_at,
        },
    }


def make_image(rng: random.Random, truth: Dict[str, Any]) -> Image.Image:
    w = rng.choice([480, 560, 640, 720])
    h = rng.choice([320, 360, 400])
    im = Image.new("RGB", (w, h), (250, 250, 248))
    d = ImageDraw.Draw(im)
    for gx in range(0, w, 40):
        d.line([(gx, 0), (gx, h)], fill=(232, 232, 230))
    for gy in range(0, h, 40):
        d.line([(0, gy), (w, gy)], fill=(232, 232, 230))
    # 스파이크마다 눈에 보이는 표식을 남긴다 — 전문가 더미가 지목할 대상
    for i in range(max(1, truth["n_spikes"])):
        cx, cy = rng.randrange(60, w - 60), rng.randrange(60, h - 60)
        r = rng.randrange(18, 40)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(190, 60, 60), width=3)
    d.line([(0, h // 2), (w, h // 2)], fill=(120, 140, 160), width=2)
    return im


def main() -> int:
    ap = argparse.ArgumentParser(description="합성 더미 데이터셋 생성")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--patients", type=int, default=6)
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "series"), exist_ok=True)
    n_samples = int(a.hz * a.seconds)

    rows: List[Dict[str, Any]] = []
    for i in range(a.n):
        rng = random.Random(a.seed + i)  # 샘플마다 결정적
        sid = f"s{i:04d}"
        s = make_series(rng, n_samples, a.hz, a.channels)
        series_rel = os.path.join("series", f"{sid}.csv")
        np.savetxt(os.path.join(out, series_rel), s["array"], delimiter=",", fmt="%.5f")

        img = make_image(rng, s["truth"])
        image_rel = os.path.join("images", f"{sid}.png")
        img.save(os.path.join(out, image_rel))

        rows.append(
            {
                "sample_id": sid,
                "patient_id": f"p{i % a.patients:02d}",
                "image_path": image_rel.replace("\\", "/"),
                "series_path": series_rel.replace("\\", "/"),
                "label": "abnormal" if s["truth"]["n_spikes"] > 0 else "normal",
                "quality_flag": "ok" if i % 17 else "bad",
                "truth_trend": s["truth"]["trend"],
                "truth_period_s": s["truth"]["period_s"],
                "truth_spikes": s["truth"]["n_spikes"],
                "image_w": img.width,
                "image_h": img.height,
            }
        )

    with open(os.path.join(out, "index.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_abn = sum(1 for r in rows if r["label"] == "abnormal")
    print(f"생성: {out}")
    print(f"  샘플 {len(rows)}건 (abnormal {n_abn} / normal {len(rows) - n_abn})")
    print(f"  시계열 {n_samples}표본 x {a.channels}채널 @ {a.hz}Hz")
    print(f"  이미지 크기는 샘플마다 다르다 — 가변 차원 경로를 태우기 위해서다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
