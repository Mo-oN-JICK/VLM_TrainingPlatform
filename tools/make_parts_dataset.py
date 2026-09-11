"""합성 부품 검사 데이터 — VLM 전용 예시용.

이미지 한 장과 라벨 하나. 시계열도 전문가 모델도 없다.
`dummy_ecg`와 **일부러 다른 모양**으로 만든 것이다 — 플랫폼이 그래프 하나에 맞춰진
도구가 아니라는 것을 확인하려면 두 번째 도메인이 필요하다.

    python tools/make_parts_dataset.py --n 48
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw

W, H = 320, 240
DEFECTS = ("scratch", "dent", "stain")
KO = {"scratch": "긁힘", "dent": "찍힘", "stain": "얼룩", "": "없음"}


def draw_part(rng: random.Random, defect: str) -> Image.Image:
    """회색 판 위에 부품 하나. 결함이 있으면 그 자리에 표가 난다."""
    img = Image.new("RGB", (W, H), (34, 36, 40))
    d = ImageDraw.Draw(img)

    # 부품 몸통
    x0, y0 = rng.randint(40, 90), rng.randint(30, 70)
    x1, y1 = x0 + rng.randint(140, 180), y0 + rng.randint(90, 120)
    d.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=(150, 155, 162), outline=(200, 204, 210))
    for i in range(3):  # 나사 구멍
        cx = x0 + 22 + i * ((x1 - x0 - 44) // 2)
        d.ellipse([cx - 7, y0 + 18, cx + 7, y0 + 32], fill=(70, 74, 80))

    if defect == "scratch":
        sx = rng.randint(x0 + 20, x1 - 60)
        sy = rng.randint(y0 + 45, y1 - 20)
        d.line([sx, sy, sx + rng.randint(35, 60), sy + rng.randint(-8, 8)], fill=(235, 235, 240), width=2)
    elif defect == "dent":
        cx, cy = rng.randint(x0 + 30, x1 - 30), rng.randint(y0 + 45, y1 - 20)
        d.ellipse([cx - 11, cy - 9, cx + 11, cy + 9], fill=(96, 100, 108), outline=(60, 63, 70))
    elif defect == "stain":
        cx, cy = rng.randint(x0 + 30, x1 - 30), rng.randint(y0 + 45, y1 - 20)
        d.ellipse([cx - 16, cy - 12, cx + 16, cy + 12], fill=(120, 104, 78))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", default="solutions/vlm_parts/data/parts")
    ap.add_argument("--seed", type=int, default=20260911)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for i in range(a.n):
        defect = rng.choice(("",) * 2 + DEFECTS)  # 40%쯤 정상
        name = f"p{i:04d}"
        draw_part(rng, defect).save(os.path.join(img_dir, f"{name}.png"))
        rows.append(
            {
                "sample_id": name,
                "lot_id": f"lot{i % 6:02d}",       # 같은 로트가 train/val에 함께 들어가지 않게
                "image_path": f"images/{name}.png",
                "verdict": "불량" if defect else "정상",
                "defect": KO[defect],
                "part_name": rng.choice(("브래킷", "커버", "하우징")),
            }
        )

    index = os.path.join(a.out, "index.jsonl")
    with open(index, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    bad = sum(1 for r in rows if r["verdict"] == "불량")
    print(f"{index}: {len(rows)}건 (불량 {bad} · 정상 {len(rows) - bad})")
    print(f"  이미지 {img_dir}  {W}x{H}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
