"""물질화 산출물 리더.

Trainer는 원본 데이터도, 그래프도 다시 읽지 않는다. 여기만 읽는다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

MANIFEST = "manifest.jsonl"


@dataclass
class ShardStats:
    shards: int = 0
    samples: int = 0
    images: int = 0
    prompt_chars: int = 0
    answer_chars: int = 0
    max_images_per_sample: int = 0

    @property
    def avg_chars(self) -> float:
        return (self.prompt_chars + self.answer_chars) / self.samples if self.samples else 0.0


def manifest(out_dir: str) -> List[Dict[str, Any]]:
    p = os.path.join(out_dir, MANIFEST)
    if not os.path.exists(p):
        return []
    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def iter_samples(out_dir: str, with_images: bool = False) -> Iterator[Dict[str, Any]]:
    """커밋된 shard만 읽는다. 매니페스트에 없는 파일은 없는 것으로 취급한다."""
    for entry in manifest(out_dir):
        name = entry["shard"]
        path = os.path.join(out_dir, name + ".jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_shard"] = name
                rec["_dir"] = os.path.join(out_dir, name)
                if with_images:
                    rec["_images"] = [_load(os.path.join(out_dir, name, r)) for r in rec.get("images", [])]
                yield rec


def _load(path: str):
    import numpy as np
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype="uint8")


def stats(out_dir: str) -> ShardStats:
    s = ShardStats(shards=len(manifest(out_dir)))
    for rec in iter_samples(out_dir):
        s.samples += 1
        n = len(rec.get("images") or [])
        s.images += n
        s.max_images_per_sample = max(s.max_images_per_sample, n)
        s.prompt_chars += len(rec.get("prompt", ""))
        s.answer_chars += len(rec.get("answer", ""))
    return s
