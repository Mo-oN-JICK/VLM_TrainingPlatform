"""중간 산출물 캐시.

무엇이 바뀌면 다시 계산해야 하는가에 대한 유일한 답.
경로는 Windows 경로 길이 상한 때문에 서술적 이름 대신 짧은 해시 키를 쓴다.
"""

from __future__ import annotations

import os
import pickle
import shutil
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DIR = ".cache"


def sample_key_hash(*parts: str) -> str:
    h = blake2b(digest_size=8)
    for p in parts:
        h.update(p.encode())
        h.update(b"|")
    return h.hexdigest()


@dataclass
class CacheStore:
    root: str = DEFAULT_DIR
    enabled: bool = True
    hits: int = 0
    misses: int = 0
    writes: int = 0

    def path_for(self, node_key: str, sample_hash: str) -> str:
        k = node_key.split(":")[-1]  # "b2:abcd..." 에서 해시 부분만
        return os.path.join(self.root, k[:2], k[2:18], f"{sample_hash}.pkl")

    def get(self, node_key: str, sample_hash: str) -> Tuple[bool, Any]:
        if not self.enabled:
            return False, None
        p = self.path_for(node_key, sample_hash)
        if not os.path.exists(p):
            self.misses += 1
            return False, None
        try:
            with open(p, "rb") as fh:
                v = pickle.load(fh)
            self.hits += 1
            return True, v
        except Exception:
            self.misses += 1
            return False, None

    def put(self, node_key: str, sample_hash: str, value: Any) -> None:
        if not self.enabled:
            return
        p = self.path_for(node_key, sample_hash)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)  # 원자 교체 — Windows에서 부분 파일이 남지 않는다
        self.writes += 1

    def clear(self, node_key: Optional[str] = None) -> int:
        if not os.path.isdir(self.root):
            return 0
        if node_key is None:
            n = sum(len(files) for _, _, files in os.walk(self.root))
            shutil.rmtree(self.root, ignore_errors=True)
            return n
        k = node_key.split(":")[-1]
        d = os.path.join(self.root, k[:2], k[2:18])
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        shutil.rmtree(d, ignore_errors=True)
        return n

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}
