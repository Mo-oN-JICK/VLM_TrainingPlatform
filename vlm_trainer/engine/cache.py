"""중간 산출물 캐시.

무엇이 바뀌면 다시 계산해야 하는가에 대한 유일한 답.
경로는 Windows 경로 길이 상한 때문에 서술적 이름 대신 짧은 해시 키를 쓴다.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Dict, Optional, Tuple

from .storage import resolve_backend

DEFAULT_DIR = ".cache"


def sample_key_hash(*parts: str) -> str:
    h = blake2b(digest_size=8)
    for p in parts:
        h.update(p.encode())
        h.update(b"|")
    return h.hexdigest()


@dataclass
class CacheStore:
    """무엇이 바뀌면 다시 계산해야 하는가에 대한 유일한 답.

    **키는 여기서만 정해진다.** 백엔드는 그 키에 바이트를 넣고 꺼낼 뿐이라, 저장소를
    갈아 끼워도 캐시 키는 한 글자도 달라지지 않는다. 이 경계가 흐려지면 백엔드 교체가
    캐시를 통째로 무효로 만들거나, 더 나쁘게는 서로 다른 계산이 같은 키를 공유한다.
    """

    root: str = DEFAULT_DIR
    enabled: bool = True
    backend: str = "local"
    hits: int = 0
    misses: int = 0
    writes: int = 0
    store: Any = None

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = resolve_backend(self.backend, self.root)

    def key_for(self, node_key: str, sample_hash: str) -> str:
        """캐시 키. 백엔드와 무관하다 — 이 함수의 결과가 곧 캐시의 정체성이다."""
        k = node_key.split(":")[-1]  # "b2:abcd..." 에서 해시 부분만
        return f"{k[:2]}/{k[2:18]}/{sample_hash}.pkl"

    def path_for(self, node_key: str, sample_hash: str) -> str:
        """로컬 백엔드에서의 실제 경로. 진단용이다."""
        return os.path.join(self.root, *self.key_for(node_key, sample_hash).split("/"))

    def get(self, node_key: str, sample_hash: str) -> Tuple[bool, Any]:
        if not self.enabled:
            return False, None
        raw = self.store.read(self.key_for(node_key, sample_hash))
        if raw is None:
            self.misses += 1
            return False, None
        try:
            v = pickle.loads(raw)
        except Exception:
            self.misses += 1
            return False, None
        self.hits += 1
        return True, v

    def put(self, node_key: str, sample_hash: str, value: Any) -> None:
        if not self.enabled:
            return
        self.store.write(
            self.key_for(node_key, sample_hash),
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
        )
        self.writes += 1

    def clear(self, node_key: Optional[str] = None) -> int:
        if node_key is None:
            return self.store.delete_prefix("")
        k = node_key.split(":")[-1]
        return self.store.delete_prefix(f"{k[:2]}/{k[2:18]}")

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}
