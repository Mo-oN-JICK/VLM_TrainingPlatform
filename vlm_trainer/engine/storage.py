"""캐시 저장 백엔드 — 바이트를 어디에 두는가만 답한다.

**키 계산은 여기 없다.** 무엇이 바뀌면 다시 계산해야 하는가는 `CacheStore`가 정하고,
백엔드는 그 키가 가리키는 자리에 바이트를 넣고 꺼낼 뿐이다. 이 경계가 흐려지는 순간
백엔드를 갈아 끼운 것만으로 캐시가 통째로 무효가 되거나, 더 나쁘게는 서로 다른 계산이
같은 키를 공유한다.

기본은 로컬 파일이다. 원격 저장소나 객체 스토리지는 이 프로토콜을 구현하면 된다.
설계 문서 08, Phase 8.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, Iterator, Optional, Protocol, Type

from ..core.errors import RegistrationError


class StorageBackend(Protocol):
    """키 하나에 바이트 하나. 그 이상을 알지 않는다."""

    def read(self, key: str) -> Optional[bytes]:
        """없으면 None. 깨진 것도 None이다 — 부르는 쪽은 다시 계산하면 된다."""

    def write(self, key: str, data: bytes) -> None:
        """원자적으로. 중간에 죽어도 반쪽짜리가 남으면 안 된다."""

    def delete_prefix(self, prefix: str) -> int:
        """지운 개수를 돌려준다."""

    def count(self, prefix: str = "") -> int:
        ...


class LocalFiles:
    """기본 백엔드. 키를 경로로 쓴다.

    Windows 경로 길이 상한 때문에 키는 짧은 해시여야 하는데, 그것을 정하는 것은
    `CacheStore`의 몫이다. 여기서는 받은 키를 그대로 경로로 쓴다.
    """

    def __init__(self, root: str) -> None:
        self.root = root

    def _path(self, key: str) -> str:
        return os.path.join(self.root, *key.split("/"))

    def read(self, key: str) -> Optional[bytes]:
        try:
            with open(self._path(key), "rb") as fh:
                return fh.read()
        except OSError:
            return None

    def write(self, key: str, data: bytes) -> None:
        p = self._path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, p)  # 원자 교체 — Windows에서 부분 파일이 남지 않는다

    def delete_prefix(self, prefix: str) -> int:
        d = self._path(prefix) if prefix else self.root
        if not os.path.isdir(d):
            return 0
        n = sum(len(files) for _, _, files in os.walk(d))
        shutil.rmtree(d, ignore_errors=True)
        return n

    def count(self, prefix: str = "") -> int:
        d = self._path(prefix) if prefix else self.root
        if not os.path.isdir(d):
            return 0
        return sum(len(files) for _, _, files in os.walk(d))


class InMemory:
    """프로세스가 살아 있는 동안만. 테스트와 일회성 실행용."""

    def __init__(self, root: str = "") -> None:
        self.root = root
        self.blobs: Dict[str, bytes] = {}

    def read(self, key: str) -> Optional[bytes]:
        return self.blobs.get(key)

    def write(self, key: str, data: bytes) -> None:
        self.blobs[key] = data

    def delete_prefix(self, prefix: str) -> int:
        hit = [k for k in self.blobs if not prefix or k.startswith(prefix)]
        for k in hit:
            del self.blobs[k]
        return len(hit)

    def count(self, prefix: str = "") -> int:
        return sum(1 for k in self.blobs if not prefix or k.startswith(prefix))


_BACKENDS: Dict[str, Type] = {"local": LocalFiles, "memory": InMemory}


def register_backend(name: str, cls: Type) -> Type:
    if name in _BACKENDS and _BACKENDS[name] is not cls:
        raise RegistrationError(f"저장 백엔드 이름이 겹친다: {name!r}")
    _BACKENDS[name] = cls
    return cls


def resolve_backend(name: str, root: str) -> StorageBackend:
    if name not in _BACKENDS:
        raise RegistrationError(
            f"알 수 없는 저장 백엔드 {name!r} (있는 것: {sorted(_BACKENDS)})"
        )
    return _BACKENDS[name](root)


def all_backends() -> Iterator[str]:
    return iter(sorted(_BACKENDS))
