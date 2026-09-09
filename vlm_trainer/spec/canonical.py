"""canonical form과 해시.

canonical spec의 해시가 곧 "같은 실험"의 정의다. 설계 문서 04 §4.3.
설계 문서는 blake3를 적었지만 표준 라이브러리만으로 돌리기 위해 blake2b(16바이트)를 쓴다.
해시 알고리즘은 engine ABI에 포함되므로 나중에 바꾸면 캐시가 전부 무효화된다.
"""

from __future__ import annotations

import json
from enum import Enum
from hashlib import blake2b
from typing import Any, Dict

HASH_PREFIX = "b2"


def canon_value(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, dict):
        return {str(k): canon_value(v[k]) for k in sorted(v, key=str)}
    if isinstance(v, (list, tuple)):
        return [canon_value(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(canon_value(x) for x in v)
    if isinstance(v, float):
        return float(repr(v))
    if v is None or isinstance(v, (str, int, bool)):
        return v
    return str(v)


def canon_params(defaults: Dict[str, Any], given: Dict[str, Any]) -> Dict[str, Any]:
    """기본값을 생략하지 않고 전부 기록한다.

    노드 버전이 올라가며 기본값이 바뀌어도 과거 실험이 재현되어야 하기 때문이다.
    """
    merged = dict(defaults)
    merged.update(given or {})
    return canon_value(merged)


def dumps(obj: Any) -> str:
    return json.dumps(canon_value(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_obj(obj: Any) -> str:
    return HASH_PREFIX + ":" + blake2b(dumps(obj).encode("utf-8"), digest_size=16).hexdigest()


def hash_parts(*parts: Any) -> str:
    return hash_obj(list(parts))
