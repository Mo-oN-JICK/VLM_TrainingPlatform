"""런타임 값의 해시와 실측 대조.

선언한 타입과 실제로 흐른 값이 같은지 재는 곳. dry-run(G3)의 핵심이며,
결정성 감사와 캐시 키도 여기의 해시를 쓴다.
"""

from __future__ import annotations

from hashlib import blake2b
from typing import Any, List

import numpy as np

from ..core.types import BaseKind, DType, PortType

_NP_DTYPE = {
    DType.U8: np.uint8,
    DType.I32: np.int32,
    DType.I64: np.int64,
    DType.F16: np.float16,
    DType.BF16: None,  # numpy에 없음 — dtype 검사를 건너뛴다
    DType.F32: np.float32,
    DType.F64: np.float64,
    DType.BOOL: np.bool_,
}

_PY_BASE = {
    BaseKind.TEXT: str,
    BaseKind.TABLE: dict,
    BaseKind.SAMPLE: dict,
    BaseKind.REPORT: dict,
    BaseKind.REGIONS: list,
}


def value_hash(v: Any) -> str:
    h = blake2b(digest_size=16)
    _feed(h, v)
    return h.hexdigest()


def _feed(h: Any, v: Any) -> None:
    if isinstance(v, np.ndarray):
        h.update(b"nd")
        h.update(str(v.dtype).encode())
        h.update(str(v.shape).encode())
        h.update(np.ascontiguousarray(v).tobytes())
    elif isinstance(v, (list, tuple)):
        h.update(b"[")
        for x in v:
            _feed(h, x)
        h.update(b"]")
    elif isinstance(v, dict):
        h.update(b"{")
        for k in sorted(v, key=str):
            h.update(str(k).encode())
            _feed(h, v[k])
        h.update(b"}")
    elif isinstance(v, (str, bytes)):
        h.update(v.encode() if isinstance(v, str) else v)
    elif v is None or isinstance(v, (int, float, bool)):
        h.update(repr(v).encode())
    else:
        digest = getattr(v, "digest", None)
        h.update(repr(digest() if callable(digest) else v).encode())


def describe(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return f"ndarray{list(v.shape)} {v.dtype}"
    if isinstance(v, list):
        return f"list[{len(v)}]" + (f" of {describe(v[0])}" if v else "")
    if isinstance(v, str):
        return f"str[{len(v)}]"
    if isinstance(v, dict):
        return f"dict{sorted(v)[:6]}"
    return type(v).__name__


def check_against(declared: PortType, value: Any, where: str = "") -> List[str]:
    """선언 타입과 실제 값의 불일치 목록. 잴 수 있는 것만 잰다.

    layout·norm·value_range·frame은 값에서 알아낼 수 없는 선언이다. 여기서는
    base kind, dtype, rank, 구체 shape, 리스트 길이를 확인한다.
    """
    out: List[str] = []
    p = f"{where}: " if where else ""

    if declared.list_of is not None:
        if not isinstance(value, (list, tuple)):
            return [f"{p}리스트가 와야 하는데 {describe(value)}"]
        n = len(value)
        if not (declared.list_of.min_n <= n <= declared.list_of.max_n):
            out.append(f"{p}리스트 길이 {n}이 선언 범위 {declared.list_of!r} 밖")
        if value:
            inner = PortType(**{**declared.__dict__, "list_of": None})
            out.extend(check_against(inner, value[0], where))
        return out

    base = declared.base
    if base in (BaseKind.IMAGE, BaseKind.TIME_SERIES, BaseKind.EMBEDDING, BaseKind.MASK, BaseKind.TOKENS):
        if not isinstance(value, np.ndarray):
            return [f"{p}배열이 와야 하는데 {describe(value)}"]
        want = _NP_DTYPE.get(declared.dtype) if isinstance(declared.dtype, DType) else None
        if want is not None and value.dtype != want:
            out.append(f"{p}dtype 선언 {declared.dtype.value}, 실제 {value.dtype}")
        if declared.shape is not None:
            if len(declared.shape) != value.ndim:
                out.append(f"{p}rank 선언 {len(declared.shape)}, 실제 {value.ndim}")
            else:
                for i, d in enumerate(declared.shape):
                    if isinstance(d, int) and value.shape[i] != d:
                        out.append(f"{p}shape[{i}] 선언 {d}, 실제 {value.shape[i]}")
    elif base in _PY_BASE:
        want_py = _PY_BASE[base]
        if not isinstance(value, want_py):
            out.append(f"{p}{want_py.__name__}이 와야 하는데 {describe(value)}")
        elif base is BaseKind.REGIONS:
            for i, r in enumerate(value[:3]):
                if not isinstance(r, dict) or "extent" not in r:
                    out.append(f"{p}regions[{i}]에 extent가 없다")
    elif base is BaseKind.SCHEMA:
        if not hasattr(value, "steps"):
            out.append(f"{p}AnswerSchema가 와야 하는데 {describe(value)}")
    return out
