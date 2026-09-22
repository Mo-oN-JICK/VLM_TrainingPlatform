"""포트 타입 구조체.

포트 타입은 이름표가 아니라 필드의 곱이다. 설계 문서 02 참조.
이 모듈이 "타입이 무엇인가"에 대한 유일한 정의이며, 다른 모듈은 필드를 임의로 늘리지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional, Union


class BaseKind(str, Enum):
    """값의 근본 종류. 서브타입도 상속도 없다."""

    IMAGE = "Image"
    TIME_SERIES = "TimeSeries"
    TEXT = "Text"
    TOKENS = "Tokens"
    EMBEDDING = "Embedding"
    REGIONS = "Regions"
    MASK = "Mask"
    TABLE = "Table"
    SCHEMA = "Schema"
    SAMPLE = "Sample"
    REPORT = "Report"
    MODEL = "Model"


# 리스트는 별도 base kind가 아니라 list_of로만 표현한다.
# 표시할 때만 ImageList / TextList 처럼 붙여 쓴다.
_LIST_DISPLAY = {BaseKind.IMAGE: "ImageList", BaseKind.TEXT: "TextList"}


class DType(str, Enum):
    U8 = "u8"
    I32 = "i32"
    I64 = "i64"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"
    BOOL = "bool"
    STR = "str"


class Layout(str, Enum):
    HWC = "HWC"
    CHW = "CHW"
    TC = "TC"  # [time, channel]
    CT = "CT"


class Range(str, Enum):
    U8_0_255 = "0-255"
    UNIT_0_1 = "0-1"
    SIGNED_M1_1 = "-1..1"
    ZSCORE = "zscore"
    RAW = "raw"


class Norm(str, Enum):
    NONE = "none"
    SCALE01 = "scale01"
    IMAGENET = "imagenet"
    PER_CHANNEL_Z = "per_channel_z"
    CUSTOM = "custom"


class Color(str, Enum):
    RGB = "RGB"
    BGR = "BGR"
    GRAY = "GRAY"


class Frame(str, Enum):
    """좌표계 / 시간축 기준."""

    ORIG_PX = "orig_px"
    CROP_PX = "crop_px"
    TILE_PX = "tile_px"
    NORM01 = "norm01"
    TS_INDEX = "ts_index"
    TS_SECONDS = "ts_seconds"


class _Any:
    """목적지 포트가 그 필드에 '무관함'을 계약으로 선언한 것.

    소스가 무엇이든 통과시키되, 이는 노드가 그 필드에 의존하지 않겠다는 약속이다.
    소스 쪽 ANY는 '알 수 없음'이 아니라 같은 무관함이며 하류로 전파되지 않는다.
    """

    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:
        return "ANY"

    def __bool__(self) -> bool:  # pragma: no cover - 실수 방지용
        return True


ANY = _Any()


class _Dyn:
    """샘플마다 달라지는 차원. '아직 모르는 값'이 아니라 '가변으로 확정된 값'이다.

    제네릭 변수와 달리 compile을 통과한다. 고정 크기를 요구하는 하류(예: 배치 조립,
    자원 예산 산정)는 DYN을 거부하고 adapt.image_resize 같은 노드를 요구한다.
    """

    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:
        return "dyn"


DYN = _Dyn()


@dataclass(frozen=True)
class Var:
    """스칼라 필드용 타입 변수. compile 종료 시점에 전부 확정되어야 한다."""

    name: str

    def __repr__(self) -> str:
        return f"?{self.name}"


@dataclass(frozen=True)
class DimVar:
    """shape 차원용 심볼 변수."""

    name: str

    def __repr__(self) -> str:
        return f"?{self.name}"


Dim = Union[int, DimVar, _Any, _Dyn]


@dataclass(frozen=True)
class TimeBase:
    hz: Union[float, Var, _Any]
    t0_policy: str = "sample_start"  # sample_start | absolute

    def __repr__(self) -> str:
        return f"hz={self.hz}"


@dataclass(frozen=True)
class ListSpec:
    min_n: int = 1
    max_n: int = 1 << 30
    homogeneous: bool = True

    def overlaps(self, other: "ListSpec") -> bool:
        return self.min_n <= other.max_n and other.min_n <= self.max_n

    def __repr__(self) -> str:
        hi = "" if self.max_n >= 1 << 30 else str(self.max_n)
        return f"[{self.min_n}..{hi}]"


# 필드 이름 -> 사람이 읽는 라벨. 에러 메시지와 표시에 쓴다.
SCALAR_FIELDS = (
    "dtype",
    "layout",
    "value_range",
    "norm",
    "colorspace",
    "frame",
    "time_base",
)


@dataclass(frozen=True)
class PortType:
    base: Union[BaseKind, Var]
    dtype: Union[DType, Var, _Any, None] = None
    shape: Optional[tuple] = None
    layout: Union[Layout, Var, _Any, None] = None
    value_range: Union[Range, Var, _Any, None] = None
    norm: Union[Norm, Var, _Any, None] = None
    colorspace: Union[Color, Var, _Any, None] = None
    frame: Union[Frame, Var, _Any, None] = None
    time_base: Union[TimeBase, Var, _Any, None] = None
    semantic: frozenset = frozenset()
    optional: bool = False
    list_of: Optional[ListSpec] = None
    extra: tuple = ()  # (key, value) 쌍. domain, image_slots 등 base별 부가 제약

    # ── 편의 ────────────────────────────────────────────────────────────
    def get_extra(self, key: str, default: Any = None) -> Any:
        for k, v in self.extra:
            if k == key:
                return v
        return default

    def with_extra(self, **kw: Any) -> "PortType":
        keep = tuple((k, v) for k, v in self.extra if k not in kw)
        return replace(self, extra=tuple(sorted(keep + tuple(kw.items()))))

    def as_list(self, min_n: int = 1, max_n: int = 1 << 30) -> "PortType":
        return replace(self, list_of=ListSpec(min_n, max_n))

    def with_sem(self, *tags: str) -> "PortType":
        return replace(self, semantic=frozenset(self.semantic | set(tags)))

    def as_optional(self, flag: bool = True) -> "PortType":
        return replace(self, optional=flag)

    @property
    def display_base(self) -> str:
        if isinstance(self.base, Var):
            return repr(self.base)
        if self.list_of is not None:
            return _LIST_DISPLAY.get(self.base, self.base.value + "List")
        return self.base.value

    def free_vars(self) -> set:
        out = set()
        for v in (self.base,) + tuple(getattr(self, f) for f in SCALAR_FIELDS):
            if isinstance(v, Var):
                out.add(v.name)
        if isinstance(self.time_base, TimeBase) and isinstance(self.time_base.hz, Var):
            out.add(self.time_base.hz.name)
        for d in self.shape or ():
            if isinstance(d, DimVar):
                out.add(d.name)
        for _, v in self.extra:
            if isinstance(v, Var):
                out.add(v.name)
        return out

    def is_ground(self) -> bool:
        return not self.free_vars()

    # ── 표기 ────────────────────────────────────────────────────────────
    def __str__(self) -> str:
        parts = []
        for f in SCALAR_FIELDS:
            v = getattr(self, f)
            if v is None:
                continue
            parts.append(_fmt(v))
        if self.shape is not None:
            parts.append("[" + ",".join(_fmt(d) for d in self.shape) + "]")
        for k, v in self.extra:
            parts.append(f"{k}={_fmt(v)}")
        if self.semantic:
            parts.append("sem=" + "|".join(sorted(self.semantic)))
        if self.list_of is not None:
            parts.append(f"n{self.list_of!r}")
        if self.optional:
            parts.append("optional")
        inner = ", ".join(parts)
        return f"{self.display_base}{{{inner}}}" if inner else self.display_base

    __repr__ = __str__


def _fmt(v: Any) -> str:
    if isinstance(v, Enum):
        return v.value
    return repr(v) if isinstance(v, (Var, DimVar, TimeBase, _Any, _Dyn)) else str(v)


# ── 자주 쓰는 타입 생성기 ────────────────────────────────────────────────


def image(
    dtype: Any = DType.U8,
    layout: Any = Layout.HWC,
    colorspace: Any = Color.RGB,
    value_range: Any = Range.U8_0_255,
    norm: Any = Norm.NONE,
    frame: Any = Frame.ORIG_PX,
    shape: Optional[tuple] = None,
    **kw: Any,
) -> PortType:
    return PortType(
        base=BaseKind.IMAGE,
        dtype=dtype,
        layout=layout,
        colorspace=colorspace,
        value_range=value_range,
        norm=norm,
        frame=frame,
        shape=shape if shape is not None else (DYN, DYN, 3),
        **kw,
    )


def timeseries(
    dtype: Any = DType.F32,
    layout: Any = Layout.TC,
    hz: Any = ANY,
    frame: Any = Frame.TS_SECONDS,
    shape: Optional[tuple] = None,
    **kw: Any,
) -> PortType:
    return PortType(
        base=BaseKind.TIME_SERIES,
        dtype=dtype,
        layout=layout,
        frame=frame,
        time_base=hz if isinstance(hz, (Var, _Any)) else TimeBase(hz),
        shape=shape if shape is not None else (DYN, DYN),
        **kw,
    )


def text(*sem: str, **kw: Any) -> PortType:
    return PortType(base=BaseKind.TEXT, semantic=frozenset(sem), **kw)


def regions(domain: Any = ANY, frame: Any = ANY, **kw: Any) -> PortType:
    t = PortType(base=BaseKind.REGIONS, frame=frame, **kw)
    return t.with_extra(domain=domain)


def simple(kind: BaseKind, **kw: Any) -> PortType:
    return PortType(base=kind, **kw)
