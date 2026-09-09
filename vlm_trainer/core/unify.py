"""타입 호환성 판정과 단일화.

"이 배선을 연결해도 되는가"에 답하는 유일한 곳.
자동 캐스팅 코드는 여기에도 다른 어디에도 들어가지 않는다. 설계 문서 02 §2.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from .types import (
    ANY,
    DYN,
    SCALAR_FIELDS,
    DimVar,
    ListSpec,
    PortType,
    TimeBase,
    Var,
    _Any,
    _Dyn,
)

Subst = Dict[str, Any]


@dataclass
class Mismatch:
    field: str
    expected: Any  # 목적지(dst)가 요구한 것
    actual: Any  # 소스(src)가 준 것
    note: str = ""

    def __str__(self) -> str:
        n = f" — {self.note}" if self.note else ""
        return f"{self.field}({_s(self.actual)} != {_s(self.expected)}){n}"


@dataclass
class Result:
    ok: bool
    subst: Subst = field(default_factory=dict)
    mismatches: List[Mismatch] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def _s(v: Any) -> str:
    from enum import Enum

    if isinstance(v, Enum):
        return v.value
    if v is None:
        return "-"
    if isinstance(v, _Dyn):
        return "dyn"
    return repr(v) if isinstance(v, (Var, DimVar, _Any, TimeBase, ListSpec)) else str(v)


def resolve(v: Any, subst: Subst) -> Any:
    """치환을 적용해 변수를 값으로 바꾼다. 미결이면 변수 자신을 돌려준다."""
    seen = set()
    while isinstance(v, (Var, DimVar)) and v.name in subst and v.name not in seen:
        seen.add(v.name)
        v = subst[v.name]
    return v


def _bind(var: Any, value: Any, subst: Subst) -> bool:
    if isinstance(value, (Var, DimVar)) and value.name == var.name:
        return True
    subst[var.name] = value
    return True


def _unify_scalar(name: str, src: Any, dst: Any, subst: Subst, out: List[Mismatch]) -> bool:
    """스칼라 필드 하나를 단일화한다.

    규칙 2: 목적지가 ANY를 선언했으면 통과(그 필드에 의존하지 않겠다는 계약).
    소스가 ANY면 확정되지 않은 값이므로 보수적으로 거부한다.
    """
    s, d = resolve(src, subst), resolve(dst, subst)

    # 목적지가 선언하지 않은 필드는 제약이 아니다(제네릭 포트가 이 경우다).
    if d is None:
        return True
    if s is None:
        out.append(Mismatch(name, d, None, "소스에 해당 필드가 없음"))
        return False
    if isinstance(d, _Any):
        return True
    # 가변 차원: 목적지가 가변을 받아들이면 통과, 고정 크기를 요구하면 거부
    if isinstance(s, _Dyn) and isinstance(d, (DimVar, Var)):
        return _bind(d, s, subst)
    if isinstance(s, _Dyn) and not isinstance(d, _Dyn):
        out.append(
            Mismatch(name, d, s, "소스 크기가 샘플마다 달라 고정 크기를 보장할 수 없음 — adapt.image_resize가 필요")
        )
        return False
    if isinstance(d, _Dyn):
        return True
    if isinstance(s, _Any):
        out.append(Mismatch(name, d, s, "소스가 이 필드를 확정하지 않음"))
        return False
    if isinstance(d, (Var, DimVar)):
        return _bind(d, s, subst)
    if isinstance(s, (Var, DimVar)):
        return _bind(s, d, subst)
    if isinstance(s, TimeBase) and isinstance(d, TimeBase):
        if not _unify_scalar(f"{name}.hz", s.hz, d.hz, subst, out):
            return False
        if s.t0_policy != d.t0_policy:
            out.append(Mismatch(f"{name}.t0_policy", d.t0_policy, s.t0_policy))
            return False
        return True
    if s == d:
        return True
    out.append(Mismatch(name, d, s))
    return False


def _unify_shape(src: PortType, dst: PortType, subst: Subst, out: List[Mismatch]) -> bool:
    ss, ds = src.shape, dst.shape
    if ds is None:
        return True
    if ss is None:
        out.append(Mismatch("shape", ds, None, "소스에 shape 정보 없음"))
        return False
    if len(ss) != len(ds):
        out.append(Mismatch("shape.rank", len(ds), len(ss)))
        return False
    ok = True
    for i, (a, b) in enumerate(zip(ss, ds)):
        if not _unify_scalar(f"shape[{i}]", a, b, subst, out):
            ok = False
    return ok


def unify_ports(src: PortType, dst: PortType, subst: Optional[Subst] = None) -> Result:
    """출력 포트 src를 입력 포트 dst에 연결할 수 있는지 판정한다.

    판정은 이진이다. 경고 등급도, 완화 옵션도 없다.
    """
    sub: Subst = dict(subst or {})
    out: List[Mismatch] = []

    # 1) base kind — 서브타입도 업캐스트도 없다
    _unify_scalar("base", src.base, dst.base, sub, out)

    # 2) 스칼라 필드
    for f in SCALAR_FIELDS:
        _unify_scalar(f, getattr(src, f), getattr(dst, f), sub, out)

    # 3) shape 단일화
    _unify_shape(src, dst, sub, out)

    # 4) semantic — 목적지가 요구한 태그를 소스가 전부 갖고 있어야 한다
    missing = set(dst.semantic) - set(src.semantic)
    if missing:
        out.append(
            Mismatch(
                "semantic",
                "|".join(sorted(dst.semantic)),
                "|".join(sorted(src.semantic)) or "-",
                "누락 태그: " + ", ".join(sorted(missing)),
            )
        )

    # 5) optional — optional 소스를 필수 입력에 꽂을 수 없다
    if src.optional and not dst.optional:
        out.append(Mismatch("optional", False, True, "optional.unwrap_or 노드가 필요"))

    # 6) list — 브로드캐스트 없음
    if (src.list_of is None) != (dst.list_of is None):
        out.append(
            Mismatch(
                "list",
                "list" if dst.list_of else "scalar",
                "list" if src.list_of else "scalar",
                "list.wrap 또는 list.map 노드가 필요",
            )
        )
    elif src.list_of is not None and not src.list_of.overlaps(dst.list_of):
        out.append(Mismatch("list.n", dst.list_of, src.list_of))

    # 7) base별 부가 제약(domain, image_slots 등)
    for k, dv in dst.extra:
        sv = src.get_extra(k, None)
        if sv is None and not isinstance(resolve(dv, sub), _Any):
            out.append(Mismatch(f"extra.{k}", dv, None, "소스에 해당 제약이 없음"))
            continue
        _unify_scalar(f"extra.{k}", sv, dv, sub, out)

    return Result(ok=not out, subst=sub, mismatches=out)


def apply_subst(pt: PortType, subst: Subst) -> PortType:
    """치환을 적용해 타입을 구체화한다. compile 종료 시 모든 타입이 ground여야 한다."""
    kw: Dict[str, Any] = {"base": resolve(pt.base, subst)}
    for f in SCALAR_FIELDS:
        v = resolve(getattr(pt, f), subst)
        if isinstance(v, TimeBase):
            v = TimeBase(resolve(v.hz, subst), v.t0_policy)
        kw[f] = v
    if pt.shape is not None:
        kw["shape"] = tuple(resolve(d, subst) for d in pt.shape)
    if pt.extra:
        kw["extra"] = tuple((k, resolve(v, subst)) for k, v in pt.extra)
    return replace(pt, **kw)


def rename_vars(pt: PortType, prefix: str) -> PortType:
    """노드 인스턴스마다 타입 변수를 분리한다(같은 이름이 다른 노드에서 충돌하지 않도록)."""

    def rn(v: Any) -> Any:
        if isinstance(v, Var):
            return Var(f"{prefix}.{v.name}")
        if isinstance(v, DimVar):
            return DimVar(f"{prefix}.{v.name}")
        if isinstance(v, TimeBase):
            return TimeBase(rn(v.hz), v.t0_policy)
        return v

    kw: Dict[str, Any] = {"base": rn(pt.base)}
    for f in SCALAR_FIELDS:
        kw[f] = rn(getattr(pt, f))
    if pt.shape is not None:
        kw["shape"] = tuple(rn(d) for d in pt.shape)
    if pt.extra:
        kw["extra"] = tuple((k, rn(v)) for k, v in pt.extra)
    return replace(pt, **kw)
