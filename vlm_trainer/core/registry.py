"""노드 레지스트리.

노드가 계약을 선언하는 입구이자, 3분류-포트형상 정합을 강제하는 문지기.
여기서 거부되면 임포트 자체가 실패한다 — 잘못된 노드가 등록된 채로 실행에 들어가는 일은 없다.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

from .errors import RegistrationError
from .node import Node, NodeDef, NodeDoc, NodeKind, Port

_REGISTRY: Dict[Tuple[str, str], NodeDef] = {}


def _check_shape(d: NodeDef) -> None:
    """3분류와 포트 형상의 1:1 대응. 설계 문서 02 §2.4."""
    n_in, n_out = len(d.inputs), len(d.outputs)
    if d.kind is NodeKind.INPUT and (n_in != 0 or n_out < 1):
        raise RegistrationError(
            f"{d.ref}: Input 노드는 입력 포트가 없고 출력이 1개 이상이어야 한다 "
            f"(현재 입력 {n_in}, 출력 {n_out})"
        )
    if d.kind is NodeKind.PROCESSING and (n_in < 1 or n_out < 1):
        raise RegistrationError(
            f"{d.ref}: Processing 노드는 입력과 출력이 각각 1개 이상이어야 한다 "
            f"(현재 입력 {n_in}, 출력 {n_out})"
        )
    if d.kind is NodeKind.OUTPUT and (n_in < 1 or n_out != 0):
        raise RegistrationError(
            f"{d.ref}: Output 노드는 입력이 1개 이상이고 출력 포트가 없어야 한다 "
            f"(현재 입력 {n_in}, 출력 {n_out})"
        )


def _check_contracts(d: NodeDef) -> None:
    if d.kind is NodeKind.INPUT and d.impl is not None:
        if d.impl.fingerprint is Node.fingerprint:
            raise RegistrationError(f"{d.ref}: Input 노드는 fingerprint()를 구현해야 한다 (C4)")
    if not d.per_sample and d.kind is not NodeKind.OUTPUT:
        raise RegistrationError(f"{d.ref}: per_sample=False는 Output 노드에만 허용된다")
    if d.external_call and d.kind is not NodeKind.PROCESSING:
        raise RegistrationError(f"{d.ref}: external_call은 Processing 노드에만 허용된다")
    if d.params is not None and not dataclasses.is_dataclass(d.params):
        raise RegistrationError(f"{d.ref}: params는 dataclass여야 한다 (C3)")

    known = set(d.default_params())
    for group, names in (("recipe_overridable", d.recipe_overridable), ("type_affecting", d.type_affecting)):
        bad = set(names) - known
        if bad:
            raise RegistrationError(f"{d.ref}: {group}에 없는 파라미터 {sorted(bad)}")
    bad = set(d.type_affecting) - set(d.recipe_overridable)
    if bad:
        raise RegistrationError(
            f"{d.ref}: type_affecting은 recipe_overridable의 부분집합이어야 한다 {sorted(bad)}"
        )

    # spawn 요구: 구현이 모듈 최상위에서 임포트 가능해야 워커로 넘길 수 있다
    if d.impl is not None:
        mod, qual = d.impl.__module__, d.impl.__qualname__
        if "<locals>" in qual:
            raise RegistrationError(
                f"{d.ref}: 노드 구현이 함수 안에 정의되어 있다. "
                "Windows spawn 워커가 임포트할 수 없다 — 모듈 최상위로 옮겨라."
            )
        if mod == "__main__":
            raise RegistrationError(f"{d.ref}: 노드 구현을 __main__에 두면 spawn 워커가 임포트할 수 없다")


def register(
    *,
    type: str,
    version: str,
    category: str,
    kind: NodeKind,
    inputs: Optional[Dict[str, Port]] = None,
    outputs: Optional[Dict[str, Port]] = None,
    params: Optional[type] = None,
    recipe_overridable: Iterable[str] = (),
    type_affecting: Iterable[str] = (),
    preview: str = "",
    external_call: bool = False,
    deterministic: bool = True,
    clears_taint: Iterable[str] = (),
    per_sample: bool = True,
    doc: Optional[NodeDoc] = None,
) -> Callable[[Type[Node]], Type[Node]]:
    def deco(cls: Type[Node]) -> Type[Node]:
        d = NodeDef(
            type=type,
            version=version,
            category=category,
            kind=kind,
            inputs=dict(inputs or {}),
            outputs=dict(outputs or {}),
            params=params,
            recipe_overridable=tuple(recipe_overridable),
            type_affecting=tuple(type_affecting),
            preview=preview,
            external_call=external_call,
            deterministic=deterministic,
            clears_taint=tuple(clears_taint),
            per_sample=per_sample,
            doc=doc or NodeDoc(),
            impl=cls,
        )
        _check_shape(d)
        _check_contracts(d)
        key = (d.type, d.version)
        if key in _REGISTRY and _REGISTRY[key].impl is not cls:
            raise RegistrationError(f"{d.ref}: 이미 등록된 노드 타입")
        _REGISTRY[key] = d
        cls.definition = d
        return cls

    return deco


# ── 조회 ────────────────────────────────────────────────────────────────


def _semver(v: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


def resolve(ref: str) -> NodeDef:
    """`name@1.0.0` 또는 `name`(최신 버전)을 NodeDef로 해소한다."""
    if "@" in ref:
        name, ver = ref.split("@", 1)
        d = _REGISTRY.get((name, ver))
        if d is None:
            have = sorted(v for (n, v) in _REGISTRY if n == name)
            raise RegistrationError(
                f"노드 {ref}를 찾을 수 없다" + (f" (등록된 버전: {have})" if have else " (등록되지 않은 타입)")
            )
        return d
    cands = [d for (n, _), d in _REGISTRY.items() if n == ref]
    if not cands:
        raise RegistrationError(f"노드 {ref}를 찾을 수 없다")
    return max(cands, key=lambda d: _semver(d.version))


def all_defs() -> List[NodeDef]:
    return sorted(_REGISTRY.values(), key=lambda d: (d.category, d.type, d.version))


def categories() -> Dict[str, List[NodeDef]]:
    out: Dict[str, List[NodeDef]] = {}
    for d in all_defs():
        out.setdefault(d.category, []).append(d)
    return out


def clear_for_tests() -> None:
    _REGISTRY.clear()


def load_builtin_nodes() -> None:
    """내장 노드 패키지를 임포트해 등록시킨다."""
    importlib.import_module("vlm_trainer.nodes")
