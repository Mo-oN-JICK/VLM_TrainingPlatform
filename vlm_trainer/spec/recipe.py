"""Parameter Recipe — 그래프는 그대로 두고 값만 바꾼다.

Mech-Vision의 메커니즘을 그대로 가져오되 목적을 확장한다. 원래 목적은 "로직은 같고
파라미터만 다른 프로젝트를 중복 구축하지 않기", 우리는 여기에 실험 스윕을 얹는다.

레시피는 **값만** 덮는다. 노드·배선·sample_space·물질화 경계·실행 프로파일은 건드릴 수 없고,
무엇을 덮어도 되는지는 노드가 화이트리스트로 선언한다(검사는 컴파일러에 있다).
설계 문서 13.
"""

from __future__ import annotations

import itertools
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from ..core.errors import PolicyError, SpecError

MIN_ID, MAX_ID = 1, 99
CUSTOMIZED = "Customized"
LOCK_NAME = "recipes.lock.yaml"

FORBIDDEN_PREFIXES = ("sample_space", "materialize", "runtime_profile", "nodes", "edges", "procedures")


@dataclass
class Recipe:
    id: int
    name: str = ""
    note: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)
    origin: str = "manual"  # manual | sweep:<name>

    @property
    def label(self) -> str:
        return f"{self.id:02d}_{self.name}" if self.name else f"{self.id:02d}"


@dataclass
class Sweep:
    name: str
    axes: Dict[str, List[Any]] = field(default_factory=dict)
    id_range: Tuple[int, int] = (10, MAX_ID)
    base: Optional[int] = None
    strategy: str = "grid"  # grid | list | random
    naming: str = ""
    samples: int = 0
    seed: int = 0


@dataclass
class RecipeBook:
    project: str = ""
    active: Optional[int] = None
    display: Dict[str, str] = field(default_factory=dict)
    recipes: Dict[int, Recipe] = field(default_factory=dict)
    sweeps: Dict[str, Sweep] = field(default_factory=dict)
    path: str = ""

    def get(self, rid: int) -> Recipe:
        if rid not in self.recipes:
            raise SpecError(f"레시피 {rid}번이 없다 (있는 번호: {sorted(self.recipes)})")
        return self.recipes[rid]

    def overrides_for(self, rid: Optional[int]) -> Dict[str, Any]:
        return dict(self.get(rid).overrides) if rid is not None else {}

    def display_name(self, path: str) -> str:
        return self.display.get(path, path)

    def next_ids(self, count: int, id_range: Tuple[int, int]) -> List[int]:
        lo, hi = id_range
        free = [i for i in range(max(lo, MIN_ID), min(hi, MAX_ID) + 1) if i not in self.recipes]
        if len(free) < count:
            raise SpecError(
                f"번호가 모자란다: {count}개가 필요한데 {lo}~{hi} 범위에 {len(free)}개만 비어 있다"
            )
        return free[:count]


# ── 로드 ────────────────────────────────────────────────────────────────


def _check_override_path(path: str) -> None:
    if "." not in path:
        raise SpecError(f"오버라이드 경로는 '노드id.파라미터' 형식이어야 한다: {path!r}")
    head = path.split(".", 1)[0]
    if head in FORBIDDEN_PREFIXES:
        raise PolicyError(
            f"레시피는 {head!r}를 바꿀 수 없다. 값만 덮을 수 있고 그래프 구조는 스펙의 것이다.\n"
            "  구조가 다르면 그것은 같은 실험의 변주가 아니라 다른 실험이다."
        )


def load(project_path: str) -> RecipeBook:
    """project.yaml 옆의 recipes.yaml. 없으면 빈 책을 돌려준다."""
    d = os.path.dirname(os.path.abspath(project_path))
    path = os.path.join(d, "recipes.yaml")
    if not os.path.exists(path):
        return RecipeBook(path=path)

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if data.get("kind") not in (None, "ParameterRecipes"):
        raise SpecError(f"{path}: kind는 ParameterRecipes여야 한다")

    book = RecipeBook(
        project=str(data.get("project", "")),
        active=data.get("active"),
        display=dict(data.get("display") or {}),
        path=path,
    )
    for r in data.get("recipes") or []:
        rid = int(r["id"])
        if not (MIN_ID <= rid <= MAX_ID):
            raise SpecError(f"{path}: 레시피 번호는 {MIN_ID}~{MAX_ID}여야 한다 (받은 값 {rid})")
        if rid in book.recipes:
            raise SpecError(f"{path}: 레시피 번호 {rid}가 중복이다")
        overrides = dict(r.get("overrides") or {})
        for k in overrides:
            _check_override_path(k)
        book.recipes[rid] = Recipe(
            id=rid, name=str(r.get("name", "")), note=str(r.get("note", "")), overrides=overrides
        )

    for s in data.get("sweeps") or []:
        axes = {str(k): list(v) for k, v in (s.get("axes") or {}).items()}
        for k in axes:
            _check_override_path(k)
        name = str(s["name"])
        rng = s.get("id_range") or [10, MAX_ID]
        book.sweeps[name] = Sweep(
            name=name,
            axes=axes,
            id_range=(int(rng[0]), int(rng[1])),
            base=s.get("base"),
            strategy=str(s.get("strategy", "grid")),
            naming=str(s.get("naming", "")),
            samples=int(s.get("samples", 0)),
            seed=int(s.get("seed", 0)),
        )
    return book


def save_book(book: RecipeBook) -> str:
    """책을 파일로 되쓴다. 원자 교체.

    파일에 있던 다른 키(`sweeps`, 주석 아닌 확장 키)는 그대로 둔다. **주석은 사라진다** —
    `yaml.safe_dump`가 주석을 모른다. CLI의 `recipe set-active`도 같은 성질이다.
    """
    data: Dict[str, Any] = {}
    if os.path.exists(book.path):
        with open(book.path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    data["kind"] = "ParameterRecipes"
    if book.project:
        data["project"] = book.project
    if book.active is None:
        data.pop("active", None)
    else:
        data["active"] = int(book.active)
    if book.display:
        data["display"] = dict(book.display)

    rows: List[Dict[str, Any]] = []
    for rid in sorted(book.recipes):
        r = book.recipes[rid]
        row: Dict[str, Any] = {"id": r.id}
        if r.name:
            row["name"] = r.name
        if r.note:
            row["note"] = r.note
        row["overrides"] = dict(r.overrides)
        rows.append(row)
    data["recipes"] = rows

    os.makedirs(os.path.dirname(book.path) or ".", exist_ok=True)
    tmp = book.path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False, width=100)
    os.replace(tmp, book.path)
    return book.path


def check_override_path(path: str) -> None:
    """레시피가 덮을 수 있는 형태인지. 화이트리스트 검사는 컴파일러가 한다."""
    _check_override_path(path)


# ── 상태: 활성인가 Customized인가 ────────────────────────────────────────


def status(book: RecipeBook, current: Dict[str, Any]) -> str:
    """활성 레시피 번호 또는 Customized.

    프로젝트를 직접 수정해 활성 레시피가 더 이상 프로젝트를 설명하지 못하면 Customized다.
    `current`는 오버라이드 경로 -> 현재 프로젝트 값. Mech-Vision의 규약을 그대로 따른다.
    """
    if book.active is None:
        return CUSTOMIZED if book.recipes else ""
    r = book.recipes.get(int(book.active))
    if r is None:
        return CUSTOMIZED
    for path, value in r.overrides.items():
        if current.get(path, object()) != value:
            return CUSTOMIZED
    return str(book.active)


# ── 스윕 전개 ────────────────────────────────────────────────────────────


def expand(book: RecipeBook, sweep_name: str) -> List[Recipe]:
    """축을 전개해 구체 레시피 목록을 만든다. 같은 입력이면 같은 목록이다."""
    if sweep_name not in book.sweeps:
        raise SpecError(f"스윕 {sweep_name!r}가 없다 (있는 것: {sorted(book.sweeps)})")
    sw = book.sweeps[sweep_name]
    keys = sorted(sw.axes)  # 결정적 순서
    if not keys:
        raise SpecError(f"스윕 {sweep_name}: axes가 비어 있다")

    if sw.strategy == "grid":
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(sw.axes[k] for k in keys))]
    elif sw.strategy == "list":
        n = {len(sw.axes[k]) for k in keys}
        if len(n) != 1:
            raise SpecError(f"스윕 {sweep_name}: list 전략은 모든 축의 길이가 같아야 한다")
        combos = [dict(zip(keys, vals)) for vals in zip(*(sw.axes[k] for k in keys))]
    elif sw.strategy == "random":
        allc = [dict(zip(keys, vals)) for vals in itertools.product(*(sw.axes[k] for k in keys))]
        rng = random.Random(sw.seed)
        combos = rng.sample(allc, min(sw.samples or len(allc), len(allc)))
    else:
        raise SpecError(f"스윕 {sweep_name}: 알 수 없는 strategy {sw.strategy!r}")

    base = dict(book.get(int(sw.base)).overrides) if sw.base is not None else {}
    ids = book.next_ids(len(combos), sw.id_range)

    out: List[Recipe] = []
    for rid, combo in zip(ids, combos):
        overrides = {**base, **combo}
        out.append(
            Recipe(
                id=rid,
                name=_name_for(sw, combo, rid),
                note=f"sweep {sw.name}",
                overrides=overrides,
                origin=f"sweep:{sw.name}",
            )
        )
    return out


def _name_for(sw: Sweep, combo: Dict[str, Any], rid: int) -> str:
    if not sw.naming:
        return f"{sw.name}_{rid}"
    short = {k.rsplit(".", 1)[-1]: v for k, v in combo.items()}
    try:
        return sw.naming.format(**short)
    except (KeyError, IndexError):
        return f"{sw.name}_{rid}"


def save_lock(book: RecipeBook, recipes: Sequence[Recipe], out_dir: str) -> str:
    """전개 결과를 구체 목록으로 고정한다.

    스윕 정의를 나중에 바꿔도 이미 실행한 번호의 의미가 변하지 않아야 한다.
    """
    path = os.path.join(out_dir, LOCK_NAME)
    payload = {
        "kind": "ParameterRecipesLock",
        "project": book.project,
        "recipes": [
            {"id": r.id, "name": r.name, "origin": r.origin, "overrides": r.overrides} for r in recipes
        ],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False, width=100)
    os.replace(tmp, path)
    return path


def load_lock(out_dir: str) -> List[Recipe]:
    path = os.path.join(out_dir, LOCK_NAME)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return [
        Recipe(id=int(r["id"]), name=r.get("name", ""), origin=r.get("origin", ""), overrides=dict(r.get("overrides") or {}))
        for r in data.get("recipes") or []
    ]


def diff(a: Recipe, b: Recipe) -> List[Tuple[str, Any, Any]]:
    keys = sorted(set(a.overrides) | set(b.overrides))
    return [(k, a.overrides.get(k), b.overrides.get(k)) for k in keys if a.overrides.get(k) != b.overrides.get(k)]
