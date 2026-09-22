"""sample_space — 그래프 밖의 샘플 공간 선언.

Input 노드는 배선 대신 여기서 값을 길어온다. split은 그룹 홀드아웃이 기본이며,
같은 그룹(예: 같은 환자)이 train과 val에 동시에 들어가지 않는다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any, Dict, List

from ..core.errors import SpecError


@dataclass
class SampleSpace:
    root: str
    key: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    index_path: str = ""

    def __len__(self) -> int:
        return len(self.rows)

    def keys(self) -> List[str]:
        return [str(r[self.key]) for r in self.rows]

    def split(self, name: str) -> "SampleSpace":
        rows = [r for r in self.rows if r.get("_split") == name]
        return SampleSpace(self.root, self.key, rows, self.index_path)

    def pick(self, n: int, seed: int = 0) -> List[Dict[str, Any]]:
        """결정적 표본 추출. split마다 최소 1건을 포함한다."""
        if n >= len(self.rows):
            return list(self.rows)
        by_split: Dict[str, List[Dict[str, Any]]] = {}
        for r in self.rows:
            by_split.setdefault(str(r.get("_split", "")), []).append(r)
        out: List[Dict[str, Any]] = []
        for split_rows in by_split.values():
            out.append(sorted(split_rows, key=lambda r: _h(str(r[self.key]), seed))[0])
        rest = [r for r in self.rows if r not in out]
        rest.sort(key=lambda r: _h(str(r[self.key]), seed))
        return (out + rest)[:n]


def _h(s: str, seed: int = 0) -> int:
    return int.from_bytes(blake2b(f"{seed}|{s}".encode(), digest_size=8).digest(), "little")


def load(spec: Dict[str, Any], base_dir: str) -> SampleSpace:
    index = str(spec.get("index") or "")
    if not index:
        raise SpecError("sample_space.index가 비어 있다")
    path = index if os.path.isabs(index) else os.path.normpath(os.path.join(base_dir, index))
    if not os.path.exists(path):
        raise SpecError(f"sample_space 인덱스 파일이 없다: {path}")

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SpecError(f"{path}:{ln} JSON 파싱 실패 — {e}") from None

    key = str(spec.get("key") or "sample_id")
    for i, r in enumerate(rows):
        if key not in r:
            raise SpecError(f"{path}:{i + 1} 행에 키 컬럼 {key!r}가 없다")

    expr = str(spec.get("filter") or "")
    if expr:
        rows = [r for r in rows if _safe_filter(expr, r)]

    _assign_splits(rows, dict(spec.get("splits") or {}), key)
    return SampleSpace(root=os.path.dirname(path), key=key, rows=rows, index_path=path)


def _safe_filter(expr: str, row: Dict[str, Any]) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": {}}, dict(row)))  # noqa: S307 - 스펙의 일부
    except Exception:
        return False


def _assign_splits(rows: List[Dict[str, Any]], splits: Dict[str, Any], key: str) -> None:
    if not rows:
        return
    ratios: Dict[str, float] = dict(splits.get("ratios") or {"train": 1.0})
    seed = int(splits.get("seed") or 0)
    strategy = str(splits.get("strategy") or "random")
    group_by = str(splits.get("group_by") or key)

    names = list(ratios)
    bounds: List[float] = []
    acc = 0.0
    total = sum(ratios.values()) or 1.0
    for n in names:
        acc += ratios[n] / total
        bounds.append(acc)

    for r in rows:
        token = str(r.get(group_by, r[key])) if strategy == "group_holdout" else str(r[key])
        if strategy == "column":
            r["_split"] = str(r.get(str(splits.get("column", "split")), names[0]))
            continue
        frac = (_h(token, seed) % 10_000) / 10_000.0
        for name, b in zip(names, bounds):
            if frac < b:
                r["_split"] = name
                break
        else:
            r["_split"] = names[-1]


def fingerprint(space: SampleSpace) -> str:
    """인덱스 파일 자체의 지문. Input 노드의 캐시 키에 함께 들어간다."""
    try:
        st = os.stat(space.index_path)
        return blake2b(f"{space.index_path}|{st.st_size}|{st.st_mtime_ns}".encode(), digest_size=12).hexdigest()
    except OSError:
        return "missing"
