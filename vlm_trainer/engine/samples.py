"""sample_space — 그래프 밖의 샘플 공간 선언.

Input 노드는 배선 대신 여기서 값을 길어온다. split은 그룹 홀드아웃이 기본이며,
같은 그룹(예: 같은 환자)이 train과 val에 동시에 들어가지 않는다.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any, Dict, List

from ..core.errors import SpecError
from . import table_edit, xlsx


@dataclass
class SampleSpace:
    root: str
    key: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    index_path: str = ""
    excluded: int = 0          # 표의 `제외` 열 때문에 건너뛴 건수. 화면에 계속 보인다

    def __len__(self) -> int:
        return len(self.rows)

    def keys(self) -> List[str]:
        return [str(r[self.key]) for r in self.rows]

    def split(self, name: str) -> "SampleSpace":
        rows = [r for r in self.rows if r.get("_split") == name]
        return SampleSpace(self.root, self.key, rows, self.index_path, self.excluded)

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
        raise SpecError("sample_space.index가 비어 있습니다")
    index, sheet = xlsx.split_sheet(index)   # `표.xlsx#시트1` — 시트는 이름 뒤에 붙여 적는다
    path = index if os.path.isabs(index) else os.path.normpath(os.path.join(base_dir, index))
    if not os.path.exists(path):
        raise SpecError(f"sample_space 인덱스 파일이 없습니다: {path}")

    rows = read_table(path, sheet)

    key = str(spec.get("key") or "sample_id")
    for i, r in enumerate(rows):
        if key not in r:
            raise SpecError(f"{path}:{i + 1} 행에 키 컬럼 {key!r}가 없습니다")

    rows, excluded = table_edit.drop_excluded(rows, path, key)

    expr = str(spec.get("filter") or "")
    if expr:
        rows = [r for r in rows if _safe_filter(expr, r)]

    _assign_splits(rows, dict(spec.get("splits") or {}), key)
    return SampleSpace(root=os.path.dirname(path), key=key, rows=rows, index_path=path, excluded=excluded)


def read_table(path: str, sheet: str = "") -> List[Dict[str, Any]]:
    """표 한 장 = 한 행이 한 건. `.csv` 면 CSV, `.xlsx` 면 엑셀 한 시트(`engine/xlsx.py`), 아니면 JSONL 이다.

    CSV 는 UTF-8 이고 BOM 이 있어도 된다(엑셀의 "CSV UTF-8" 저장). 값은 전부 문자열로 읽는다 —
    "3" 을 숫자로 짐작해 바꾸는 것도 자동 캐스팅이다.
    """
    if path.lower().endswith((".xlsx", ".xls")):
        return xlsx.read_xlsx(path, sheet)
    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames:
                raise SpecError(f"{path}: 첫 줄(열 이름)이 비어 있습니다")
            names = [str(n).strip() for n in rd.fieldnames]
            dup = sorted({n for n in names if names.count(n) > 1})
            if dup:
                raise SpecError(f"{path}: 열 이름이 겹칩니다 {dup} — 같은 이름이 둘이면 어느 열인지 정해지지 않습니다")
            rd.fieldnames = names
            rows = []
            for ln, r in enumerate(rd, 2):
                if None in r:
                    raise SpecError(f"{path}:{ln} 행에 열 이름보다 값이 더 많습니다 — 값 안의 쉼표는 큰따옴표로 감싸야 합니다")
                if all(not (v or "").strip() for v in r.values()):
                    continue  # 엑셀이 남기는 빈 줄
                rows.append({k: (v if v is not None else "") for k, v in r.items()})
            return rows
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SpecError(f"{path}:{ln} JSON 파싱 실패 — {e}") from None
    return rows


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

    if strategy == "column":
        col = str(splits.get("column", "split"))
        for i, r in enumerate(rows):
            # 열이 없거나 칸이 비면 "첫 split 으로" 같은 짐작을 하지 않는다. 검증 건이 0이 되는 길이다.
            if not str(r.get(col, "")).strip():
                raise SpecError(
                    f"split 열 {col!r}이 {i + 1}번째 행({r[key]})에서 비었거나 없습니다.\n"
                    "  안 잡혔다면: 그 행이 짐작으로 한쪽에 들어가 학습과 검증이 섞이거나 검증이 0건이 되고,\n"
                    "  결과를 본 뒤에야 알게 됩니다.\n"
                    f"  표의 {col!r} 열에 모든 행의 값(예: train / val)을 채우세요.")
            r["_split"] = str(r[col]).strip()
        return
    if strategy == "group_holdout":
        missing = [r[key] for r in rows if group_by not in r]
        if missing:
            raise SpecError(
                f"group_by 열 {group_by!r}이 표에 없습니다 (예: {missing[0]}).\n"
                "  안 잡혔다면: 같은 그룹을 묶지 못하고 행마다 따로 나뉘어, 같은 부품·로트가 학습과 검증에\n"
                "  양쪽으로 들어갑니다. 검증 점수가 부풀어도 눈치채기 어렵습니다.\n"
                "  그룹 열 이름을 확인하거나, 그룹 없이 나누려면 strategy 를 random 으로 바꾸세요.")

    for r in rows:
        token = str(r.get(group_by, r[key])) if strategy == "group_holdout" else str(r[key])
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
