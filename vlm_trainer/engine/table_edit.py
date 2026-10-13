"""표(CSV)의 한 건을 고치거나 제외한다 — NG 로 나온 건을 화면에서 고치는 길.

화면과 CLI(`vlmt row-set`, `vlmt row-exclude`)가 같은 이 함수를 쓴다. 원칙은 셋이다.

* **그 셀만 쓴다.** 다른 칸과 파일 모양(BOM, 줄바꿈)은 그대로 둔다.
* **원본을 한 번 남긴다.** 처음 고치는 순간 `표.bak.csv` 를 만들고, 그 뒤로는 덮지 않는다(가장 처음 모습이 남는다).
* **지우지 않는다.** 제외는 `제외` 열에 `x` 를 적는 것이다. 읽는 쪽(`drop_excluded`)이 그 행을 건너뛰고,
  몇 건을 건너뛰었는지 화면에 계속 보인다. `x` 를 지우면 되살아난다.

엑셀(.xlsx)과 JSONL 은 되쓰지 않는다 — 엑셀은 서식·병합을 깨뜨릴 수 있다. CSV 로 저장해 쓰게 한다.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
from typing import Any, Dict, List, Sequence, Tuple

from ..core.errors import SpecError
from . import xlsx

EXCLUDE_COL = "제외"
EXCLUDE_MARK = "x"


def target(data: Dict[str, Any], spec_dir: str) -> Dict[str, str]:
    """스펙(Line 또는 Project)이 읽는 표의 경로와 건 구분 열."""
    ss = data.get("sample_space") or {}
    path, _sheet = xlsx.split_sheet(str(ss.get("index", "")))
    if not path:
        raise SpecError("표 파일이 정해지지 않았습니다.\n  안 잡혔다면: 고칠 표를 알 수 없습니다.\n  `데이터` 에서 표를 고르세요.")
    return {"path": path if os.path.isabs(path) else os.path.normpath(os.path.join(spec_dir, path)),
            "key": str(ss.get("key", "sample_id"))}


# ── 읽는 쪽: 제외된 행을 건너뛴다 ────────────────────────────────────────
def drop_excluded(rows: List[Dict[str, Any]], path: str, key: str) -> Tuple[List[Dict[str, Any]], int]:
    """`제외` 열이 `x` 인 행을 뺀다. (남은 행, 뺀 건수). 비었거나 `x` 가 아닌 값은 거부한다."""
    kept, n = [], 0
    for i, r in enumerate(rows, 1):
        v = str(r.get(EXCLUDE_COL, "") or "").strip()
        if not v:
            kept.append(r)
        elif v.lower() == EXCLUDE_MARK:
            n += 1
        else:
            raise SpecError(f"{path}: {i}번째 건({r.get(key, '')})의 `{EXCLUDE_COL}` 칸 값 {v!r}를 읽을 수 없습니다.\n"
                            "  안 잡혔다면: 제외하려던 건이 학습에 들어가거나, 넣으려던 건이 빠집니다.\n"
                            f"  제외하려면 {EXCLUDE_MARK} 만 적고, 아니면 칸을 비우세요.")
    return kept, n


# ── 쓰는 쪽 ─────────────────────────────────────────────────────────────
def _need_csv(path: str) -> None:
    if not path.lower().endswith(".csv"):
        raise SpecError(f"{path}: 화면에서 고칠 수 있는 표는 CSV 뿐입니다.\n"
                        "  안 잡혔다면: 엑셀의 서식·병합이 깨지거나, 고친 값이 원본에 안 남습니다.\n"
                        "  엑셀에서 `다른 이름으로 저장`으로 CSV UTF-8 로 저장해 쓰세요.")


def _load(path: str) -> Tuple[List[List[str]], bool, bool]:
    try:
        raw = open(path, "rb").read()
    except FileNotFoundError:
        raise SpecError(f"표 파일이 없습니다: {path}\n  안 잡혔다면: 고친 값을 쓸 곳이 없습니다.\n  파일 경로를 확인하세요.") from None
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    return list(csv.reader(io.StringIO(text, newline=""))), bom, "\r\n" in text


def _save(path: str, rows: Sequence[Sequence[str]], bom: bool, crlf: bool) -> None:
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\r\n" if crlf else "\n").writerows(rows)
    tmp = path + ".tmp"
    base, ext = os.path.splitext(path)
    backup = base + ".bak" + ext
    try:
        if not os.path.exists(backup):
            shutil.copyfile(path, backup)
        with open(tmp, "wb") as fh:
            fh.write((b"\xef\xbb\xbf" if bom else b"") + buf.getvalue().encode("utf-8"))
        os.replace(tmp, path)
    except PermissionError:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SpecError(f"{path}: 엑셀에서 열려 있어 쓸 수 없습니다.\n  안 잡혔다면: 고친 값이 저장되지 않은 채 다시 읽습니다.\n"
                        "  엑셀에서 파일을 닫고 다시 시도하세요.") from None


def _find(rows: List[List[str]], key: str, sample_id: str, path: str) -> Tuple[List[str], int, int]:
    """(열 이름들, 키 열 위치, 그 건의 행 위치)."""
    start = next((i for i, r in enumerate(rows) if any(c.strip() for c in r)), None)
    if start is None:
        raise SpecError(f"{path}: 비어 있습니다.\n  안 잡혔다면: 고칠 건이 없습니다.\n  표 파일을 확인하세요.")
    names = [c.strip() for c in rows[start]]
    if key not in names:
        raise SpecError(f"{path}: 건 구분 열 {key!r}가 없습니다 (있는 열: {names}).\n"
                        "  안 잡혔다면: 어느 행이 그 건인지 알 수 없습니다.\n  `데이터` 에서 건 구분 열을 다시 고르세요.")
    k = names.index(key)
    hit = [i for i in range(start + 1, len(rows)) if len(rows[i]) > k and rows[i][k].strip() == str(sample_id)]
    if len(hit) != 1:
        raise SpecError(f"{path}: 건 {sample_id!r}가 {'없습니다' if not hit else f'{len(hit)}줄에 있습니다'}.\n"
                        "  안 잡혔다면: 엉뚱한 행을 고치거나 아무것도 안 고쳐집니다.\n"
                        + ("  표를 고쳐 다시 읽은 뒤 NG 목록을 다시 확인하세요." if not hit else "  같은 건 구분 값이 겹치지 않게 표를 고치세요."))
    return names, k, hit[0]


def row_values(path: str, key: str, sample_id: str) -> Dict[str, str]:
    """그 건의 열 이름 → 값. 화면이 고칠 칸을 보여 줄 때 쓴다."""
    _need_csv(path)
    rows, _, _ = _load(path)
    names, _, i = _find(rows, key, sample_id, path)
    row = rows[i] + [""] * (len(names) - len(rows[i]))
    return dict(zip(names, row))


def set_cells(path: str, key: str, sample_id: str, edits: Dict[str, str]) -> List[str]:
    """그 건의 칸을 고쳐 쓴다. 실제로 값이 바뀐 열 이름을 돌려준다(바뀐 것이 없으면 파일을 안 쓴다)."""
    _need_csv(path)
    rows, bom, crlf = _load(path)
    names, k, i = _find(rows, key, sample_id, path)
    bad = [c for c in edits if c not in names and c != EXCLUDE_COL]
    if bad:
        raise SpecError(f"{path}: 없는 열 {bad}는 고칠 수 없습니다 (있는 열: {names}).\n"
                        "  안 잡혔다면: 이름이 틀린 값이 어디에도 안 쓰입니다.\n  열 이름을 확인하세요.")
    if key in edits:
        raise SpecError(f"건 구분 열 {key!r}는 고칠 수 없습니다.\n  안 잡혔다면: NG 목록과 표가 서로 다른 건을 가리킵니다.\n"
                        "  건 이름을 바꾸려면 표 파일을 직접 고치세요.")
    if EXCLUDE_COL in edits and edits[EXCLUDE_COL].strip().lower() not in ("", EXCLUDE_MARK):
        raise SpecError(f"`{EXCLUDE_COL}` 칸에는 {EXCLUDE_MARK} 만 적을 수 있습니다 (받은 값: {edits[EXCLUDE_COL]!r}).\n"
                        "  안 잡혔다면: 제외하려던 건이 학습에 들어갑니다.\n  제외하려면 x, 되살리려면 빈 칸으로 보내세요.")
    start = next(j for j, r in enumerate(rows) if any(c.strip() for c in r))
    if EXCLUDE_COL in edits and EXCLUDE_COL not in names:
        rows[start] = rows[start] + [""] * (len(names) - len(rows[start])) + [EXCLUDE_COL]
        names.append(EXCLUDE_COL)
    rows[i] = rows[i] + [""] * (len(names) - len(rows[i]))
    changed = []
    for col, val in edits.items():
        j = names.index(col)
        if rows[i][j] != val:
            rows[i][j] = val
            changed.append(col)
    if changed:
        _save(path, rows, bom, crlf)
    return changed


def set_excluded(path: str, key: str, sample_id: str, on: bool) -> List[str]:
    return set_cells(path, key, sample_id, {EXCLUDE_COL: EXCLUDE_MARK if on else ""})
