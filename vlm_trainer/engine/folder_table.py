"""사진 폴더 → 표(CSV). 폴더 입력 방식이 엔진에 건네는 것은 결국 이 표 한 장이다.

엔진은 표만 읽는다 — "한 행 = 한 건" 그대로다. 여기서는 폴더를 훑어 표를 만들고, 폴더가 바뀌면 표를 맞춘다.

* **훑기.** 하위 폴더까지, 파일 이름순(상대 경로 순). 읽는 것은 `jpg / jpeg / png / bmp` 이고 나머지는
  건너뛰되 몇 개인지 센다.
* **건 이름.** 확장자를 뺀 상대 경로이고, 하위 폴더 구분은 `__` 다(`검정/a01.jpg` → `검정__a01`).
  `/` 를 쓰면 사전 준비가 사진을 `images/<건>_0.png` 로 저장하다 없는 폴더에서 멈춘다.
  확장자만 다른 같은 이름(`a.jpg`, `a.png`)은 거부한다 — 어느 쪽이 그 건인지 알 수 없다.
* **맞추기.** 새 사진은 정답이 빈 행으로 덧붙는다(빈 정답은 ③ 에서 NG). 사라진 사진의 행은 지우지 않고
  `제외` 에 x 를 적는다 — 입력해 둔 정답이 남는다. 되살리는 것은 사람이 x 를 지워서 한다.
  정답 칸이 새로 생기면 열을 덧붙이고, 없어진 항목의 열은 지우지 않는다.
"""

from __future__ import annotations

import csv
import io
import os
from typing import Dict, List, Sequence, Tuple

from ..core.errors import SpecError
from .table_edit import EXCLUDE_COL, EXCLUDE_MARK

EXTS = (".jpg", ".jpeg", ".png", ".bmp")
KEY_COL = "건"
IMAGE_COL = "사진"
SEP = "__"


def scan(folder: str) -> Tuple[List[Tuple[str, str]], int]:
    """([(건 이름, 폴더 기준 상대 경로)], 건너뛴 파일 수). 상대 경로는 `/` 로 적는다."""
    if not os.path.isdir(folder):
        raise SpecError(f"사진 폴더가 없습니다: {folder}\n  안 잡혔다면: 읽을 사진이 없어 표가 빈 채로 만들어집니다.\n"
                        "  폴더 경로를 확인하세요.")
    found: List[str] = []
    skipped = 0
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for f in files:
            if f.lower().endswith(EXTS):
                found.append(os.path.relpath(os.path.join(root, f), folder).replace("\\", "/"))
            else:
                skipped += 1
    found.sort()
    out: List[Tuple[str, str]] = []
    seen: Dict[str, str] = {}
    for rel in found:
        key = os.path.splitext(rel)[0].replace("/", SEP)
        if key in seen:
            raise SpecError(f"{folder}: 건 이름 {key!r}가 두 사진에 겹칩니다 ({seen[key]}, {rel}).\n"
                            "  안 잡혔다면: 어느 사진이 그 건인지 알 수 없어 정답이 엉뚱한 사진에 붙습니다.\n"
                            "  둘 중 하나의 파일 이름을 바꾸세요.")
        seen[key] = rel
        out.append((key, rel))
    return out, skipped


def _read(path: str) -> List[List[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return [r for r in csv.reader(fh) if any(c.strip() for c in r)]


def _write(path: str, rows: Sequence[Sequence[str]]) -> None:
    """엑셀에서 바로 열리게 BOM 과 CRLF 로 쓴다. 다 쓴 뒤에 바꿔치기한다."""
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\r\n").writerows(rows)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            fh.write(b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"))
        os.replace(tmp, path)
    except PermissionError:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SpecError(f"{path}: 엑셀에서 열려 있어 쓸 수 없습니다.\n  안 잡혔다면: 폴더와 표가 어긋난 채 학습합니다.\n"
                        "  엑셀에서 파일을 닫고 다시 시도하세요.") from None


def sync(folder: str, csv_path: str, answer_cols: Sequence[str]) -> Dict[str, int]:
    """폴더에 맞춰 표를 만들거나 맞춘다. 사진 경로는 표가 있는 폴더 기준 상대 경로로 적는다.

    돌려주는 것: 건수(total), 덧붙인 건(added), 새로 제외한 건(excluded), 건너뛴 파일(skipped).
    바뀐 것이 없으면 파일을 쓰지 않는다.
    """
    items, skipped = scan(folder)
    base = os.path.dirname(os.path.abspath(csv_path))
    rel_of = {}
    for key, rel in items:
        full = os.path.join(folder, rel)
        try:
            rel_of[key] = os.path.relpath(full, base).replace("\\", "/")
        except ValueError:          # 다른 드라이브면 상대 경로가 없다
            rel_of[key] = os.path.abspath(full).replace("\\", "/")

    rows = _read(csv_path) if os.path.exists(csv_path) else []
    names = [c.strip() for c in rows[0]] if rows else [KEY_COL, IMAGE_COL]
    for need in (KEY_COL, IMAGE_COL):
        if need not in names:
            raise SpecError(f"{csv_path}: `{need}` 열이 없습니다 (있는 열: {names}).\n"
                            "  안 잡혔다면: 어느 행이 어느 사진인지 맞출 수 없습니다.\n"
                            "  이 표를 지우고 폴더를 다시 고르거나, 첫 줄에 그 열을 되돌리세요.")
    body = [r + [""] * (len(names) - len(r)) for r in rows[1:]]
    changed = not rows
    for col in [*answer_cols, EXCLUDE_COL]:
        if col not in names:
            names.append(col)
            body = [r + [""] for r in body]
            changed = True
    k, im, ex = names.index(KEY_COL), names.index(IMAGE_COL), names.index(EXCLUDE_COL)

    have = {r[k].strip(): r for r in body}
    excluded = 0
    for key, r in have.items():
        if key not in rel_of and r[ex].strip().lower() != EXCLUDE_MARK:
            r[ex] = EXCLUDE_MARK
            excluded += 1
    added = 0
    for key, _ in items:
        if key in have:
            if have[key][im] != rel_of[key]:      # 표를 옮겼거나 폴더를 옮겼다
                have[key][im] = rel_of[key]
                changed = True
            continue
        row = [""] * len(names)
        row[k], row[im] = key, rel_of[key]
        body.append(row)
        added += 1
    if changed or added or excluded:
        _write(csv_path, [names, *body])
    return {"total": len(items), "added": added, "excluded": excluded, "skipped": skipped}
