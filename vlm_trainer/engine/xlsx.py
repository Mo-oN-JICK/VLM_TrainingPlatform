"""`.xlsx` 한 시트를 표로 읽는다 — 표준 라이브러리만 쓴다(zip 안의 XML).

CSV 와 같은 규칙이다: 첫 줄이 열 이름, 한 행이 한 건, 값은 전부 문자열. 엑셀은 화면에 보이는 값과 저장된 값이
다를 수 있어서(날짜 · 퍼센트 · `00123` 같은 자릿수 서식) 그런 칸은 **읽지 않고 거부한다** — 짐작해서 바꾸면
조용히 다른 값이 학습된다. 병합된 칸도 거부한다(값이 왼쪽 위 칸에만 있어 나머지가 빈 칸으로 읽힌다).
"""

from __future__ import annotations

import re
import zipfile
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

from ..core.errors import SpecError

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
       "p": "http://schemas.openxmlformats.org/package/2006/relationships"}
_M = "{%s}" % _NS["m"]
# 화면 값 = 저장 값인 내장 서식: 일반, 0, 0.00, #,##0, #,##0.00, 텍스트
_PLAIN_FMT = {0, 1, 2, 3, 4, 49}
_PLAIN_CODE = re.compile(r"^(General|@|#?[#,]*0(\.0+)?)$")


def split_sheet(index: str) -> Tuple[str, str]:
    """`data/표.xlsx#시트1` → (`data/표.xlsx`, `시트1`). 시트를 안 적었으면 시트는 빈 문자열이다."""
    head, sep, tail = index.rpartition("#")
    if sep and head.lower().endswith(".xlsx"):
        return head, tail
    return index, ""


def _open(path: str) -> zipfile.ZipFile:
    if path.lower().endswith(".xls"):
        raise SpecError(f"{path}: 구형 엑셀 파일(.xls)은 읽지 않습니다.\n"
                        "  안 잡혔다면: 열 수 없는 파일을 표로 읽다가 첫 줄에서 멈춥니다.\n"
                        "  엑셀에서 `다른 이름으로 저장`으로 .xlsx 또는 CSV UTF-8 로 저장하세요.")
    try:
        return zipfile.ZipFile(path)
    except PermissionError:
        raise SpecError(f"{path}: 엑셀에서 열려 있어 읽을 수 없습니다.\n"
                        "  안 잡혔다면: 저장 전의 옛 내용을 읽거나 실행이 멈춥니다.\n"
                        "  엑셀에서 파일을 닫거나 복사본을 쓰세요.") from None
    except zipfile.BadZipFile:
        raise SpecError(f"{path}: 엑셀(.xlsx) 파일이 아니거나 깨졌습니다.\n"
                        "  안 잡혔다면: 표 내용을 알 수 없어 열 지정이 전부 어긋납니다.\n"
                        "  엑셀에서 .xlsx 로 다시 저장하세요.") from None


def _xml(z: zipfile.ZipFile, name: str):
    try:
        return ET.fromstring(z.read(name))
    except KeyError:
        return None


def _sheets(z: zipfile.ZipFile) -> List[Tuple[str, str]]:
    """(시트 이름, zip 안의 경로). 화면에 보이는 순서다."""
    wb = _xml(z, "xl/workbook.xml")
    rels = _xml(z, "xl/_rels/workbook.xml.rels")
    if wb is None or rels is None:
        raise SpecError("엑셀 파일 안에 시트 목록이 없습니다.\n  안 잡혔다면: 어느 시트를 읽는지 알 수 없습니다.\n"
                        "  엑셀에서 .xlsx 로 다시 저장하세요.")
    target = {r.get("Id"): r.get("Target", "") for r in rels.findall("p:Relationship", _NS)}
    out = []
    for s in wb.findall("m:sheets/m:sheet", _NS):
        t = target.get(s.get("{%s}id" % _NS["r"]), "")
        out.append((s.get("name", ""), t.lstrip("/") if t.startswith("/") else "xl/" + t))
    return out


def sheet_names(path: str) -> List[str]:
    with _open(path) as z:
        return [n for n, _ in _sheets(z)]


def _shared(z: zipfile.ZipFile) -> List[str]:
    root = _xml(z, "xl/sharedStrings.xml")
    if root is None:
        return []
    # 읽기 표기(rPh)의 글자는 화면에 보이지 않으므로 `si/t` 와 `si/r/t` 만 모은다
    return ["".join(t.text or "" for t in list(si.findall("m:t", _NS)) + list(si.findall("m:r/m:t", _NS)))
            for si in root.findall("m:si", _NS)]


def _formats(z: zipfile.ZipFile) -> List[bool]:
    """셀 서식 번호마다 '화면 값 = 저장 값' 인가."""
    root = _xml(z, "xl/styles.xml")
    if root is None:
        return []
    custom = {int(f.get("numFmtId")): f.get("formatCode", "") for f in root.findall("m:numFmts/m:numFmt", _NS)}
    out = []
    for xf in root.findall("m:cellXfs/m:xf", _NS):
        i = int(xf.get("numFmtId", 0))
        out.append(i in _PLAIN_FMT or (i in custom and bool(_PLAIN_CODE.match(custom[i]))))
    return out


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def _number(text: str) -> str:
    x = float(text)
    return str(int(x)) if x == int(x) and abs(x) < 1e15 else text


def read_xlsx(path: str, sheet: str = "") -> List[Dict[str, str]]:
    with _open(path) as z:
        sheets = _sheets(z)
        if not sheets:
            raise SpecError(f"{path}: 시트가 하나도 없습니다.\n  안 잡혔다면: 읽을 건이 없습니다.\n  시트가 있는 파일을 고르세요.")
        names = [n for n, _ in sheets]
        if sheet and sheet not in names:
            raise SpecError(f"{path}: 시트 {sheet!r}가 없습니다 (있는 시트: {names}).\n"
                            "  안 잡혔다면: 다른 시트를 읽거나 실행이 멈춥니다.\n  `데이터` 에서 시트를 다시 고르세요.")
        name, inner = sheets[names.index(sheet)] if sheet else sheets[0]
        root = _xml(z, inner)
        if root is None:
            raise SpecError(f"{path}: 시트 {name!r}의 내용을 찾을 수 없습니다.\n  안 잡혔다면: 표가 비어 있는 것으로 읽힙니다.\n"
                            "  엑셀에서 .xlsx 로 다시 저장하세요.")
        shared, plain = _shared(z), _formats(z)

    where = f"{path} 시트 {name!r}"
    if root.find("m:mergeCells/m:mergeCell", _NS) is not None:
        raise SpecError(f"{where}: 병합된 칸이 있습니다.\n  안 잡혔다면: 값이 왼쪽 위 칸에만 있어 나머지 행이 빈 값으로 학습됩니다.\n"
                        "  병합을 풀고 모든 칸에 값을 채우세요.")
    grid: List[Tuple[int, Dict[int, str]]] = []
    for row in root.findall("m:sheetData/m:row", _NS):
        cells: Dict[int, str] = {}
        for c in row.findall("m:c", _NS):
            ref, kind = c.get("r", ""), c.get("t", "n")
            v = c.find("m:v", _NS)
            if kind == "inlineStr":
                val = "".join(t.text or "" for t in c.iter(_M + "t"))
            elif v is None or v.text is None:
                continue
            elif kind == "s":
                val = shared[int(v.text)]
            elif kind == "str":
                val = v.text
            elif kind == "b":
                val = "TRUE" if v.text == "1" else "FALSE"
            elif kind == "e":
                raise SpecError(f"{where} {ref}: 수식 오류 값({v.text})이 들어 있습니다.\n"
                                "  안 잡혔다면: 오류 글자가 그대로 학습 데이터가 됩니다.\n  그 칸을 고치세요.")
            else:
                s = int(c.get("s", 0))
                if s < len(plain) and not plain[s]:
                    raise SpecError(f"{where} {ref}: 날짜·퍼센트·자릿수 같은 서식이 있는 숫자 칸입니다.\n"
                                    "  안 잡혔다면: 화면에 보이는 값과 저장된 값이 달라(예: 00123 ↔ 123) 다른 값이 학습됩니다.\n"
                                    "  그 열을 `텍스트` 서식으로 바꿔 다시 입력하거나 CSV 로 저장하세요.")
                val = _number(v.text)
            cells[_col(ref)] = val
        if any(x.strip() for x in cells.values()):
            grid.append((int(row.get("r", len(grid) + 1)), cells))
    if not grid:
        raise SpecError(f"{where}: 비어 있습니다.\n  안 잡혔다면: 읽을 건이 없어 학습이 빈 데이터로 시작됩니다.\n  첫 줄에 열 이름을 적으세요.")

    _, head = grid[0]
    width = max(head) + 1
    names_ = [head.get(i, "").strip() for i in range(width)]
    if "" in names_:
        raise SpecError(f"{where} 1번째 줄: {names_.index('') + 1}번째 열의 이름이 비었습니다.\n"
                        "  안 잡혔다면: 어느 열인지 정해지지 않아 열 지정에서 고를 수 없습니다.\n  열 이름을 채우거나 그 열을 지우세요.")
    dup = sorted({n for n in names_ if names_.count(n) > 1})
    if dup:
        raise SpecError(f"{where}: 열 이름이 겹칩니다 {dup} — 같은 이름이 둘이면 어느 열인지 정해지지 않습니다")
    rows: List[Dict[str, str]] = []
    for ln, cells in grid[1:]:
        if max(cells) >= width and any(cells[i].strip() for i in cells if i >= width):
            raise SpecError(f"{where} {ln}번째 줄: 열 이름보다 값이 더 많습니다.\n"
                            "  안 잡혔다면: 이름 없는 열의 값이 조용히 버려집니다.\n  첫 줄에 열 이름을 적거나 그 값을 지우세요.")
        rows.append({n: cells.get(i, "") for i, n in enumerate(names_)})
    return rows
