"""엑셀(.xlsx) 읽기 — 표준 라이브러리만으로 한 시트를 CSV 와 같은 규칙의 표로 읽는다.

시험용 .xlsx 는 이 파일이 직접 만든다(zip 안의 XML). 엑셀이 없어도 돈다.
"""

import csv
import os
import zipfile
from xml.sax.saxutils import escape

import pytest

from vlm_trainer.core.errors import SpecError
from vlm_trainer.engine import samples, xlsx

# 셀 서식 이름 → 서식 번호. 164 는 `00000` 이라는 사용자 서식이다.
FMT = {"일반": 0, "날짜": 14, "퍼센트": 9, "0.00": 2, "자릿수": 164}


def make_xlsx(path, sheets, *, merge=False):
    """sheets: {시트 이름: [[칸, ...], ...]}. 칸은 문자열, 숫자, 또는 (숫자, 서식 이름)이다. None 은 빈 칸."""
    shared, sst = {}, []

    def sid(s):
        if s not in shared:
            shared[s] = len(sst)
            sst.append(s)
        return shared[s]

    sheet_xml = []
    for rows in sheets.values():
        out = []
        for r, row in enumerate(rows, 1):
            cs = []
            for c, v in enumerate(row):
                ref = f"{chr(65 + c)}{r}"
                if v is None:
                    continue
                if isinstance(v, str):
                    cs.append(f'<c r="{ref}" t="s"><v>{sid(v)}</v></c>')
                else:
                    val, fmt = v if isinstance(v, tuple) else (v, "일반")
                    cs.append(f'<c r="{ref}" s="{list(FMT).index(fmt)}"><v>{val}</v></c>')
            out.append(f'<row r="{r}">{"".join(cs)}</row>')
        mc = '<mergeCells count="1"><mergeCell ref="A1:B1"/></mergeCells>' if merge else ""
        sheet_xml.append('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                         f'<sheetData>{"".join(out)}</sheetData>{mc}</worksheet>')
    names = list(sheets)
    wb = ('<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
          + "".join(f'<sheet name="{escape(n)}" sheetId="{i}" r:id="rId{i}"/>' for i, n in enumerate(names, 1))
          + "</sheets></workbook>")
    rels = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(names) + 1))
            + "</Relationships>")
    xfs = "".join(f'<xf numFmtId="{FMT[k]}"/>' for k in FMT)
    styles = ('<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<numFmts count="1"><numFmt numFmtId="164" formatCode="00000"/></numFmts>'
              f'<cellXfs count="{len(FMT)}">{xfs}</cellXfs></styleSheet>')
    sst_xml = ('<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
               + "".join(f"<si><t>{escape(s)}</t></si>" for s in sst) + "</sst>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        z.writestr("xl/styles.xml", styles)
        z.writestr("xl/sharedStrings.xml", sst_xml)
        for i, x in enumerate(sheet_xml, 1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", x)
    return str(path)


HEAD = ["번호", "사진", "질문", "부품"]
BODY = [["a1", "img/a1.png", "무슨 부품입니까?", "기어"], ["a2", "img/a2.png", "무슨 부품입니까?", "나사"]]


def test_a_sheet_reads_like_the_same_csv(tmp_path):
    p = make_xlsx(tmp_path / "t.xlsx", {"시트1": [HEAD] + BODY})
    with open(tmp_path / "t.csv", "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows([HEAD] + BODY)
    assert samples.read_table(p) == samples.read_table(str(tmp_path / "t.csv"))


def test_numbers_are_text_without_a_trailing_point_zero_and_empty_cells_and_rows_stay_straight(tmp_path):
    p = make_xlsx(tmp_path / "t.xlsx", {"S": [["id", "n", "m", "x"], ["a", 12.0, None, "끝"], [None, None, None, None],
                                             ["b", 0.1, 3, (7.5, "0.00")]]})
    assert samples.read_table(p) == [{"id": "a", "n": "12", "m": "", "x": "끝"},
                                     {"id": "b", "n": "0.1", "m": "3", "x": "7.5"}]


def test_the_sheet_comes_after_a_hash_and_the_first_sheet_is_the_default(tmp_path):
    p = make_xlsx(tmp_path / "t.xlsx", {"첫째": [["a"], ["1"]], "둘째": [["b"], ["2"]]})
    assert xlsx.sheet_names(p) == ["첫째", "둘째"]
    assert samples.read_table(p) == [{"a": "1"}]
    assert samples.read_table(p, "둘째") == [{"b": "2"}]
    assert xlsx.split_sheet(p + "#둘째") == (p, "둘째") and xlsx.split_sheet("a#b.csv") == ("a#b.csv", "")
    space = samples.load({"index": p + "#둘째", "key": "b", "splits": {"strategy": "random", "ratios": {"train": 1}}},
                         str(tmp_path))
    assert space.keys() == ["2"]
    with pytest.raises(SpecError, match="시트 '셋째'가 없습니다"):
        samples.read_table(p, "셋째")


@pytest.mark.parametrize("fmt", ["날짜", "퍼센트", "자릿수"])
def test_a_cell_whose_shown_value_may_differ_from_the_stored_one_is_refused(tmp_path, fmt):
    p = make_xlsx(tmp_path / "t.xlsx", {"S": [["id", "v"], ["a", (123, fmt)]]})
    with pytest.raises(SpecError, match="B2") as e:
        samples.read_table(p)
    assert "텍스트" in str(e.value) and "CSV" in str(e.value)


def test_merged_cells_blank_or_duplicate_headers_and_stray_values_are_refused(tmp_path):
    with pytest.raises(SpecError, match="병합"):
        samples.read_table(make_xlsx(tmp_path / "a.xlsx", {"S": [["a", "b"], ["1", "2"]]}, merge=True))
    with pytest.raises(SpecError, match="이름이 비었습니다"):
        samples.read_table(make_xlsx(tmp_path / "b.xlsx", {"S": [["a", None, "c"], ["1", "2", "3"]]}))
    with pytest.raises(SpecError, match="겹칩니다"):
        samples.read_table(make_xlsx(tmp_path / "c.xlsx", {"S": [["a", "a"], ["1", "2"]]}))
    with pytest.raises(SpecError, match="3번째 줄: 열 이름보다 값이 더 많습니다"):
        samples.read_table(make_xlsx(tmp_path / "d.xlsx", {"S": [["a"], ["1"], ["2", "x"]]}))
    with pytest.raises(SpecError, match="비어 있습니다"):
        samples.read_table(make_xlsx(tmp_path / "e.xlsx", {"S": [[None]]}))


def test_an_old_xls_and_a_broken_file_say_what_to_do(tmp_path):
    old = tmp_path / "t.xls"
    old.write_bytes(b"x")
    with pytest.raises(SpecError, match="구형 엑셀") as e:
        samples.read_table(str(old))
    assert ".xlsx" in str(e.value) and "CSV" in str(e.value)
    bad = tmp_path / "t.xlsx"
    bad.write_bytes(b"not a zip")
    with pytest.raises(SpecError, match="깨졌습니다"):
        samples.read_table(str(bad))


def test_the_header_reports_sheets_and_the_dialog_picks_one(tmp_path):
    pytest.importorskip("PySide6", reason="편집기는 선택 사항이다")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from vlm_trainer.spec import scaffold
    from vlm_trainer.ui import line_data as LD
    from vlm_trainer.ui.api import Editor
    from vlm_trainer.ui.app.data_dialog import DataDialog

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path, _ = scaffold.new_line(str(tmp_path / "proj"))
    make_xlsx(tmp_path / "proj" / "t.xlsx", {"첫째": [["a", "b"], ["1", "2"]], "둘째": [["c"], ["3"], ["4"]]})
    h = LD.read_header(str(tmp_path / "proj"), "t.xlsx#둘째")
    assert h["columns"] == ["c"] and h["count"] == 2 and h["sheets"] == ["첫째", "둘째"] and h["sheet"] == "둘째"
    d = DataDialog(Editor.open_line(path))
    d.ed_path.setText("t.xlsx")
    d._reread_table()
    assert d.cb_sheet.count() == 2 and d.header == ["a", "b"] and d._index() == "t.xlsx#첫째"
    d.cb_sheet.setCurrentIndex(1)                       # 사람이 시트를 바꾼다
    assert d.header == ["c"] and d._index() == "t.xlsx#둘째" and "2건" in d.lbl_table.text()
    d.ed_path.setText("t.csv")
    d._reread_table()
    assert d.cb_sheet.count() == 0 and d._index() == "t.csv"
