"""NG 한 건을 표(CSV)에서 고치거나 제외한다 — 그 칸만 쓰고, 원본을 남기고, 지우지 않는다."""

from __future__ import annotations

import os
import shutil

import pytest
import yaml

from vlm_trainer.cli.main import main
from vlm_trainer.core import registry
from vlm_trainer.core.errors import SpecError
from vlm_trainer.engine import samples, table_edit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOL = os.path.join(ROOT, "solutions")

TEXT = ("번호,사진,질문,부품\r\n"
        "a1,img/a1.png,\"무슨 부품입니까, 알려 주세요\",기어\r\n"
        "a2,img/a2.png,무슨 부품입니까?,나사\r\n"
        "a3,img/a3.png,무슨 부품입니까?,너트\r\n")


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes(b"\xef\xbb\xbf" + TEXT.encode("utf-8"))
    return str(p)


def _raw(p):
    return open(p, "rb").read()


def test_only_that_cell_changes_and_the_file_keeps_its_bom_newlines_and_quotes(csv_path):
    before = _raw(csv_path)
    assert table_edit.set_cells(csv_path, "번호", "a2", {"부품": "볼트"}) == ["부품"]
    after = _raw(csv_path)
    assert after == before.replace("나사".encode(), "볼트".encode())
    assert after.startswith(b"\xef\xbb\xbf") and b"\r\n" in after and "\"무슨 부품입니까, 알려 주세요\"".encode() in after


def test_the_original_is_kept_once_and_the_first_copy_is_never_overwritten(csv_path):
    first = _raw(csv_path)
    bak = csv_path[:-4] + ".bak.csv"
    table_edit.set_cells(csv_path, "번호", "a1", {"부품": "핀"})
    table_edit.set_cells(csv_path, "번호", "a3", {"부품": "와셔"})
    assert _raw(bak) == first                      # 두 번 고쳐도 가장 처음 모습이다


def test_an_unchanged_value_writes_nothing_and_makes_no_backup(csv_path):
    before = _raw(csv_path)
    assert table_edit.set_cells(csv_path, "번호", "a1", {"부품": "기어"}) == []
    assert _raw(csv_path) == before and not os.path.exists(csv_path[:-4] + ".bak.csv")


def test_refusals_say_which_row_or_column_is_wrong(csv_path):
    with pytest.raises(SpecError, match="없는 열"):
        table_edit.set_cells(csv_path, "번호", "a1", {"무게": "3"})
    with pytest.raises(SpecError, match="건 구분 열"):
        table_edit.set_cells(csv_path, "번호", "a1", {"번호": "z9"})
    with pytest.raises(SpecError, match="'zz'가 없습니다"):
        table_edit.set_cells(csv_path, "번호", "zz", {"부품": "x"})
    with pytest.raises(SpecError, match="건 구분 열 '키'가 없습니다"):
        table_edit.set_cells(csv_path, "키", "a1", {"부품": "x"})
    open(csv_path, "ab").write("a1,img/x.png,q,중복\r\n".encode())
    with pytest.raises(SpecError, match="2줄에 있습니다"):
        table_edit.set_cells(csv_path, "번호", "a1", {"부품": "x"})


def test_an_excel_or_jsonl_table_is_not_written_and_the_message_says_to_save_as_csv(tmp_path):
    for name in ("t.xlsx", "t.jsonl"):
        with pytest.raises(SpecError, match="CSV 뿐입니다") as e:
            table_edit.set_cells(str(tmp_path / name), "번호", "a1", {"부품": "x"})
        assert "CSV UTF-8" in str(e.value)


def test_excluding_adds_a_column_keeps_the_row_and_the_reader_skips_and_counts_it(csv_path, tmp_path):
    table_edit.set_excluded(csv_path, "번호", "a2", True)
    text = _raw(csv_path).decode("utf-8-sig")
    assert text.splitlines()[0].endswith(",제외") and "a2,img/a2.png,무슨 부품입니까?,나사,x" in text
    assert len(text.splitlines()) == 4                                   # 행은 그대로 4줄
    space = samples.load({"index": csv_path, "key": "번호", "splits": {"strategy": "random", "ratios": {"train": 1}}},
                         str(tmp_path))
    assert space.keys() == ["a1", "a3"] and space.excluded == 1
    table_edit.set_excluded(csv_path, "번호", "a2", False)                   # 되살린다
    space = samples.load({"index": csv_path, "key": "번호", "splits": {"strategy": "random", "ratios": {"train": 1}}},
                         str(tmp_path))
    assert space.keys() == ["a1", "a2", "a3"] and space.excluded == 0


def test_undoing_an_exclusion_that_was_never_made_does_not_touch_the_file(csv_path):
    before = _raw(csv_path)
    assert table_edit.set_excluded(csv_path, "번호", "a1", False) == []
    assert _raw(csv_path) == before


def test_a_value_other_than_x_in_the_exclude_column_is_refused_when_read(csv_path, tmp_path):
    table_edit.set_excluded(csv_path, "번호", "a1", True)
    text = _raw(csv_path).decode("utf-8-sig").replace(",x\r\n", ",o\r\n")
    open(csv_path, "wb").write(text.encode("utf-8"))
    with pytest.raises(SpecError, match="제외` 칸 값 'o'") as e:
        samples.load({"index": csv_path, "key": "번호", "splits": {"strategy": "random", "ratios": {"train": 1}}}, str(tmp_path))
    assert "x 만 적고" in str(e.value)
    with pytest.raises(SpecError, match="x 만 적을 수"):
        table_edit.set_cells(csv_path, "번호", "a3", {"제외": "o"})


def test_the_cli_edits_and_excludes_through_the_same_functions(csv_path, tmp_path, capsys):
    spec = tmp_path / "line.yaml"
    spec.write_text(yaml.safe_dump({"sample_space": {"index": "t.csv", "key": "번호"}}, allow_unicode=True), encoding="utf-8")
    assert main(["row-set", str(spec), "a1", "부품=핀", "질문=다시 쓴 질문"]) == 0
    assert "부품, 질문 고침" in capsys.readouterr().out
    assert table_edit.row_values(csv_path, "번호", "a1")["부품"] == "핀"
    assert main(["row-exclude", str(spec), "a3"]) == 0 and "제외함" in capsys.readouterr().out
    assert main(["row-exclude", str(spec), "a3", "--undo"]) == 0 and "되살림" in capsys.readouterr().out
    assert main(["row-set", str(spec), "a1", "부품핀"]) != 0                    # 열=값 꼴이 아니면 거부


# ── 화면 ───────────────────────────────────────────────────────────────
pytest.importorskip("PySide6", reason="편집기는 선택 사항이다 — pip install -e .[app]")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app, tmp_path):
    """작은 CSV 를 가리키게 한 Line 창. NG 표를 거치지 않고 고치기 칸을 바로 연다."""
    from vlm_trainer.ui.api import Editor
    from vlm_trainer.ui.app.line_window import LineWindow

    registry.load_builtin_nodes()
    dst = tmp_path / "solutions"
    shutil.copytree(SOL, dst, ignore=shutil.ignore_patterns("images", "runs", "edit_history*", "layout.yaml", "series"))
    line = str(dst / "vlm_open" / "projects" / "01_open" / "line.yaml")
    shutil.copy(os.path.join(HERE, "..", "solutions", "vlm_open", "data", "open_parts", "index.jsonl"), tmp_path / "x.jsonl") \
        if False else None
    csvp = tmp_path / "t.csv"
    csvp.write_bytes(b"\xef\xbb\xbf" + TEXT.encode("utf-8"))
    w = LineWindow(Editor.open_line(line))
    ss = w.editor.line_data["sample_space"]        # 창이 자기 편집기를 다시 만든 뒤에 표를 바꿔 끼운다
    ss["index"], ss["key"] = str(csvp), "번호"
    return w, str(csvp)


def test_the_fix_box_shows_the_row_and_saving_writes_only_the_changed_cell(window):
    w, p = window
    w._show_fix("a2")
    assert w.fix_box.isVisibleTo(w) and w.tbl_fix.rowCount() == 4
    from PySide6 import QtCore

    assert not w.tbl_fix.item(0, 1).flags() & QtCore.Qt.ItemIsEditable       # 건 이름 칸은 고칠 수 없다
    assert w.tbl_fix.item(3, 1).flags() & QtCore.Qt.ItemIsEditable
    w.tbl_fix.item(3, 1).setText("볼트")
    w._fix_save()
    assert table_edit.row_values(p, "번호", "a2")["부품"] == "볼트" and os.path.exists(p[:-4] + ".bak.csv")


def test_excluding_from_the_window_marks_x_and_a_non_csv_table_only_explains(window, tmp_path):
    w, p = window
    w._show_fix("a1")
    w._fix_exclude(True, confirm=False)
    assert _raw(p).decode("utf-8-sig").splitlines()[1].endswith(",x")      # 그 행 끝에 x
    w.editor.line_data["sample_space"]["index"] = str(tmp_path / "t.xlsx")
    w._show_fix("a1")
    assert not w.btn_fix.isEnabled() and "CSV 뿐" in w.lbl_fixmsg.text()
