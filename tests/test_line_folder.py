"""폴더 입력 방식 — 폴더·질문·정답 항목을 한 번에 받고, ③ 에서 정답을 넣어 저장까지 (단계 6)."""

from __future__ import annotations

import os
import shutil

import pytest
import yaml

from vlm_trainer.cli.main import main
from vlm_trainer.core import registry
from vlm_trainer.core.errors import SpecError
from vlm_trainer.engine import samples
from vlm_trainer.spec import scaffold
from vlm_trainer.ui import line_data as LD
from vlm_trainer.ui.api import Editor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "solutions", "vlm_open", "data", "open_parts", "images")
ROWS = [{"항목": "색상", "이름": "공의 색상", "종류": "목록", "허용값": "검정색|파란색|흰색"}]


@pytest.fixture
def proj(tmp_path):
    """새 5노드 프로젝트와, 하위 폴더 셋에 사진을 나눠 담은 폴더."""
    registry.load_builtin_nodes()
    pics = tmp_path / "pics"
    names = sorted(os.listdir(SRC))[:12]
    for i, n in enumerate(names):
        sub = pics / ["검정색", "파란색", "흰색"][i % 3]
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(SRC, n), sub / n)
    path, _ = scaffold.new_line(str(tmp_path / "proj"))
    return path, str(pics)


def test_the_image_mark_goes_only_at_the_start_or_the_end():
    assert LD.parse_question("{image} 공의 색상은?") == ("공의 색상은?", "prepend")
    assert LD.parse_question("공의 색상은? {Image}") == ("공의 색상은?", "append")
    assert LD.parse_question("공의 색상은?") == ("공의 색상은?", "prepend")
    with pytest.raises(SpecError, match="가운데"):
        LD.parse_question("이 {image} 공의 색상은?")
    with pytest.raises(SpecError, match="비었습니다"):
        LD.parse_question("{image}")
    assert LD.question_text("공의 색상은?", "prepend") == "{image} 공의 색상은?"


def test_choosing_a_folder_with_question_and_items_compiles_at_once_and_saves(proj):
    path, pics = proj
    ed = Editor.open_line(path)
    assert ed.compiled is None                                  # 새 프로젝트는 아직 컴파일되지 않는다
    res = ed.line_use_folder(pics, "{image} 공의 색상은?", ROWS, val_pct=25)
    assert res["ok"], res
    assert res["counts"]["total"] == 12 and ed.compiled is not None and ed.valid
    assert ed.line_save()["ok"]
    data = yaml.safe_load(open(path, encoding="utf-8"))
    assert data["folder"] == "../pics" and data["sample_space"]["index"] == "data.csv"
    q = LD.station(data, "question")["params"]
    assert (q["template"], q["policy"]) == ("공의 색상은?", "prepend")

    ans = ed.line_answers()
    assert ans["editable"] and len(ans["rows"]) == 12 and ans["cols"] == ["색상"]
    assert ans["rows"][0]["key"].startswith("검정색__")


def test_a_refused_folder_setup_leaves_no_table_behind(proj):
    path, pics = proj
    ed = Editor.open_line(path)
    res = ed.line_use_folder(pics, "공의 {image} 색상은?", ROWS)
    assert not res["ok"] and "가운데" in res["detail"]
    assert not os.path.exists(os.path.join(os.path.dirname(path), "data.csv"))


def test_answers_wait_for_save_and_fill_from_subfolders_never_overwrites(proj):
    path, pics = proj
    ed = Editor.open_line(path)
    assert ed.line_use_folder(pics, "{image} 공의 색상은?", ROWS)["ok"]
    first = ed.line_answers()["rows"][0]["key"]
    ed.line_set_answer(first, "색상", "흰색")                    # 사람이 먼저 적은 값
    csv_path = os.path.join(os.path.dirname(path), "data.csv")
    assert "흰색" not in open(csv_path, encoding="utf-8-sig").read().split("\n")[1]   # 저장 전에는 표에 없다
    got = ed.line_fill_from_folder("색상")
    assert (got["filled"], got["kept"], got["not_allowed"]) == (11, 1, 0)
    assert ed.line_save()["ok"]
    rows = {r["건"]: r for r in samples.read_table(csv_path)}
    assert rows[first]["색상"] == "흰색"                          # 덮지 않았다
    assert all(r["색상"] == k.split("__")[0] for k, r in rows.items() if k != first)


def test_reopening_syncs_new_and_missing_photos_and_items_and_val_ratio_edit(proj, tmp_path):
    path, pics = proj
    ed = Editor.open_line(path)
    assert ed.line_use_folder(pics, "{image} 공의 색상은?", ROWS)["ok"] and ed.line_save()["ok"]
    gone = sorted(os.listdir(os.path.join(pics, "흰색")))[0]
    os.remove(os.path.join(pics, "흰색", gone))
    ed2 = Editor.open_line(path)
    assert ed2.folder_sync["excluded"] == 1 and len(ed2.line_answers()["rows"]) == 11

    rows = ROWS + [{"항목": "크기", "이름": "", "종류": "문장", "허용값": ""}]
    assert ed2.line_set_schema(rows)["ok"]
    assert ed2.line_answers()["cols"] == ["색상", "크기"]
    assert ed2.line_set_val_pct(50)["ok"]
    assert ed2.line_data["sample_space"]["splits"]["ratios"] == {"train": 0.5, "val": 0.5}
    assert ed2.line_set_question("공의 색상은? {image}")["ok"]
    assert LD.station(ed2.line_data, "question")["params"]["policy"] == "append"


def test_the_cli_does_the_same_folder_setup_and_fill(proj, capsys):
    path, pics = proj
    assert main(["line-folder", path, pics, "--question", "{image} 공의 색상은?",
                 "--item", "색상:목록:검정색|파란색|흰색"]) == 0
    assert "12건" in capsys.readouterr().out
    assert main(["row-fill-folder", path, "색상"]) == 0
    assert "12건 채움" in capsys.readouterr().out
    rows = samples.read_table(os.path.join(os.path.dirname(path), "data.csv"))
    assert {r["색상"] for r in rows} == {"검정색", "파란색", "흰색"}


def test_a_jsonl_table_shows_answers_read_only_with_the_reason():
    registry.load_builtin_nodes()
    ed = Editor.open_line(os.path.join(os.path.dirname(HERE), "solutions", "vlm_open", "projects", "01_open", "line.yaml"))
    ans = ed.line_answers()
    assert ans["ok"] and not ans["editable"] and "CSV 뿐" in ans["why"] and ans["cols"] == ["part_type", "orientation"]


# ── 화면 ───────────────────────────────────────────────────────────────
pytest.importorskip("PySide6", reason="편집기는 선택 사항이다 — pip install -e .[app]")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _sid(w, mark):
    return next(i for i, c in w.canvas.cards.items() if c.shown.label.startswith(mark))


def test_a_new_project_shows_five_cards_and_the_image_panel_before_it_compiles(app, proj):
    from PySide6 import QtWidgets
    from vlm_trainer.ui.app.window import EditorWindow

    path, _ = proj
    w = EditorWindow(Editor.open_line(path))
    assert w.editor.compiled is None and len(w.canvas.cards) == 5 and len(w.canvas.chains) == 4
    w.pick_first_card()                                         # 새 프로젝트는 ① 이 골라진 채 열린다
    radios = [r for r in w.inspector.findChildren(QtWidgets.QRadioButton)]
    assert {r.text() for r in radios} == {"사진 폴더", "표 (CSV · 엑셀)"}
    assert next(r for r in radios if r.text().startswith("표")).isChecked()


def test_panels_after_choosing_a_folder_preview_question_and_answers(app, proj):
    from PySide6 import QtWidgets
    from vlm_trainer.ui.app.window import EditorWindow

    path, pics = proj
    ed = Editor.open_line(path)
    assert ed.line_use_folder(pics, "{image} 공의 색상은?", ROWS)["ok"] and ed.line_save()["ok"]
    w = EditorWindow(Editor.open_line(path))
    w._on_pick(_sid(w, "①"))
    assert w.preview_pair                                         # 원본과 모델이 보는 크기 두 장
    caps = [lb.text() for lb in w.inspector.findChildren(QtWidgets.QLabel)]
    assert "모델이 보는 크기\n448×448" in caps

    w._on_pick(_sid(w, "②"))
    le = w.inspector.findChild(QtWidgets.QLineEdit, "question")
    assert le.text() == "{image} 공의 색상은?"
    w._line_apply(w.editor.line_set_question, "공의 {image} 색상은?")
    assert w.lbl_refuse.text().startswith("거부") and "가운데" in w.lbl_refuse.text()

    w._on_pick(_sid(w, "③"))
    t = w.answers_table
    assert t.rowCount() == 12 and t.item(0, 2).text() == "Q. {image} 공의 색상은?"
    assert t.horizontalHeaderItem(3).text() == "A. 공의 색상"
    t.item(0, 3).setText("검정색")
    assert w.editor.answer_edits[t.item(0, 1).text()] == {"색상": "검정색"} and "저장 안 됨" in w.windowTitle()
    w.on_save()
    rows = {r["건"]: r for r in samples.read_table(os.path.join(os.path.dirname(path), "data.csv"))}
    assert rows[t.item(0, 1).text()]["색상"] == "검정색"
