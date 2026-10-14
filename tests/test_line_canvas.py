"""5노드를 기존 캔버스 화면에 카드 5장으로 그린다 (단계 5-A)."""

import os
import shutil

import pytest
import yaml

pytest.importorskip("PySide6", reason="편집기는 선택 사항이다 — pip install -e .[app]")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vlm_trainer.core import registry
from vlm_trainer.ui.api import Editor

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.join(os.path.dirname(HERE), "solutions")


@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def line(tmp_path):
    registry.load_builtin_nodes()
    dst = tmp_path / "solutions"
    shutil.copytree(SOL, dst, ignore=shutil.ignore_patterns("images", "runs", "edit_history*", "layout.yaml", "series"))
    return str(dst / "vlm_open" / "projects" / "01_open" / "line.yaml")


def test_the_line_opens_as_five_cards_joined_by_four_arrows(app, line):
    from vlm_trainer.ui.app.window import EditorWindow

    w = EditorWindow(Editor.open_line(line))
    assert len(w.canvas.cards) == 5 and len(w.canvas.chains) == 4 and not w.canvas.wires
    labels = [c.shown.label for c in sorted(w.canvas.cards.values(), key=lambda c: c.shown.lane)]
    assert [x[0] for x in labels] == ["①", "②", "③", "④", "⑤"]
    ys = [c.pos().y() for c in sorted(w.canvas.cards.values(), key=lambda c: c.shown.lane)]
    assert ys == sorted(ys)                                   # 위에서 아래로 한 줄
    assert not os.path.exists(os.path.join(os.path.dirname(line), "layout.yaml"))   # 열기만 해서는 파일이 안 생긴다


def test_moving_a_card_is_remembered_in_layout_yaml_and_the_arrow_follows(app, line):
    from vlm_trainer.ui.app.window import EditorWindow

    w = EditorWindow(Editor.open_line(line))
    sid = next(iter(w.canvas.cards))
    w._move_node(sid, 480, 160)
    saved = yaml.safe_load(open(os.path.join(os.path.dirname(line), "layout.yaml"), encoding="utf-8"))
    assert saved["nodes"][sid] == [480, 160]
    w2 = EditorWindow(Editor.open_line(line))
    assert w2.canvas.cards[sid].pos().x() == 480


def test_wiring_and_removing_are_refused_on_a_line(app, line):
    from vlm_trainer.ui.app.window import EditorWindow

    e = Editor.open_line(line)
    EditorWindow(e)
    assert not e.disconnect("a:b")["ok"] and not e.remove_node(next(iter(e.compiled.nodes)))["ok"]


def test_a_card_unfolds_into_its_inner_boxes_and_folds_back(app, line):
    from vlm_trainer.ui.app.window import EditorWindow
    from vlm_trainer.ui.line_model import _members

    e = Editor.open_line(line)
    w = EditorWindow(e)
    st = e.line_describe()["stations"][3]                       # ④ 모델 학습
    assert e.toggle_expand(st["id"])["ok"]
    w.reload_graph()
    inner = [c.shown for c in w.canvas.cards.values() if c.shown.station == st["id"]]
    assert len(inner) >= 2 and st["id"] not in w.canvas.cards and len(w.canvas.cards) == 4 + len(inner)
    covered = {i for s in inner for i in (s.state_ids or [s.id])}
    assert covered == set(_members(e.compiled, st["members"]))   # 안쪽 상자가 하나도 안 빠졌다
    assert w.canvas.wires and len(w.canvas.chains) == 4          # 이웃 카드와는 화살표로 이어진다
    lanes = [c.shown.lane for c in w.canvas.cards.values()]
    assert len(set(lanes)) == len(lanes) - len(inner) + len({s.lane for s in inner})   # 안쪽 상자는 레인이 겹치지 않는다
    assert not e.disconnect(next(iter(w.canvas.wires)).dst.ref)["ok"]                   # 펼쳐도 고치지는 못한다
    assert e.toggle_expand(st["id"])["ok"]
    w.reload_graph()
    assert len(w.canvas.cards) == 5 and not w.canvas.wires


def test_expanding_every_card_keeps_all_inner_boxes_and_a_wire_or_arrow_between_cards(app, line):
    from vlm_trainer.ui.app.window import EditorWindow
    from vlm_trainer.ui.layout import fold

    e = Editor.open_line(line)
    for s in e.line_describe()["stations"]:
        e.toggle_expand(s["id"])
    w = EditorWindow(e)
    boxes, _ = fold(e.compiled, ())
    assert set(w.canvas.cards) == set(boxes)                     # 열한 상자 전부(소속 없는 것이 없다)


def _pick(w, station_index):
    sid = w.editor.line_describe()["stations"][station_index]["id"]
    w._on_pick(sid)
    return sid


def _table(w):
    from PySide6 import QtWidgets

    return w.inspector.widget().widget().findChild(QtWidgets.QTableWidget)


def test_picking_a_card_shows_its_basic_settings_and_expert_mode_adds_the_rest(app, line):
    from vlm_trainer.ui.app.window import EditorWindow

    w = EditorWindow(Editor.open_line(line))
    _pick(w, 3)                                              # ④ 모델 학습
    basic = _table(w).rowCount()
    assert basic >= 1 and "학습량" in _table(w).item(0, 0).text()
    w.act_expert.setChecked(True)
    assert _table(w).rowCount() > basic                      # 전문가: 숫자 직접 입력과 안쪽 상자 값
    w.act_expert.setChecked(False)
    assert _table(w).rowCount() == basic


def test_choosing_an_amount_edits_the_line_and_saving_round_trips(app, line):
    from PySide6 import QtWidgets
    from vlm_trainer.ui.app.window import EditorWindow

    e = Editor.open_line(line)
    w = EditorWindow(e)
    sid = _pick(w, 3)
    combo = _table(w).cellWidget(0, 1)
    assert isinstance(combo, QtWidgets.QComboBox)
    combo.setCurrentText("많게")
    combo.activated.emit(combo.currentIndex())
    app.processEvents()                                       # 값 적용은 입력칸 신호가 끝난 뒤로 미룬다
    assert e.dirty and [s for s in e.line_describe()["stations"] if s["id"] == sid][0]["amount"]["value"] == "많게"
    w.on_save()
    assert not e.dirty
    again = Editor.open_line(line)
    assert [s for s in again.line_describe()["stations"] if s["id"] == sid][0]["amount"]["value"] == "많게"


def test_dropping_a_library_node_onto_a_line_is_refused_with_the_reason(app, line):
    from vlm_trainer.ui.app.window import EditorWindow

    w = EditorWindow(Editor.open_line(line))
    before = len(w.canvas.cards)
    w._add_node("source.field", 0, 0)
    assert "5노드(Line)" in w.lbl_refuse.text() and len(w.canvas.cards) == before


# ── 단계 5-D: 상태·NG 표 ───────────────────────────────────────────────
def _snap(ed, tmp_path):
    import json

    cg = ed.compiled
    snap = {
        "run_id": "t", "phase": "running", "order": list(cg.order), "total": 10, "processed": 10,
        "node_state": {n: {"success": 8} for n in cg.order} | {"n_guard": {"success": 8, "failed": 2}},
        "node_ms": {}, "cache": {}, "quarantine_total": 2,
        "quarantine": [{"sample_key": f"o010{i}", "node_id": "n_guard",
                        "cause": "정답 어휘가 프롬프트에 있다: ['기어']", "hint": "템플릿에서 뺀다"} for i in (1, 2)],
    }
    p = tmp_path / "progress.json"
    p.write_text(json.dumps(snap), encoding="utf-8")
    ed.progress_path = str(p)


def test_ejected_samples_color_the_cards_and_fill_the_ng_table_under_the_canvas(app, line, tmp_path):
    from vlm_trainer.ui.app.window import EditorWindow

    ed = Editor.open_line(line)
    _snap(ed, tmp_path)
    w = EditorWindow(ed)
    cards = {c.shown.label[0]: c for c in w.canvas.cards.values()}
    assert cards["③"].state == "partial" and cards["③"].state_extra.endswith("NG 2")
    assert cards["②"].state == "success"
    d = w.ng_dock
    assert d.tbl_ng.rowCount() == 2 and d.lbl_ng.text().startswith("NG 2건")
    assert d.tbl_ng.item(0, 2).text() == "③ 정답 형태"
    d.tbl_ng.selectRow(1)
    assert d.lbl_sum.text().startswith("③ 정답 형태 — ") and not d.detail.isVisibleTo(w)
    d.btn_more.setChecked(True)
    assert "o0102" in d.detail.toPlainText() and "해결: 템플릿에서 뺀다" in d.detail.toPlainText()
    w.reload_graph()                                          # 다시 그려도 색이 남는다
    assert {c.shown.label[0]: c for c in w.canvas.cards.values()}["③"].state == "partial"


def test_the_status_line_names_a_card_in_shop_language_and_a_line_has_no_preview(app, line):
    from vlm_trainer.ui.app.window import EditorWindow

    w = EditorWindow(Editor.open_line(line))
    sid = next(i for i, c in w.canvas.cards.items() if c.shown.label.startswith("③"))
    w._on_pick(sid)
    assert w.lbl_run.text().startswith("③") and "Station" not in w.lbl_run.text()
    assert all(not a.isVisible() for a in w._preview_acts)


def test_fixing_a_row_from_the_dock_writes_the_csv_and_rereads(app, line, tmp_path):
    from vlm_trainer.engine import table_edit
    from vlm_trainer.ui.app.window import EditorWindow

    w = EditorWindow(Editor.open_line(line))
    csvp = tmp_path / "t.csv"
    csvp.write_bytes("번호,사진,질문,부품\r\na1,x.png,무슨 부품입니까?,기어\r\n".encode("utf-8-sig"))
    ss = w.editor.line_data["sample_space"]
    ss["index"], ss["key"] = str(csvp), "번호"
    d = w.ng_dock
    d._show_fix("a1")
    assert d.fix_box.isVisibleTo(w) and d.tbl_fix.rowCount() == 4
    d.tbl_fix.item(3, 1).setText("볼트")
    d._fix_save()
    assert table_edit.row_values(str(csvp), "번호", "a1")["부품"] == "볼트"
    d._fix_exclude(True, confirm=False)
    assert csvp.read_bytes().decode("utf-8-sig").splitlines()[1].endswith(",x")
    old = w.editor
    w.reread(confirm=False)
    assert w.editor is not old and d.editor is w.editor and len(w.canvas.cards) == 5
