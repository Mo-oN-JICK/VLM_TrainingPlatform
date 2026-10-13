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
