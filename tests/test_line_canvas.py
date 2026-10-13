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
