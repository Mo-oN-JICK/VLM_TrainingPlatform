"""네이티브 편집기(PySide6) — 웹판이 HTML로 검사하던 성질을 위젯에 대고 본다.

웹 편집기를 걷어내며 HTML 문자열을 들여다보던 테스트 15개가 사라졌다. 그것들이
검사하던 성질은 화면을 무엇으로 그리든 유효하다 — 접힌 Procedure는 상자 하나로
보인다, 라이브러리에 노드가 다 있다, 경계가 카드에 표시된다 같은 것들이다.

**두 갈래로 나눈다.** 위젯이 필요 없는 것은 `ui/layout.py` 에 직접 묻고(빠르고
디스플레이가 필요 없다), 위젯이 필요한 것만 QApplication 을 띄운다.

`QT_QPA_PLATFORM=offscreen` 에서는 **폰트가 0종**이라 글자 너비나 클릭 대상 크기를
재는 검사는 여기서 하지 않는다. 구조만 본다.
"""

from __future__ import annotations

import os
import shutil

import pytest

from vlm_trainer.core import registry
from vlm_trainer.ui import layout as L
from vlm_trainer.ui.api import Editor, compat_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOLUTION = os.path.join(ROOT, "solutions", "dummy_ecg")

pytest.importorskip("PySide6", reason="편집기는 선택 사항이다 — pip install -e .[app]")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets

    registry.load_builtin_nodes()
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def ed(tmp_path):
    """**solutions 전체**를 복사한다. Procedure 가 solution 위쪽에 있어서
    프로젝트 디렉터리만 복사하면 컴파일되지 않는다.

    원본을 여는 것도 안 된다 — `Editor.open()` 은 edit_history 와 layout.yaml 을 쓴다.
    """
    registry.load_builtin_nodes()
    work = tmp_path / "solutions"
    shutil.copytree(os.path.join(ROOT, "solutions"), work,
                    ignore=shutil.ignore_patterns("data", "runs", "edit_history*", "layout.yaml"))
    return Editor.open(str(work / "dummy_ecg" / "projects" / "01_dummy" / "project.yaml"))


@pytest.fixture
def win(app, ed):
    from vlm_trainer.ui.app.window import EditorWindow

    return EditorWindow(ed)


# ── 위젯이 필요 없는 것 ─────────────────────────────────────────────────


def test_a_folded_procedure_is_one_box(ed):
    """접힌 Procedure 는 상자 하나로 보이고 안쪽 노드는 어디에도 나오지 않는다.
    25개 카드는 사람이 붙들 수 있는 수가 아니다."""
    shown, _ = L.fold(ed.compiled)
    assert "p_crop" in shown
    assert not any("/" in k for k in shown), "안쪽 노드가 새어 나왔다"
    assert shown["p_crop"].inner >= 1

    shown2, _ = L.fold(ed.compiled, {"p_crop"})
    assert "p_crop" not in shown2
    assert any(k.startswith("p_crop/") for k in shown2), "펼치면 안쪽이 보여야 한다"


def test_exposed_params_are_visible_while_folded(ed):
    """접힌 채로도 노출 파라미터는 고칠 수 있어야 한다 — 접기는 보는 방식일 뿐이다."""
    proc = next(p for p in ed.compiled.procedures if p["id"] == "p_crop")
    assert "topk" in (proc.get("exposed_params") or {})
    assert ed.set_param("p_crop", "topk", 3)["ok"]


def test_the_compat_table_says_why_a_port_is_refused(ed):
    """드래그 중에 꽂을 수 없는 자리를 죽이는 판단이 이 표에서 온다.
    캔버스는 타입을 따지지 않는다 — 판단하는 곳은 하나여야 한다."""
    compat = compat_matrix(ed.compiled)
    assert compat, "표가 비어 있으면 아무 데나 놓을 수 있게 된다"
    refused = [(a, b, why) for a, row in compat.items() for b, why in row.items() if why]
    assert refused, "모든 포트가 서로 호환된다면 게이트가 없는 것이다"
    assert all(why.strip() for _, _, why in refused), "거부에는 이유가 붙어야 한다"


def test_the_library_lists_every_registered_node(ed):
    """빈 그래프에서도 라이브러리는 온전하다."""
    lib = ed.library()
    assert len(lib) >= 25
    assert any(n["type"].startswith("adapt.image_resize") for n in lib)
    assert all(n["category"] and n["kind"] for n in lib)


# ── 위젯이 필요한 것 ────────────────────────────────────────────────────


def test_the_canvas_draws_one_card_per_box(win, ed):
    shown, _ = L.fold(ed.compiled, ed.expanded)
    assert set(win.canvas.cards) == set(shown)
    assert win.canvas.wires, "배선이 하나도 안 그려졌다"
    assert win.canvas.chips, "포트 칩이 없으면 배선을 잡을 자리가 없다"


def test_every_wire_ends_on_the_chips_it_connects(win):
    """배선이 칩에서 떨어진 허공에서 시작하면 어디로 가는지 읽을 수 없다."""
    for w in win.canvas.wires:
        a, b = w.path().pointAtPercent(0.0), w.path().currentPosition()
        assert abs(a.x() - w.src.anchor().x()) < 1.0
        assert abs(b.x() - w.dst.anchor().x()) < 1.0


def test_moving_a_card_drags_its_chips_and_wires(win):
    nid = next(iter(win.canvas.cards))
    chip = next(c for c in win.canvas.chips.values() if c.node_id == nid)
    before = chip.anchor().x()
    card = win.canvas.cards[nid]
    card.setPos(card.pos().x() + 64, card.pos().y())
    assert abs(chip.anchor().x() - before - 64) < 1.0, "칩이 카드를 안 따라왔다"


def test_the_materialize_boundary_shows_on_the_card(win, ed):
    """경계는 그래프 밖의 한 줄이지만 어디까지 미리 굽는지를 정한다.
    카드에 보이지 않으면 그 한 줄이 어디에 걸리는지 알 수 없다."""
    marked = {n for n, c in win.canvas.cards.items() if c.boundary}
    assert marked, "경계가 하나도 표시되지 않았다"
    assert marked <= set(win.canvas.cards)


def test_selecting_a_box_opens_its_settings(win, ed):
    """상자를 누르면 그 상자의 설정이 나온다. 누르기 전에는 안내 한 줄뿐이다."""
    from PySide6 import QtWidgets

    def labels():
        panel = win.inspector.widget().widget()
        return " ".join(w.text() for w in panel.findChildren(QtWidgets.QLabel))

    assert "상자를 하나 누르면" in labels()

    nid = "n_stats"
    win.canvas.cards[nid].setSelected(True)
    text = labels()
    assert nid in text
    assert "파라미터" in [g.title() for g in
                       win.inspector.widget().widget().findChildren(QtWidgets.QGroupBox)]


def test_an_unknown_box_is_refused_with_the_boxes_that_exist(ed):
    res = ed.set_param("없는상자", "x", 1)
    assert not res["ok"] and "있는 상자" in res["detail"]


def test_running_locks_editing(win):
    """실행은 디스크의 스펙을 도는데 편집은 메모리를 바꾼다 — 둘이 갈라지면
    화면과 다른 것이 돌아간다."""
    win.set_running(True)
    assert not win.act_run.isEnabled() and win.act_stop.isEnabled()
    assert not win.library.isEnabled() and not win.canvas.isInteractive()
    win.set_running(False)
    assert win.act_run.isEnabled() and not win.act_stop.isEnabled()
    assert win.library.isEnabled() and win.canvas.isInteractive()


def test_the_run_clock_reads_like_the_cli():
    """같은 값이 창과 터미널에서 다르게 보이면 둘 중 하나가 틀린 것처럼 읽힌다."""
    from vlm_trainer.cli.common import _took
    from vlm_trainer.ui.app.progress import fmt_ms

    for ms in (0, 440, 1234, 42_000, 95_000, 3_700_000):
        assert fmt_ms(ms) == _took(ms), ms


# ── 스트레스: 실제 이벤트로 두들긴다 ────────────────────────────────────


def test_random_edits_do_not_crash(win, ed):
    """실제 마우스 이벤트로 변경 경로를 무작위로 두들긴다.

    **세그폴트를 이 방식으로 잡았다.** 이벤트 핸들러 안에서 `scene.clear()` 를 부르면
    Qt 가 지금 처리 중인 아이템이 사라지고, 죽는 곳이 C++ 이라 파이썬 트레이스백이
    나오지 않는다. 위젯을 하나씩 호출하는 테스트로는 걸리지 않는 부류다.

    이 테스트가 통과한다는 것은 "죽지 않았다"는 뜻이고, 그것이 여기서 볼 전부다.
    """
    import random

    from PySide6 import QtCore, QtGui, QtWidgets

    c = win.canvas
    rng = random.Random(7)
    app = QtWidgets.QApplication.instance()

    def mouse(kind, pos):
        app.sendEvent(c.viewport(), QtGui.QMouseEvent(
            kind, QtCore.QPointF(pos), c.viewport().mapToGlobal(pos),
            QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))

    def wire_drag():
        outs = [k for k in c.chips if k.startswith("out:")]
        ins = [k for k in c.chips if k.startswith("in:")]
        if not outs or not ins:
            return
        a = c.mapFromScene(c.chips[rng.choice(outs)].anchor())
        b = c.mapFromScene(c.chips[rng.choice(ins)].anchor())
        mouse(QtCore.QEvent.MouseButtonPress, a)
        mouse(QtCore.QEvent.MouseMove, QtCore.QPoint((a.x() + b.x()) // 2, (a.y() + b.y()) // 2))
        mouse(QtCore.QEvent.MouseButtonRelease, b)

    def move_card():
        nid = rng.choice(list(c.cards))
        card = c.cards[nid]
        card.setPos(card.pos().x() + rng.randint(-48, 48),
                    max(0, card.pos().y() + rng.randint(-48, 48)))
        win._move_node(nid, int(card.pos().x()), int(card.pos().y()) - card.top_h)

    def drop_node():
        mime = QtCore.QMimeData()
        mime.setData("application/x-vlmt-node", QtCore.QByteArray(b"ts.stats@1.0.0"))
        pos = QtCore.QPoint(rng.randint(60, 600), rng.randint(60, 400))
        app.sendEvent(c, QtGui.QDropEvent(QtCore.QPointF(pos), QtCore.Qt.CopyAction, mime,
                                          QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))

    acts = [
        wire_drag, move_card, drop_node,
        lambda: c.cards[rng.choice(list(c.cards))].setSelected(True),
        lambda: c.disconnectRequested.emit(sorted(c.occupied)[rng.randrange(len(c.occupied))])
        if c.occupied else None,
        lambda: win._mutate(ed.undo, "undo"),
        c.fit,
    ]
    for _ in range(60):
        rng.choice(acts)()
        for _ in range(4):
            app.processEvents()

    assert win.canvas.cards, "그래프가 통째로 사라졌다"
