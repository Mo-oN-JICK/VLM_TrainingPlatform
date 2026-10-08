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
    assert not win.library.isEnabled()
    assert win.canvas._read_only, "그래프를 바꿀 수 있는 상태다"
    win.set_running(False)
    assert win.act_run.isEnabled() and not win.act_stop.isEnabled()
    assert win.library.isEnabled() and not win.canvas._read_only


def test_running_does_not_lock_looking(win):
    """**이것이 버그였다.** `setInteractive(False)` 로 캔버스의 마우스 이벤트를 통째로
    껐더니 팬도 휠 줌도 선택도 죽었다 — 실행 중에 아래쪽 노드를 보러 갈 수가 없다.
    실행 중에 제일 하고 싶은 일이 그것인데.
    """
    from PySide6 import QtWidgets

    win.set_running(True)
    assert win.canvas.isInteractive(), "마우스 이벤트가 통째로 꺼졌다"

    # 팬이 실제로 먹는다
    before = win.canvas.horizontalScrollBar().value()
    win.canvas._panning = True
    win.canvas.horizontalScrollBar().setValue(before + 40)
    assert win.canvas.horizontalScrollBar().value() != before or before == 0
    win.canvas._panning = False

    # 줌도, 따라가기도 산다
    win.canvas.zoom_to(1.4)
    assert abs(win.canvas.transform().m11() - 1.4) < 1e-6
    nid = next(iter(win.canvas.cards))
    win.canvas.center_on_node(nid)           # 죽지 않아야 한다

    # 그런데 상자는 끌 수 없다
    card = win.canvas.cards[nid]
    assert not card.flags() & QtWidgets.QGraphicsItem.ItemIsMovable
    win.set_running(False)
    assert card.flags() & QtWidgets.QGraphicsItem.ItemIsMovable


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


# ── 실행 버튼 ───────────────────────────────────────────────────────────
def test_the_run_button_calls_one_cli_command(win, monkeypatch):
    """버튼 하나가 CLI 명령 하나다. 편집기가 세 명령을 엮어 돌리기 시작하면
    그것이 CLI 에 없는 경로이고, 터미널에서 재현할 수 없는 결과가 나온다."""
    seen = {}

    def fake_spawn(cmd, phase):
        seen["cmd"], seen["phase"] = cmd, phase
        return {"ok": True}

    monkeypatch.setattr(win.editor, "_spawn", fake_spawn)
    monkeypatch.setattr(win.editor, "_ready_to_launch", lambda: {"ok": True})
    win._run_pipeline()

    assert seen["phase"] == "pipeline"
    assert seen["cmd"][2:5] == ["vlm_trainer.cli.main", "pipeline", win.editor.path]


def test_leftover_shards_stop_the_run_and_ask(win, monkeypatch):
    """말없이 이어받으면 예전 파라미터로 구운 샘플이 섞이고, 말없이 지우면 구운 시간이
    날아간다. 양쪽 다 나빠서 사람이 고르기 전에는 아무것도 띄우지 않는다."""
    from vlm_trainer.engine import pipeline as pipe

    win.editor.run_id = "r_left"
    monkeypatch.setattr(pipe, "leftover",
                        lambda rid, root="runs": pipe.Leftover(rid, "runs/x", 2, 100))
    monkeypatch.setattr(win.editor, "_ready_to_launch", lambda: {"ok": True})
    monkeypatch.setattr(win.editor, "_spawn",
                        lambda *a: pytest.fail("묻기 전에 실행을 띄웠다"))

    res = win.editor.pipeline_start()
    assert not res["ok"] and res["ask_resume"]["shards"] == 2


def test_answering_the_question_passes_the_choice_to_the_cli(win, monkeypatch):
    """고른 답이 명령줄에 그대로 실려야 한다. 창에만 남으면 하위 프로세스는 모른다."""
    from vlm_trainer.engine import pipeline as pipe

    win.editor.run_id = "r_left"
    monkeypatch.setattr(pipe, "leftover",
                        lambda rid, root="runs": pipe.Leftover(rid, "runs/x", 2, 100))
    monkeypatch.setattr(win.editor, "_ready_to_launch", lambda: {"ok": True})
    got = []
    monkeypatch.setattr(win.editor, "_spawn", lambda cmd, phase: got.append(cmd) or {"ok": True})

    win.editor.pipeline_start(resume=True)
    win.editor.pipeline_start(resume=False)
    assert "--resume" in got[0] and "--fresh" in got[1]


def test_the_stage_line_says_which_part_is_running(win):
    """숫자만 흐르면 굽는 중인지 학습 중인지 알 수 없는데, 둘은 걸리는 시간이
    자릿수로 다르다."""
    from vlm_trainer.core.compiler import compile_project
    from vlm_trainer.engine import pipeline as pipe

    p = pipe.plan(compile_project(win.editor.path), "r1")
    p.stages[0].state = pipe.DONE
    p.stages[1].state = pipe.RUNNING
    line = win.watcher._headline(
        {"kind": "pipeline", "run_id": "r1", "pipeline": pipe.as_dict(p), "total": 0})

    assert "준비 v" in line and "사전 준비 >" in line


# ── 실행 따라가기 ───────────────────────────────────────────────────────
def test_following_moves_only_when_the_running_box_changes(win):
    """폴링마다 화면을 옮기면 사람이 다른 곳을 보려 할 때마다 끌려 돌아온다."""
    moved = []
    win.canvas.centerOn = lambda item: moved.append(item)

    nid = next(iter(win.canvas.cards))
    card = win.canvas.cards[nid]
    card.setPos(0, 100_000)                  # 확실히 화면 밖으로

    win.watcher._follow(nid)
    win.watcher._follow(nid)                 # 같은 상자 — 다시 옮기지 않는다
    assert len(moved) == 1, f"같은 상자에 {len(moved)}번 따라갔다"


def _shown(win, w=1480, h=940):
    """실제로 창을 띄운다. `show()` 없이는 Qt 가 레이아웃을 안 돌려 캔버스가
    100x30 으로 남고, 무엇을 재든 거짓이 된다."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    win.resize(w, h)
    win.show()
    for _ in range(12):
        app.processEvents()
    return app


def test_following_leaves_a_box_alone_if_it_is_already_visible(win):
    """이미 보이는 것을 화면 가운데로 끌어오면 그래프가 이유 없이 출렁인다."""
    app = _shown(win)
    nid = next(iter(win.canvas.cards))
    win.canvas.centerOn(win.canvas.cards[nid])      # 화면 한가운데에 둔다
    for _ in range(4):
        app.processEvents()

    moved = []
    win.canvas.centerOn = lambda *a: moved.append(a)
    win.watcher._follow_last = ""
    win.watcher._follow(nid)
    win.hide()
    assert not moved, "가운데 있는 상자를 따라갔다"


def test_following_moves_when_the_box_is_only_half_on_screen(win):
    """**이것이 "따라가기가 하나도 동작을 안 한다" 의 정체였다.** 전에는 상자의
    중심점 하나만 보고 "보인다" 로 쳤다 — 가장자리에 반쯤 잘려 글자를 못 읽는
    상태인데도 가만히 있는다. 실측에서 18회 중 4회만 움직였다."""
    app = _shown(win)
    nid = next(iter(win.canvas.cards))
    card = win.canvas.cards[nid]

    # **상자보다 좁은 화면**을 만든다. 그러면 어디에 두든 전체가 들어갈 수 없고,
    # 중심만 보는 옛 검사는 "보인다" 로 지나간다 — 재현하려던 상황이 그것이다.
    # 상자가 화면보다 **확실히 커지도록** 필요한 배율을 재서 쓴다. 숫자를 손으로
    # 적으면 카드 크기나 화면 크기가 바뀌는 날 조용히 시험이 아니게 된다.
    vw = win.canvas.viewport().width()
    cw = card.sceneBoundingRect().width()
    win.canvas.zoom_to(min(4.0, (vw / cw) * 1.6))
    win.canvas.centerOn(card)
    for _ in range(4):
        app.processEvents()

    vp = win.canvas.viewport().rect()
    r = win.canvas.mapFromScene(card.sceneBoundingRect()).boundingRect()
    assert vp.contains(win.canvas.mapFromScene(card.sceneBoundingRect().center())),         "중심이 화면 밖이다 — 옛 검사도 움직였을 상황이라 시험이 안 된다"
    assert not vp.contains(r), "상자가 다 들어간다 — 더 확대해야 한다"

    moved = []
    win.canvas.centerOn = lambda *a: moved.append(a)
    win.watcher._follow_last = ""
    win.watcher._follow(nid)
    win.hide()
    assert moved, "반쯤 잘린 상자를 따라가지 않았다"


def test_the_follow_toggle_is_obeyed(win):
    moved = []
    win.canvas.centerOn = lambda item: moved.append(item)
    nid = next(iter(win.canvas.cards))
    win.canvas.cards[nid].setPos(0, 100_000)

    win.act_follow.setChecked(False)
    win.watcher._follow(nid)
    assert not moved, "따라가기를 껐는데 옮겼다"


def test_a_box_that_is_not_on_the_canvas_does_not_crash_the_follow(win):
    """접힌 Procedure 안쪽 id 가 올라오면 캔버스에 그 상자가 없다. 죽지 않아야 한다."""
    win.watcher._follow("없는노드")


# ── 화면이 흔들리지 않는다 ──────────────────────────────────────────────
def test_adding_a_node_does_not_move_the_view(win):
    """줌을 맞춰 놓고 노드를 하나 추가하면 **화면 비율이 깨져 보이던** 것이 이것이다.
    `load()` 가 scene 을 통째로 비우고 다시 채우는데, 그때 Qt 가 경계를 다시 셈하면서
    배율과 위치가 날아갔다."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    win.resize(1480, 940)
    for _ in range(5):
        app.processEvents()

    win.canvas.zoom_to(2.2)
    win.canvas.centerOn(300, 400)
    z0 = win.canvas.transform().m11()
    c0 = win.canvas.mapToScene(win.canvas.viewport().rect().center())

    win._add_node("source.field@1.0.0", 100, 100)
    for _ in range(6):
        app.processEvents()

    z1 = win.canvas.transform().m11()
    c1 = win.canvas.mapToScene(win.canvas.viewport().rect().center())
    assert abs(z1 - z0) < 1e-6, f"배율이 {z0:.3f} -> {z1:.3f} 로 바뀌었다"
    # 스크롤 위치는 정수 픽셀로 떨어지므로 몇 px 는 어쩔 수 없다. 화면이 튀는 것은
    # 수백 px 단위였다 — 그 차이를 잡는다.
    assert abs(c1.x() - c0.x()) < 12 and abs(c1.y() - c0.y()) < 12, \
        f"보던 자리가 ({c0.x():.0f},{c0.y():.0f}) -> ({c1.x():.0f},{c1.y():.0f}) 로 날아갔다"


def test_the_scene_has_room_so_the_view_does_not_jitter(win):
    """경계를 상자에 딱 맞추면 노드를 추가할 때마다 스크롤바가 생겼다 사라지며
    화면이 출렁인다."""
    r = win.canvas._scene.sceneRect()
    b = win.canvas._scene.itemsBoundingRect()
    assert r.contains(b), "경계가 상자를 다 담지 못한다"
    assert r.width() > b.width() + 20, "여유가 없다"


# ── 처음 보는 사람이 읽을 수 있는가 ─────────────────────────────────────
def test_a_port_tooltip_is_written_for_a_person_not_for_the_gate(win):
    """포트에 호버하면 `Image{u8, HWC, 0-255, none, RGB, ANY, [dyn,dyn,3]}` 가
    그대로 떴다. 그 표기는 게이트가 쓰는 말이지 처음 보는 사람이 읽을 말이 아니다."""
    tip = win.canvas.chips["out:n_img:image"].toolTip()
    assert "사진" in tip, "무엇인지 사람 말로 말하지 않는다"
    assert "예시" in tip, "어떻게 생긴 값인지 보여 주지 않는다"
    assert "Image" in tip, "정확한 타입 이름도 남아 있어야 한다"
    assert "{" not in tip.split("Image")[-1], "타입 표기가 그대로 들어갔다"


def test_an_input_port_says_what_is_plugged_into_it(win):
    """처음 보는 사람이 제일 먼저 궁금해하는 것이다."""
    tip = win.canvas.chips["in:n_stats:series"].toolTip()
    assert "지금 꽂힌 것" in tip
    # 앞 상자를 id(`n_win`)가 아니라 한국어 이름으로 부른다
    assert "n_" not in tip.split("지금 꽂힌 것")[1].split("<br>")[1]


def test_an_empty_required_port_says_so_and_an_output_does_not(win):
    """출력은 꽂는 자리가 아니다. 거기에 "아직 안 꽂혔다" 가 뜨면 혼란만 준다."""
    out = win.canvas.chips["out:n_img:image"].toolTip()
    assert "꽂혀 있지 않" not in out

    from vlm_trainer.ui import layout as L
    from vlm_trainer.core.types import text

    empty = L.port_help(text(), "질문", wired="", required=True, is_input=True)
    assert "꽂혀 있지 않" in empty and "끌어다 놓" in empty


def test_an_optional_port_says_it_is_optional():
    from vlm_trainer.core.types import text
    from vlm_trainer.ui import layout as L

    tip = L.port_help(text().as_optional(), "정답", required=False)
    assert "없어도 됩니다" in tip


def test_the_node_label_is_used_not_the_id():
    from vlm_trainer.core import registry
    from vlm_trainer.core.compiler import compile_project
    from vlm_trainer.ui import layout as L

    registry.load_builtin_nodes()
    cg = compile_project(os.path.join(ROOT, "solutions", "vlm_open",
                                      "projects", "01_open", "project.yaml"))
    assert L.node_label(cg, "n_train") == "모델 학습"
    assert L.node_label(cg, "없는노드") == "없는노드", "모르는 id 에 죽지 않아야 한다"


# ── 실행 따라가기 ───────────────────────────────────────────────────────
def test_the_follow_toggle_shows_whether_it_is_on(win):
    """QSS 에 `QToolButton:checked` 가 없어서, 눌러도 화면에 아무 차이가 없었다 —
    켜졌는지 꺼졌는지 알 길이 없다."""
    assert win.act_follow.isCheckable() and win.act_follow.isChecked()
    assert "QToolButton:checked" in win.styleSheet()


def test_slow_mode_is_off_by_default_and_asks_the_engine_to_wait(win):
    """빠른 상자는 눈 깜짝할 새에 지나가 어디를 도는지 안 보인다. 1초씩 쉬면
    따라가기가 실제로 움직이는 것이 보인다."""
    assert win.act_slow.isCheckable() and not win.act_slow.isChecked()
    assert win._step_delay() == 0.0

    win.act_slow.setChecked(True)
    assert win._step_delay() == 1.0


def test_the_delay_reaches_the_command_line(win, monkeypatch):
    """앱이 제 실행 경로를 갖지 않는다 — `vlmt run --step-delay` 를 띄운다."""
    got = []
    monkeypatch.setattr(win.editor, "_ready_to_launch", lambda: {"ok": True})
    monkeypatch.setattr(win.editor, "_spawn", lambda cmd, phase: got.append(cmd) or {"ok": True})

    win.editor.run_start(4, False, 1.0)
    assert "--step-delay" in got[0] and "1.0" in got[0]

    got.clear()
    win.editor.run_start(4, False, 0.0)
    assert "--step-delay" not in got[0], "꺼져 있으면 넘기지 않는다"


def test_the_sample_count_says_which_button_uses_it(win):
    """`샘플 8` 만 적혀 있어 무슨 숫자인지 알 수 없었다."""
    assert "건" in win.spin_limit.suffix()
    assert "미리보기" in win.spin_limit.toolTip()


# ── 쓰는 사람이 아는 말로 ───────────────────────────────────────────────
#
# 사용자는 데이터 / 질문 양식 / 정답 양식 / 모델 넷을 넣고 학습하면 되는 사람이다.
# `물질화 경계` 나 `shard` 는 우리가 설계하며 쓴 말이지 그 사람이 알아야 할 말이 아니다.

def test_no_screen_text_uses_the_old_words(win):
    """`vlm_trainer/ui/words.py` 의 표를 기준으로 화면 전체를 훑는다.
    말이 한곳에 모여 있지 않으면 다음에 또 어긋난다."""
    from PySide6 import QtWidgets

    from vlm_trainer.ui import words as W

    win._show_inspector("n_sample")
    found = []
    for widget in win.findChildren(QtWidgets.QWidget):
        text = (getattr(widget, "text", lambda: "")() or "") + (widget.toolTip() or "")
        for old in W.RENAMED:
            if old in text:
                found.append(f"{old} → {text[:40]}")
    assert not found, found


def test_the_words_are_defined_in_one_place():
    """흩어 두면 같은 것이 두 이름으로 불린다."""
    from vlm_trainer.ui import words as W

    assert W.BOUNDARY == "사전 준비" and W.BAKE == "사전 준비 실행"
    assert W.RENAMED["물질화 경계"] == W.BOUNDARY
    assert W.BOUNDARY_WHY and "GPU" in W.BOUNDARY_WHY, "왜 필요한지도 적어야 한다"


def test_the_boundary_is_still_a_real_setting(win):
    """말만 바꾼 것이지 개념을 없앤 것이 아니다. 여전히 게이트가 요구하는 값이다."""
    from PySide6 import QtWidgets

    win._show_inspector("n_sample")
    boxes = [c for c in win.inspector.findChildren(QtWidgets.QCheckBox) if c.text()]
    assert boxes and boxes[0].text() == "사전 준비"
    assert boxes[0].isChecked(), "이 그래프는 여기를 경계로 쓴다"


# ── 라이브러리 ──────────────────────────────────────────────────────────
def test_the_library_shows_korean_names_not_type_ids(win):
    """`[?] source.field` 가 뜨던 자리다. `?` 는 `kind` 를 소문자 키로 찾아서 나온
    것이고, 타입 이름은 처음 보는 사람에게 아무 뜻도 없다."""
    from PySide6 import QtCore, QtWidgets

    tree = win.library.findChild(QtWidgets.QTreeWidget)
    names, refs = [], []
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        for j in range(top.childCount()):
            it = top.child(j)
            names.append(it.text(0))
            refs.append(it.data(0, QtCore.Qt.UserRole))

    assert names, "라이브러리가 비었다"
    assert not any("[?]" in n for n in names), "kind 표식이 물음표로 떨어진다"
    assert not any("." in n and "@" not in n and n.islower() for n in names), \
        "타입 이름이 그대로 뜬다"
    assert "단일 열 업로드" in names, "source.field 가 한국어 이름으로 안 보인다"
    assert all(r and "@" in r for r in refs), "끌어다 놓을 때 쓸 타입은 그대로 있어야 한다"


def test_hovering_a_library_item_says_what_goes_in(win):
    """이름만으로는 무엇을 꽂아야 하는지 알 수 없다 — `입력: image, regions` 가
    전부였다."""
    from PySide6 import QtCore, QtWidgets

    tree = win.library.findChild(QtWidgets.QTreeWidget)
    tip = ""
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        for j in range(top.childCount()):
            it = top.child(j)
            if (it.data(0, QtCore.Qt.UserRole) or "").startswith("image.crop_by_regions"):
                tip = it.toolTip(0)
    assert tip, "그 노드를 못 찾았다"
    assert "받는 것" in tip and "내보내는 것" in tip
    assert "사진" in tip, "포트 종류를 사람 말로 말하지 않는다"
    assert "image.crop_by_regions" in tip, "정확한 타입도 남아 있어야 한다"


def test_the_library_payload_carries_labels():
    """패널이 타입 이름을 쓸 수밖에 없었던 이유가 이것이다 — `library()` 가
    `label` 을 안 줬다."""
    from vlm_trainer.core import registry
    from vlm_trainer.ui.api import Editor

    registry.load_builtin_nodes()
    ed = Editor.open(os.path.join(ROOT, "solutions", "vlm_open",
                                  "projects", "01_open", "project.yaml"))
    by_type = {n["type"]: n for n in ed.library()}
    one = by_type["infer.vlm@1.0.0"]
    assert one["label"] == "모델 추론" and one["hint"]
    assert one["port_help"]["in:model"] and one["port_help"]["out:answer"]
    assert all(n.get("label") for n in by_type.values()), "이름 없는 노드가 있다"


# ── 상자가 겹치지 않는다 ────────────────────────────────────────────────
#
# 겹친 둘은 **둘 다** 못 읽는다. `fixed` 좌표(사람이 옮긴 자리, 새로 놓은 자리)가
# 자동 배치와 충돌을 전혀 안 따져서, 새 상자가 기존 상자 위에 그대로 얹혔다.

def _overlaps(canvas):
    items = list(canvas.cards.items())
    return [f"{a}~{b}" for i, (a, ca) in enumerate(items)
            for b, cb in items[i + 1:]
            if ca.sceneBoundingRect().intersects(cb.sceneBoundingRect())]


def test_no_two_boxes_overlap_when_the_graph_opens(win):
    assert not _overlaps(win.canvas)


def test_two_boxes_pinned_to_the_same_spot_are_pulled_apart(win):
    """사람이 상자를 끌어다 다른 상자 위에 놓을 수 있다. 그 자리는 `layout.yaml` 에
    박히고, 자동 배치는 `fixed` 좌표를 그대로 쓴다 — 그래서 겹친 채로 그려졌다.

    **둘 다 고정해야 재현된다.** 한쪽만 고정하면 다른 쪽이 자동 배치로 비켜서서
    겹침이 생기지 않는다.
    """
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    a, b = list(win.canvas.cards)[:2]
    win.editor.move_node(a, 200, 200)
    win.editor.move_node(b, 200, 200)        # 같은 자리에 겹쳐 놓는다
    win.reload_graph()
    for _ in range(6):
        app.processEvents()

    assert not _overlaps(win.canvas), "겹친 채로 그려진다 — 둘 다 못 읽는다"


def test_a_new_box_lands_somewhere_free(win):
    """끌어다 놓은 자리에 **그대로** 두면 기존 상자 위에 얹힌다."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    some = next(iter(win.canvas.cards))
    at = win.canvas.cards[some].scenePos()
    before = len(win.canvas.cards)

    win._add_node("source.field@1.0.0", int(at.x()), int(at.y()))
    for _ in range(6):
        app.processEvents()

    assert len(win.canvas.cards) == before + 1, "노드가 안 늘었다"
    assert not _overlaps(win.canvas)


def test_the_layout_pushes_a_clashing_box_down_not_sideways():
    """좌우로 밀면 레인 안의 순서(상류의 무게중심 순)가 흐트러진다.
    위로 밀면 윗 레인을 침범한다."""
    from vlm_trainer.ui import layout as L

    a = L.Placed(id="a", x=100, y=100, lane=0, w=200, h=80)
    b = L.Placed(id="b", x=100, y=100, lane=0, w=200, h=80)   # 완전히 겹친다
    placed = {"a": a, "b": b}
    L._unoverlap(placed)

    moved = b if b.y != 100 else a
    stayed = a if moved is b else b
    assert moved.x == stayed.x, "좌우로 밀었다"
    assert moved.y > stayed.y, "아래로 밀지 않았다"
    assert moved.y >= stayed.y + stayed.h, "아직 겹친다"


def test_boxes_far_apart_are_left_alone():
    """겹치지 않는 것까지 건드리면 사람이 옮겨 둔 자리가 날아간다."""
    from vlm_trainer.ui import layout as L

    a = L.Placed(id="a", x=0, y=0, lane=0, w=200, h=80)
    b = L.Placed(id="b", x=600, y=0, lane=0, w=200, h=80)
    L._unoverlap({"a": a, "b": b})
    assert (a.x, a.y, b.x, b.y) == (0, 0, 600, 0)


def test_narrowing_the_window_shrinks_the_zoom(win):
    """창을 줄였는데 배율이 그대로면 상자가 화면 밖으로 잘려 나간다."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    win.show()
    win.resize(1480, 940)
    for _ in range(10):
        app.processEvents()
    win.canvas.fit()
    for _ in range(4):
        app.processEvents()
    wide = win.canvas.transform().m11()

    win.resize(700, 940)
    for _ in range(10):
        app.processEvents()
    narrow = win.canvas.transform().m11()
    win.hide()

    assert narrow < wide, f"창을 줄였는데 배율이 {wide:.3f} -> {narrow:.3f} 로 안 줄었다"
