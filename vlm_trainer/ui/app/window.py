"""편집기 창 — QMainWindow.

로직은 전부 `ui/api.py`에 있다. 이 파일은 그것을 부르고 결과를 그리기만 한다.
웹판의 `server.py`가 하던 자리이고, 하는 일도 같다 — 껍데기다.

**거부는 조용히 넘기지 않는다.** `api`가 게이트를 이유로 편집을 물리면 화면을
원래대로 되돌리고 그 이유를 그대로 보여 준다.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from PySide6 import QtCore, QtGui, QtWidgets

from .. import tokens as T
from ..api import Editor, compat_matrix, occupied_inputs
from .canvas import GraphCanvas
from .history import HistoryDock, RecipeDock
from .inspector import Inspector
from .library import LibraryDock
from .progress import RunDock, RunWatcher

FONT = "Malgun Gothic"

# 창 전체 색. tokens.py 의 값을 그대로 쓴다 — 웹판과 같은 화면이어야 한다.
QSS = f"""
QMainWindow, QWidget {{ background: {T.SURFACE['panel']}; color: #D8DCDF;
                        font-family: "{FONT}"; font-size: 12px; }}
QMenuBar {{ background: {T.SURFACE['chrome']}; color: #C6CCD1; }}
QMenuBar::item:selected {{ background: {T.NODE['bg_selected']}; }}
QMenu {{ background: {T.SURFACE['panel']}; border: 1px solid {T.SURFACE['line']}; }}
QMenu::item {{ padding: 4px 22px; }}
QMenu::item:selected {{ background: {T.NODE['bg_selected']}; }}
QToolBar {{ background: {T.SURFACE['toolbar']}; border: 0;
            border-bottom: 1px solid {T.SURFACE['line']}; spacing: 6px; padding: 5px 8px; }}
QToolButton {{ background: {T.SURFACE['panel']}; color: #D8DCDF;
               border: 1px solid {T.NODE['border']}; border-radius: 2px; padding: 3px 10px; }}
QToolButton:hover {{ background: {T.NODE['bg_selected']}; }}
QToolButton:disabled {{ color: #5F6468; border-color: #3A3A3A; }}
QStatusBar {{ background: {T.SURFACE['chrome']}; color: #8A9196; }}
QStatusBar::item {{ border: 0; }}
QDockWidget {{ titlebar-close-icon: none; color: #C6CCD1; font-size: 11px; }}
QDockWidget::title {{ background: {T.SURFACE['toolbar']}; padding: 5px 8px;
                      border-bottom: 1px solid {T.SURFACE['line']}; }}
QTreeWidget {{ background: {T.SURFACE['panel']}; border: 0; outline: 0; font-size: 11px; }}
QTreeWidget::item {{ padding: 3px 2px; }}
QTreeWidget::item:selected {{ background: {T.NODE['bg_selected']}; }}
QLineEdit {{ background: {T.SURFACE['canvas']}; border: 1px solid {T.SURFACE['line']};
             border-radius: 2px; padding: 4px 6px; color: #D8DCDF; }}
QLabel#runline {{ color: {T.STATE['running']};
                  font-family: Consolas, ui-monospace, monospace; font-size: 11px; }}
QToolTip {{ background: {T.SURFACE['toolbar']}; color: #D8DCDF;
            border: 1px solid {T.NODE['border']}; padding: 4px; }}
"""


class EditorWindow(QtWidgets.QMainWindow):
    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.editor = editor
        self.resize(1480, 940)
        self.setStyleSheet(QSS)

        self.canvas = GraphCanvas(self)
        self.setCentralWidget(self.canvas)

        self.library = LibraryDock(editor.library(), self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.library)

        self.inspector = Inspector(self)
        self.inspector.bind(editor)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.inspector)

        self.history = HistoryDock(self)
        self.recipe = RecipeDock(self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.history)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.recipe)
        # 오른쪽은 탭으로 겹친다 — 셋을 세로로 쌓으면 어느 것도 제대로 안 보인다
        self.run_dock = RunDock(self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.run_dock)
        self.tabifyDockWidget(self.inspector, self.history)
        self.tabifyDockWidget(self.history, self.recipe)
        self.tabifyDockWidget(self.recipe, self.run_dock)
        self.inspector.raise_()
        self.resizeDocks([self.library, self.inspector], [250, 330], QtCore.Qt.Horizontal)

        self._connect_canvas()
        self._connect_panels()
        self._build_menu()
        self._build_toolbar()
        self._build_status()
        self.run_dock.bind_preview_reader(self.editor.preview_file)
        self.watcher = RunWatcher(self)
        self.run_dock.hide()        # 실행 전에는 자리를 차지하지 않는다
        self.reload_graph()
        QtCore.QTimer.singleShot(0, self.canvas.fit)

    # ── 배선 ────────────────────────────────────────────────────────────
    @staticmethod
    def _later(fn: Any) -> None:
        """다음 이벤트 차례로 미룬다.

        캔버스에서 온 요청은 **마우스 이벤트나 메뉴의 이벤트 루프 안에서** 도착한다.
        그 자리에서 그래프를 다시 그리면 `scene.clear()` 가 지금 처리 중인 아이템을
        지워 버리고, Qt 는 이미 없는 것을 마저 만지다 죽는다(세그폴트).
        이벤트가 풀린 뒤에 바꾼다.
        """
        QtCore.QTimer.singleShot(0, fn)

    def _connect_canvas(self) -> None:
        c = self.canvas
        c.selected.connect(self._on_pick)          # 읽기만 한다 — 미룰 필요 없다
        c.connectRequested.connect(
            lambda a, b: self._later(
                lambda: self._mutate(lambda: self.editor.connect(a, b), f"배선 {a} → {b}")))
        c.disconnectRequested.connect(
            lambda d: self._later(
                lambda: self._mutate(lambda: self.editor.disconnect(d), f"배선 끊기 {d}")))
        c.removeRequested.connect(lambda n: self._later(lambda: self._remove_node(n)))
        c.expandRequested.connect(
            lambda n: self._later(
                lambda: self._mutate(lambda: self.editor.toggle_expand(n), "펼치기")))
        c.boundaryRequested.connect(
            lambda n, on: self._later(
                lambda: self._mutate(lambda: self.editor.set_boundary(n, on), "물질화 경계")))
        c.addRequested.connect(lambda t, x, y: self._later(lambda: self._add_node(t, x, y)))
        c.movedNode.connect(self._move_node)       # 다시 그리지 않는다 — 미룰 필요 없다

    def _connect_panels(self) -> None:
        i = self.inspector
        i.paramChanged.connect(self._set_param)
        i.boundaryToggled.connect(
            lambda n, on: self._later(
                lambda: self._mutate(lambda: self.editor.set_boundary(n, on), "물질화 경계")))
        i.expandToggled.connect(
            lambda n: self._later(
                lambda: self._mutate(lambda: self.editor.toggle_expand(n), "펼치기")))
        i.removeRequested.connect(lambda n: self._later(lambda: self._remove_node(n)))
        i.focusRequested.connect(self.canvas.center_on_node)

        self.history.rewindRequested.connect(
            lambda idx: self._later(
                lambda: self._mutate(lambda: self.editor.rewind(idx), "되감기")))
        self.recipe.selectRecipe.connect(
            lambda rid: self._later(
                lambda: self._mutate(lambda: self.editor.recipe_select(rid), "레시피 적용")))
        self.recipe.dropPath.connect(
            lambda path: self._later(
                lambda: self._mutate(lambda: self.editor.recipe_drop_path(path), "축 빼기")))
        self.recipe.storeRequested.connect(
            lambda rid, name: self._later(
                lambda: self._mutate(
                    lambda: self.editor.recipe_store(rid, name, ""), "레시피에 담기")))

    def _set_param(self, nid: str, param: str, value: Any) -> None:
        """값이 바뀌는 순간 보낸다. 거부되면 패널을 다시 그려 원래 값으로 되돌린다 —
        화면에만 남은 값은 다음에 여는 사람을 속인다."""
        def apply() -> None:
            res = self.editor.set_param(nid, param, value)
            if not res.get("ok"):
                self._refuse(f"{nid}.{param} = {value!r}", res)
            self.reload_graph()

        # 값을 보낸 입력칸은 패널을 다시 그리는 순간 지워진다. 그 칸의 시그널
        # 처리가 끝난 뒤에 바꾼다.
        self._later(apply)

    # ── 화면 구성 ───────────────────────────────────────────────────────
    def _title(self) -> str:
        cg = self.editor.compiled
        name = (cg.name or cg.id) if cg is not None else os.path.basename(self.editor.path)
        return f"{name} — VLM Trainer" + ("  *저장 안 됨" if self.editor.dirty else "")

    def _build_menu(self) -> None:
        m = self.menuBar()
        f = m.addMenu("파일(&F)")
        f.addAction("저장", QtGui.QKeySequence.Save, self.on_save)
        f.addSeparator()
        f.addAction("닫기", QtGui.QKeySequence.Quit, self.close)

        e = m.addMenu("편집(&E)")
        e.addAction("실행 취소", QtGui.QKeySequence.Undo,
                    lambda: self._mutate(self.editor.undo, "실행 취소"))
        e.addAction("다시 실행", QtGui.QKeySequence.Redo,
                    lambda: self._mutate(self.editor.redo, "다시 실행"))
        e.addSeparator()
        self.act_del = e.addAction("선택한 노드 삭제", QtGui.QKeySequence.Delete,
                                   lambda: self._remove_node(self._picked))
        self.act_del.setEnabled(False)

        v = m.addMenu("보기(&V)")
        v.addAction("전체 맞춤", QtGui.QKeySequence("Ctrl+0"), self.canvas.fit)
        v.addAction("확대", QtGui.QKeySequence.ZoomIn,
                    lambda: self.canvas.zoom_to(self.canvas.transform().m11() * 1.25))
        v.addAction("축소", QtGui.QKeySequence.ZoomOut,
                    lambda: self.canvas.zoom_to(self.canvas.transform().m11() / 1.25))
        v.addSeparator()
        v.addAction("자동 정렬로 되돌리기",
                    lambda: self._mutate(self.editor.reset_layout, "자동 정렬"))
        v.addAction(self.library.toggleViewAction())

        m.addMenu("도움말(&H)").addAction("조작법", self._show_help)

    def _build_toolbar(self) -> None:
        tb = QtWidgets.QToolBar("실행")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_run = tb.addAction("Run", lambda: self._launch(
            lambda: self.editor.run_start(self.spin_limit.value(), self.chk_debug.isChecked()),
            "Run"))
        self.act_mat = tb.addAction("Materialize", lambda: self._launch(
            self.editor.materialize_start, "Materialize"))
        self.act_train = tb.addAction("Train", lambda: self._launch(
            self.editor.train_start, "Train"))
        self.act_stop = tb.addAction("Stop", self._stop)
        self.act_stop.setEnabled(False)

        tb.addSeparator()
        self.act_follow = tb.addAction("실행 따라가기")
        self.act_follow.setCheckable(True)
        self.act_follow.setChecked(True)
        self.act_follow.setToolTip("실행 중인 상자가 화면 밖이면 그쪽으로 옮긴다")

        self.chk_debug = QtWidgets.QCheckBox("Debug Output")
        self.chk_debug.setToolTip("꺼져 있으면 미리보기를 생성조차 하지 않는다")
        tb.addWidget(self.chk_debug)

        tb.addWidget(QtWidgets.QLabel("  샘플 "))
        self.spin_limit = QtWidgets.QSpinBox()
        self.spin_limit.setRange(1, 99999)
        self.spin_limit.setValue(8)
        self.spin_limit.setFixedWidth(72)
        tb.addWidget(self.spin_limit)

        tb.addSeparator()
        tb.addAction("전체 맞춤", self.canvas.fit)
        tb.addAction("저장", self.on_save)

    def _build_status(self) -> None:
        sb = self.statusBar()
        self.lbl_graph = QtWidgets.QLabel()
        self.lbl_run = QtWidgets.QLabel()
        self.lbl_run.setObjectName("runline")
        sb.addWidget(self.lbl_graph)
        sb.addPermanentWidget(self.lbl_run)
        self.lbl_refuse = QtWidgets.QLabel()
        self.lbl_refuse.setStyleSheet(f"color:{T.STATE['failed']}; font-size:11px;")
        sb.addWidget(self.lbl_refuse, 1)
        self._refuse_timer = QtCore.QTimer(self)
        self._refuse_timer.setSingleShot(True)
        self._refuse_timer.timeout.connect(lambda: self.lbl_refuse.setText(""))
        self._picked = ""
        self._state: Dict[str, Any] = {}

    # ── 그래프 ──────────────────────────────────────────────────────────
    def reload_graph(self) -> None:
        cg = self.editor.compiled
        self.setWindowTitle(self._title())
        if cg is None:
            self.lbl_graph.setText("컴파일되지 않았다: "
                                   + (self.editor.error or "").splitlines()[0])
            return
        fixed = {k: (int(v[0]), int(v[1])) for k, v in (self.editor.layout or {}).items()
                 if isinstance(v, (list, tuple)) and len(v) == 2}
        self.canvas.load(
            cg, self.editor.expanded, fixed or None,
            compat=compat_matrix(cg),
            occupied=occupied_inputs(cg),
            boundary=self.editor.graph.materialize.boundary,
        )
        warn = "" if self.editor.valid else "   ⚠ 저장·실행 불가"
        self.lbl_graph.setText(
            f"상자 {len(self.canvas.cards)} · 노드 {len(cg.nodes)} · 배선 {len(cg.edges)} · "
            f"레인 {max(cg.lanes.values()) + 1}단   {cg.spec_hash}{warn}")

        # 패널은 한 번 만든 state 를 나눠 쓴다. 도크마다 다시 부르면 샘플 공간을
        # 세 번 읽게 된다 — 인덱스가 큰 프로젝트에서 그대로 체감된다.
        self._state = self.editor.state()
        self.history.update_view(self._state.get("history") or [])
        self.recipe.update_view(self._state.get("recipe") or {})
        self._show_inspector(self._picked)

    def _mutate(self, fn: Any, label: str = "편집") -> bool:
        """편집을 시도한다. `api`가 물리면 화면을 되돌리고 이유를 보여 준다."""
        res = fn()
        if not res.get("ok"):
            self._refuse(label, res)
            self.reload_graph()      # 화면에 남은 흔적을 스펙 기준으로 되돌린다
            return False
        self.reload_graph()
        return True

    def _refuse(self, label: str, res: Dict[str, Any]) -> None:
        """거부를 **창을 막지 않고** 알린다.

        모달을 띄우면 배선을 끄는 중에 대화상자가 앞을 가로막는다. 웹판도 토스트였다.
        전문은 실행 도크의 콘솔 칸에 남겨 두어, 읽고 싶을 때 읽는다.
        """
        reason = res.get("reason", "") or "이유 없음"
        self.lbl_refuse.setText(f"거부: {label} — {reason}")
        self.lbl_refuse.setToolTip(res.get("detail") or reason)
        self._refuse_timer.start(9000)
        detail = res.get("detail") or ""
        if detail and detail != reason:
            self.run_dock.console.setPlainText(detail)
            self.run_dock.console.setVisible(True)

    # ── 편집 ────────────────────────────────────────────────────────────
    def _add_node(self, node_type: str, x: int, y: int) -> None:
        res = self.editor.add_node(node_type)
        if not res.get("ok"):
            self._refuse(f"노드 추가 {node_type}", res)
            return
        nid = res.get("id", "")
        if nid:
            self.editor.move_node(nid, x, y)     # 놓은 자리에 둔다
        self.reload_graph()
        if nid in self.canvas.cards:
            self.canvas.cards[nid].setSelected(True)
        self.statusBar().showMessage(f"추가: {nid}  ({node_type})", 4000)

    def _remove_node(self, nid: str) -> None:
        if not nid:
            return
        ok = QtWidgets.QMessageBox.question(
            self, "노드 삭제",
            f"'{nid}' 를 지운다. 이 노드에 닿은 배선도 함께 사라진다.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if ok == QtWidgets.QMessageBox.Yes:
            self._mutate(lambda: self.editor.remove_node(nid), f"노드 삭제 {nid}")

    def _move_node(self, nid: str, x: int, y: int) -> None:
        """자리는 스펙이 아니다 — layout.yaml 로 가고 spec_hash 를 건드리지 않는다.
        그래서 여기서는 그래프를 다시 그리지 않는다. 다시 그리면 방금 옮긴 상자가
        자동 배치로 튕겨 돌아간다."""
        res = self.editor.move_node(nid, x, y)
        if not res.get("ok"):
            self._refuse(f"상자 이동 {nid}", res)
            self.reload_graph()

    def on_save(self) -> None:
        res = self.editor.save()
        if res.get("ok"):
            self.statusBar().showMessage(f"저장했다: {res.get('path', '')}", 5000)
            self.setWindowTitle(self._title())
        else:
            self._refuse("저장", res)

    # ── 실행 ────────────────────────────────────────────────────────────
    def _launch(self, fn: Any, label: str) -> None:
        """버튼 하나가 CLI 명령 하나다. 편집기가 여러 명령을 엮어 돌리기 시작하면
        그것이 CLI 에 없는 경로다."""
        res = fn()
        if not res.get("ok"):
            self._refuse(label, res)
            return
        self.run_dock.show()
        self.run_dock.raise_()
        self.watcher.start()
        self.statusBar().showMessage(f"{label} 시작", 3000)

    def _stop(self) -> None:
        res = self.editor.run_stop()
        if not res.get("ok"):
            self.statusBar().showMessage(res.get("reason", ""), 4000)

    def set_running(self, on: bool) -> None:
        """돌고 있는 동안에는 그래프를 못 고치게 한다. 실행은 **디스크의 스펙**을
        도는데 편집은 메모리를 바꾼다 — 둘이 갈라지면 화면과 다른 것이 돌아간다."""
        for a in (self.act_run, self.act_mat, self.act_train):
            a.setEnabled(not on)
        self.act_stop.setEnabled(on)
        self.library.setEnabled(not on)
        self.inspector.setEnabled(not on)
        self.canvas.setInteractive(not on)

    # ── 선택 ────────────────────────────────────────────────────────────
    def _on_pick(self, nid: str) -> None:
        self._picked = nid
        self.act_del.setEnabled(bool(nid))
        self._show_inspector(nid)
        if not nid:
            self.lbl_run.setText("")
            return
        s = self.canvas.cards[nid].shown
        self.lbl_run.setText(f"{nid}   {s.ref}")

    def _show_inspector(self, nid: str) -> None:
        card = self.canvas.cards.get(nid)
        if card is None or self.editor.compiled is None:
            self.inspector.show_empty()
            return
        self.inspector.show_node(nid, card.shown, self.editor.compiled,
                                 getattr(self, "_state", {}) or {})

    def _show_help(self) -> None:
        QtWidgets.QMessageBox.information(
            self, "조작법",
            "화면\n"
            "  휠 — 확대·축소 (마우스 아래 지점 고정)\n"
            "  가운데 버튼 드래그, 또는 Ctrl + 왼쪽 드래그 — 화면 밀기\n"
            "  Ctrl+0 — 전체 맞춤\n\n"
            "편집\n"
            "  아래쪽 색칠된 칩을 끌어 다른 상자의 위쪽 칩에 놓으면 이어집니다.\n"
            "  끄는 동안 꽂을 수 없는 자리는 어두워집니다.\n"
            "  배선이나 위쪽 칩을 우클릭 — 끊기\n"
            "  왼쪽 라이브러리에서 끌어다 놓기 — 노드 추가\n"
            "  상자를 끌면 이동합니다. 자리는 layout.yaml 로 가고 spec_hash 는 그대로입니다.\n"
            "  상자 우클릭 — 펼치기/접기 · 물질화 경계 · 삭제\n\n"
            "실행(Run·Materialize·Train)은 5단계에서 붙습니다.")


def launch(project_path: str, extra_modules: tuple = ()) -> int:
    """창을 띄운다. `vlmt app`이 부르는 자리다."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("VLM Trainer")
    editor = Editor.open(project_path)
    editor.extra_modules = tuple(extra_modules)
    win = EditorWindow(editor)
    win.show()
    return app.exec()
