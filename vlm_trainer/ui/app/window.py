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
from .. import words as W
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
/* 켜진 토글. 이것이 없어서 `실행 따라가기` 가 켜졌는지 꺼졌는지 화면에 안 보였습니다. */
QToolButton:checked {{ background: {T.STATE['running']}; color: #0F1113;
                       border-color: {T.STATE['running']}; font-weight: 600; }}
QToolButton:checked:hover {{ background: {T.STATE['running']}; }}
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
                lambda: self._mutate(lambda: self.editor.set_boundary(n, on), W.BOUNDARY)))
        c.addRequested.connect(lambda t, x, y: self._later(lambda: self._add_node(t, x, y)))
        c.movedNode.connect(self._move_node)       # 다시 그리지 않는다 — 미룰 필요 없다

    def _connect_panels(self) -> None:
        i = self.inspector
        i.paramChanged.connect(self._set_param)
        i.boundaryToggled.connect(
            lambda n, on: self._later(
                lambda: self._mutate(lambda: self.editor.set_boundary(n, on), W.BOUNDARY)))
        i.expandToggled.connect(
            lambda n: self._later(
                lambda: self._mutate(lambda: self.editor.toggle_expand(n), "펼치기")))
        i.removeRequested.connect(lambda n: self._later(lambda: self._remove_node(n)))
        i.focusRequested.connect(self.canvas.center_on_node)
        i.exportRequested.connect(lambda n: self._later(self._export_model))

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

        r = m.addMenu("실행(&R)")
        r.addAction("실행", QtGui.QKeySequence("F5"), self._run_pipeline)
        r.addAction("미리보기", QtGui.QKeySequence("F6"), lambda: self._launch(
            lambda: self.editor.run_start(self.spin_limit.value(),
                                          self.chk_debug.isChecked(),
                                          self._step_delay()),
            "미리보기"))
        r.addSeparator()
        # 단계 하나만 돌리는 길은 남겨 둔다. CLI 에 있는 명령이 앱에서 사라지면,
        # 한 단계만 다시 하고 싶은 사람이 터미널로 나가야 한다.
        r.addAction(f"{W.BAKE} 만",
                    lambda: self._launch(self.editor.materialize_start, W.BAKE))
        r.addAction("학습만", lambda: self._launch(self.editor.train_start, "학습"))
        r.addSeparator()
        r.addAction("중단", self._stop)

        m.addMenu("도움말(&H)").addAction("조작법", self._show_help)

    def _build_toolbar(self) -> None:
        tb = QtWidgets.QToolBar("실행")
        tb.setMovable(False)
        self.addToolBar(tb)

        # 버튼은 둘이다. **실행**은 이 그래프가 할 수 있는 데까지 끝까지 가고,
        # **미리보기**는 몇 건만 돌려 중간값을 본다. 단계를 버튼으로 나눠 두면
        # 따로 두면 누르는 순서를 사람이 외워야 하는데, 그 순서는 그래프가 이미 안다.
        self.act_run = tb.addAction("실행", self._run_pipeline)
        self.act_run.setToolTip(
            f"준비 → {W.BAKE} → 학습 → 추론 순으로 끝까지 갑니다.\n"
            "이 그래프에 없는 단계는 건너뜁니다.")
        self.act_preview = tb.addAction("미리보기", lambda: self._launch(
            lambda: self.editor.run_start(self.spin_limit.value(),
                                          self.chk_debug.isChecked(),
                                          self._step_delay()),
            "미리보기"))
        self.act_preview.setToolTip("샘플 몇 건만 돌려 중간값을 봅니다. 학습하지 않습니다")
        self.act_stop = tb.addAction("중단", self._stop)
        self.act_stop.setEnabled(False)

        tb.addSeparator()
        self.act_follow = tb.addAction("실행 따라가기")
        self.act_follow.setCheckable(True)
        self.act_follow.setChecked(True)
        self.act_follow.setToolTip(
            "켜면 지금 도는 상자가 화면 밖일 때 그쪽으로 화면을 옮깁니다.\n"
            "켜진 동안 버튼이 밝게 표시됩니다.")

        self.act_slow = tb.addAction("천천히 보기")
        self.act_slow.setCheckable(True)
        self.act_slow.setToolTip(
            "상자 하나를 지날 때마다 1초씩 쉽니다.\n"
            "빠른 상자는 눈 깜짝할 새에 지나가서 어디를 도는지 보이지 않습니다.\n"
            "무엇이 어떤 순서로 도는지 눈으로 확인할 때만 켭니다 — 그만큼 느려집니다.")

        self.chk_debug = QtWidgets.QCheckBox("Debug Output")
        self.chk_debug.setToolTip(
            "상자마다 중간 결과를 그림으로 남깁니다.\n"
            "꺼져 있으면 만들지도 않습니다 — 켜면 그만큼 느려집니다.")
        tb.addWidget(self.chk_debug)

        # `샘플 8` 만 적혀 있어 무슨 숫자인지 알 수 없었다. 어느 버튼이 쓰는 값인지
        # 이름에 넣는다.
        lbl = QtWidgets.QLabel("  미리보기 샘플 ")
        lbl.setToolTip("`미리보기` 가 돌릴 샘플 수입니다. `실행` 은 전부 돕니다.")
        tb.addWidget(lbl)
        self.spin_limit = QtWidgets.QSpinBox()
        self.spin_limit.setRange(1, 99999)
        self.spin_limit.setValue(8)
        self.spin_limit.setFixedWidth(72)
        self.spin_limit.setSuffix(" 건")
        self.spin_limit.setToolTip(lbl.toolTip())
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
            self.lbl_graph.setText("컴파일되지 않았습니다: "
                                   + (self.editor.error or "").splitlines()[0])
            return
        fixed = {k: (int(v[0]), int(v[1])) for k, v in (self.editor.layout or {}).items()
                 if isinstance(v, (list, tuple)) and len(v) == 2}
        # 어느 포트에 무엇이 꽂혀 있는지. 포트에 호버했을 때 "지금 꽂힌 것" 으로 뜬다 —
        # 처음 보는 사람이 제일 먼저 궁금해하는 것이다.
        from ..layout import node_label

        wired = {}
        for n in cg.nodes.values():
            for port, src in (n.inputs or {}).items():
                sn, _, sp = src.partition(":")
                wired[f"{n.id}:{port}"] = f"{node_label(cg, sn)} 의 {sp}"

        self.canvas.load(
            cg, self.editor.expanded, fixed or None,
            compat=compat_matrix(cg),
            occupied=occupied_inputs(cg),
            boundary=self.editor.graph.materialize.boundary,
            wired_from=wired,
        )
        warn = "" if self.editor.valid else "   ⚠ 저장·실행 불가"
        self.lbl_graph.setText(
            f"상자 {len(self.canvas.cards)} · 노드 {len(cg.nodes)} · 배선 {len(cg.edges)} · "
            f"{W.DEPTH} {max(cg.lanes.values()) + 1}단   {cg.spec_hash}{warn}")

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
            # 놓은 자리에 **그대로** 두면 기존 상자 위에 겹친다. 실제로 그랬다 —
            # 새 상자가 아래 상자를 덮어 둘 다 못 읽게 됐다. 비어 있는 가장 가까운
            # 자리로 밀어 둔다.
            self.editor.move_node(nid, *self.canvas.free_spot(x, y))
        self.reload_graph()
        if nid in self.canvas.cards:
            self.canvas.cards[nid].setSelected(True)
        self.statusBar().showMessage(f"추가: {nid}  ({node_type})", 4000)

    def _remove_node(self, nid: str) -> None:
        if not nid:
            return
        ok = QtWidgets.QMessageBox.question(
            self, "노드 삭제",
            f"'{nid}' 를 지웁니다. 이 노드에 닿은 배선도 함께 사라집니다.",
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
            self.statusBar().showMessage(f"저장했습니다: {res.get('path', '')}", 5000)
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

    def _run_pipeline(self, resume: Any = None) -> None:
        """`실행`. 굽다 만 것이 있으면 편집기가 되묻고, 그 답을 갖고 다시 부른다.

        말없이 이어받으면 예전 파라미터로 구운 샘플이 새 학습에 섞이고, 말없이 지우면
        구운 시간이 날아간다. 양쪽 다 나빠서 사람이 고른다.
        """
        res = self.editor.pipeline_start(resume, self._step_delay())
        ask = res.get("ask_resume")
        if ask:
            self._ask_resume(ask)
            return
        if not res.get("ok"):
            self._refuse("실행", res)
            return
        self.run_dock.show()
        self.run_dock.raise_()
        self.watcher.start()
        self.statusBar().showMessage("실행 시작", 3000)

    def _step_delay(self) -> float:
        """`천천히 보기` 가 켜져 있으면 1초. 결과는 바뀌지 않고 보일 뿐이다."""
        return 1.0 if self.act_slow.isChecked() else 0.0

    def _ask_resume(self, ask: Dict[str, Any]) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("굽다 만 것이 있습니다")
        box.setText(f"{W.SHARD} {ask['shards']}개 · 샘플 {ask['samples']}건이 "
                    "이미 준비되어 있습니다.")
        box.setInformativeText(
            f"{ask['out_dir']}\n\n"
            "이어 받으면 그 샘플은 다시 굽지 않습니다. 전처리 설정을 고쳤다면 "
            "예전 값으로 구운 샘플이 섞이므로 처음부터 굽는 편이 맞습니다.")
        keep = box.addButton("이어 받기", QtWidgets.QMessageBox.AcceptRole)
        fresh = box.addButton("처음부터", QtWidgets.QMessageBox.DestructiveRole)
        box.addButton("취소", QtWidgets.QMessageBox.RejectRole)
        box.exec()
        picked = box.clickedButton()
        if picked is keep:
            self._later(lambda: self._run_pipeline(True))
        elif picked is fresh:
            self._later(lambda: self._run_pipeline(False))

    def _export_model(self) -> None:
        """`모델 Export`. 버튼 하나가 CLI 명령 하나다 — `vlmt export` 를 그대로 띄운다."""
        res = self.editor.export_start()
        if not res.get("ok"):
            self._refuse("모델 Export", res)
            return
        self.run_dock.show()
        self.run_dock.raise_()
        self.watcher.start()
        self.statusBar().showMessage("Export 시작", 3000)

    def _stop(self) -> None:
        res = self.editor.run_stop()
        if not res.get("ok"):
            self.statusBar().showMessage(res.get("reason", ""), 4000)

    def set_running(self, on: bool) -> None:
        """돌고 있는 동안에는 그래프를 못 고치게 한다. 실행은 **디스크의 스펙**을
        도는데 편집은 메모리를 바꾼다 — 둘이 갈라지면 화면과 다른 것이 돌아간다."""
        for a in (self.act_run, self.act_preview):
            a.setEnabled(not on)
        self.act_stop.setEnabled(on)
        self.library.setEnabled(not on)
        self.inspector.setEnabled(not on)
        # **보는 것은 막지 않는다.** 전에는 `setInteractive(False)` 로 캔버스의 마우스
        # 이벤트를 통째로 껐는데, 그러면 팬도 휠 줌도 선택도 죽는다 — 실행 중에 아래쪽
        # 노드를 보러 갈 수가 없었다. 실행 중에 제일 하고 싶은 일이 그것인데.
        #
        # 막아야 하는 것은 **그래프를 바꾸는 것**뿐이다. 실행은 디스크의 스펙을 도는데
        # 편집은 메모리를 바꾸고, 둘이 갈라지면 화면과 다른 것이 돌아간다.
        self.canvas.set_read_only(on)

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
            f"  상자 우클릭 — 펼치기/접기 · {W.BOUNDARY} · 삭제\n\n"
            f"실행 — `실행` 은 준비 → {W.BAKE} → 학습 → 추론을 끝까지,\n"
            "       `미리보기` 는 샘플 몇 건만 돌려 중간값을 봅니다.")


def launch(project_path: str, extra_modules: tuple = ()) -> int:
    """창을 띄운다. `vlmt app`이 부르는 자리다."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("VLM Trainer")
    from ...spec.line import is_line

    if is_line(project_path):  # 5노드는 전용 화면이 연다
        from .line_window import launch_line

        return launch_line(project_path, extra_modules)
    editor = Editor.open(project_path)
    editor.extra_modules = tuple(extra_modules)
    win = EditorWindow(editor)
    win.show()
    return app.exec()
