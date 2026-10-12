"""5노드 화면 — Line 파일을 읽기 전용으로 보여 주고 실행한다.

왼쪽은 위에서 아래로 흐르는 카드 다섯 장(스테이션), 오른쪽 위는 고른 스테이션의 설정,
오른쪽 아래는 **항상 보이는 NG 표**다. NG 가 숨으면 라인 옆 배출함을 가린 것과 같다.

실행은 `vlmt pipeline <line.yaml>` 을 하위 프로세스로 띄우는 것이 전부다(버튼 하나가 CLI 명령 하나).
이 창은 그 진행 파일을 읽어 그릴 뿐이고, 무엇을 그릴지의 계산은 `ui/line_model.py` 에 있다.
설정 편집은 3b 에서 붙는다 — 지금은 읽기 전용이다.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .. import tokens as T
from .. import line_model as M
from ..api import Editor
from ...engine import runner as runner_mod
from ...engine import samples as samples_mod
from ...spec.line import describe_file
from .window import FONT, QSS

POLL_MS = 700

LAMP_COLOR = {
    M.PENDING: T.STATE["pending"], M.RUNNING: T.STATE["running"], M.OK: T.STATE["success"],
    M.NG: T.STATE["partial"], M.ERROR: T.STATE["failed"],
}
MUTED = f"color:{T.NODE['muted']}; font-size:11px;"


class CardWidget(QtWidgets.QFrame):
    clicked = QtCore.Signal(str)

    def __init__(self, card: M.Card, parent: Any = None) -> None:
        super().__init__(parent)
        self.cid = card.id
        self.setObjectName("card")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        head = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-size:14px; font-weight:600; color:#FFFFFF;")
        self.lamp = QtWidgets.QLabel()
        head.addWidget(self.title, 1)
        head.addWidget(self.lamp)
        lay.addLayout(head)
        self.line2 = QtWidgets.QLabel()
        self.line2.setStyleSheet("font-size:12px;")
        lay.addWidget(self.line2)
        self.hint = QtWidgets.QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(MUTED)
        lay.addWidget(self.hint)
        self.update_card(card)
        self.set_selected(False)

    def update_card(self, c: M.Card) -> None:
        self.title.setText(f"{c.number}  {c.label}")
        color = LAMP_COLOR[c.lamp]
        # 색만으로 구분하지 않는다 — 램프 옆에 글자가 있다.
        self.lamp.setText(f'<span style="color:{color}">●</span> {c.lamp_text}')
        self.line2.setText(c.line2)
        self.hint.setText(c.hint)
        self.setToolTip(f"{c.label} — {c.hint}")

    def set_selected(self, on: bool) -> None:
        border = T.NODE["border_selected"] if on else T.NODE["border"]
        bg = T.NODE["bg_selected"] if on else T.NODE["bg"]
        self.setStyleSheet(f"QFrame#card {{ background:{bg}; border:1px solid {border}; border-radius:3px; }}"
                           "QFrame#card QLabel { background:transparent; }")

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:  # noqa: N802
        self.clicked.emit(self.cid)
        super().mousePressEvent(ev)


class LineWindow(QtWidgets.QMainWindow):
    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.editor = editor
        self.resize(1180, 900)
        self.setStyleSheet(QSS)
        self.desc: Dict[str, Any] = {}
        self.cards: List[M.Card] = []
        self.widgets: Dict[str, CardWidget] = {}
        self.picked = ""
        self.thumbs: Dict[str, str] = {}      # 건 id -> 사진 경로
        self._ng_rows: List[M.NgRow] = []

        self._build_toolbar()
        self._build_body()
        self.lbl_run = QtWidgets.QLabel()
        self.lbl_run.setObjectName("runline")
        self.statusBar().addPermanentWidget(self.lbl_run)
        self.lbl_info = QtWidgets.QLabel()
        self.statusBar().addWidget(self.lbl_info, 1)

        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self.refresh)

        self.reload()

    # ── 구성 ────────────────────────────────────────────────────────────
    def _build_toolbar(self) -> None:
        tb = QtWidgets.QToolBar("실행")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.act_run = tb.addAction("실행", self._run)
        self.act_run.setToolTip("준비 → 사전 준비 → 학습 → 검증 순으로 끝까지 갑니다.")
        self.act_stop = tb.addAction("중단", self._stop)
        self.act_stop.setEnabled(False)
        tb.addSeparator()
        self.act_reread = tb.addAction("다시 읽기", self.reload)
        self.act_reread.setToolTip("표(CSV)와 Line 파일을 다시 읽습니다. 표를 고쳤다면 눌러 주세요.")

    def _build_body(self) -> None:
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(split)

        left = QtWidgets.QWidget()
        self.left_lay = QtWidgets.QVBoxLayout(left)
        self.left_lay.setContentsMargins(14, 14, 14, 14)
        self.left_lay.setSpacing(0)
        split.addWidget(left)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 820])

        top = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(top)
        tl.setContentsMargins(10, 10, 10, 6)
        self.lbl_settings = QtWidgets.QLabel("스테이션을 누르면 설정이 보입니다")
        self.lbl_settings.setStyleSheet("font-weight:600;")
        tl.addWidget(self.lbl_settings)
        self.tbl_settings = QtWidgets.QTableWidget(0, 2)
        self.tbl_settings.setHorizontalHeaderLabels(["설정", "값"])
        self.tbl_settings.horizontalHeader().setStretchLastSection(True)
        self.tbl_settings.setColumnWidth(0, 190)
        self.tbl_settings.verticalHeader().setVisible(False)
        self.tbl_settings.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)  # 읽기 전용
        self.tbl_settings.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        tl.addWidget(self.tbl_settings)
        right.addWidget(top)

        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(10, 6, 10, 10)
        self.lbl_ng = QtWidgets.QLabel("NG 0건")
        self.lbl_ng.setStyleSheet("font-weight:600;")
        bl.addWidget(self.lbl_ng)
        self.tbl_ng = QtWidgets.QTableWidget(0, 4)
        self.tbl_ng.setHorizontalHeaderLabels(["사진", "건", "스테이션", "원인"])
        self.tbl_ng.horizontalHeader().setStretchLastSection(True)
        self.tbl_ng.verticalHeader().setVisible(False)
        self.tbl_ng.verticalHeader().setDefaultSectionSize(48)
        self.tbl_ng.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_ng.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_ng.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_ng.itemSelectionChanged.connect(self._show_detail)
        bl.addWidget(self.tbl_ng, 1)
        self.detail = QtWidgets.QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMaximumHeight(150)
        self.detail.setPlaceholderText("NG 건을 누르면 원인과 고치는 방법이 보입니다.")
        bl.addWidget(self.detail)
        self.lbl_fix = QtWidgets.QLabel("고치려면 표(CSV)를 고치고 `다시 읽기` 를 누르세요.")
        self.lbl_fix.setStyleSheet(MUTED)
        bl.addWidget(self.lbl_fix)
        right.addWidget(bottom)
        right.setSizes([260, 560])

    # ── 읽기 ────────────────────────────────────────────────────────────
    def reload(self) -> None:
        """Line 과 표를 다시 읽는다. 읽기 전용이라 디스크에 쓰는 것은 없다."""
        try:
            old = self.editor
            new = Editor.open_line(old.path)
            # 돌고 있는 실행의 손잡이를 넘긴다. 새로 만든 편집기가 빈 손이면 다시 읽는 순간
            # 실행 중인 것이 화면에서 사라진다.
            for f in ("proc", "run_id", "progress_path", "train_progress_path", "console_path",
                      "phase", "stopped", "launched_at", "finished_at", "extra_modules"):
                setattr(new, f, getattr(old, f))
            self.editor = new
            self.desc = describe_file(self.editor.path)
        except Exception as exc:  # 스펙이 어긋난 채 열린 창이 죽지 않게 한다
            self.lbl_info.setText("읽지 못했다: " + str(exc).splitlines()[0])
            self.detail.setPlainText(str(exc))
            return
        cg = self.editor.compiled
        self.lbl_info.setText(
            f"{self.desc['name'] or self.desc['id']} · 스테이션 {len(self.desc['stations'])} · "
            f"노드 {len(cg.nodes) if cg else 0} · {cg.spec_hash if cg else ''}")
        self.setWindowTitle(f"{self.desc['name'] or self.desc['id']} — VLM Trainer (5노드)")
        self._load_thumbs()
        self.refresh()

    def _load_thumbs(self) -> None:
        """NG 표의 사진. 건 id → 사진 경로. 표를 읽지 못하면 사진 없이 간다."""
        self.thumbs = {}
        cg = self.editor.compiled
        if cg is None:
            return
        try:
            space = samples_mod.load(cg.sample_space, os.path.dirname(self.editor.path))
        except Exception:
            return
        col = next((n.params.get("column", "image_path") for n in cg.nodes.values()
                    if n.ref.startswith("source.image")), "image_path")
        for r in space.rows:
            self.thumbs[str(r[space.key])] = space_path(space.root, str(r.get(col, "")))

    def refresh(self) -> None:
        cg = self.editor.compiled
        if cg is None:
            return
        run = self.editor.run_state()
        snap = self.editor._read_progress()
        rep = runner_mod.report_from_snapshot(snap) if snap else None
        self.cards = M.cards(self.desc, cg, rep, run)
        self._paint_cards()
        quarantine = list((snap or {}).get("quarantine") or [])
        self._paint_ng(M.ng_rows(self.cards, quarantine), int((snap or {}).get("quarantine_total") or len(quarantine)))
        if self.picked:
            self._show_settings(self.picked)

        running = bool(run.get("running"))
        self.act_run.setEnabled(not running)
        self.act_stop.setEnabled(running)
        head = str((run.get("pipeline") or {}).get("headline") or "")
        self.lbl_run.setText(f"{run.get('run_id', '')} {head}".strip())
        if running and not self._poll.isActive():
            self._poll.start()
        if not running and self._poll.isActive():
            self._poll.stop()
            self._finished(run)

    def _finished(self, run: Dict[str, Any]) -> None:
        if run.get("aborted"):
            self.statusBar().showMessage(f"중단: {run['aborted']}", 10000)
        elif run.get("exit") and not run.get("stopped"):
            self.statusBar().showMessage(f"실패 (exit {run['exit']})", 10000)
            self.detail.setPlainText(str(run.get("console") or ""))
        elif run.get("stopped"):
            self.statusBar().showMessage("중지", 5000)
        else:
            self.statusBar().showMessage("끝", 5000)

    # ── 그리기 ──────────────────────────────────────────────────────────
    def _paint_cards(self) -> None:
        if not self.widgets:
            for i, c in enumerate(self.cards):
                w = CardWidget(c)
                w.clicked.connect(self._pick)
                self.widgets[c.id] = w
                self.left_lay.addWidget(w)
                if i < len(self.cards) - 1:
                    arrow = QtWidgets.QLabel("▼")
                    arrow.setAlignment(QtCore.Qt.AlignCenter)
                    arrow.setStyleSheet(f"color:{T.NODE['muted']}; padding:3px;")
                    self.left_lay.addWidget(arrow)
            self.left_lay.addStretch(1)
        for c in self.cards:
            self.widgets[c.id].update_card(c)

    def _paint_ng(self, rows: List[M.NgRow], total: int) -> None:
        self._ng_rows = rows
        self.lbl_ng.setText(f"NG {total}건" + (f" (표시 {len(rows)}건)" if total > len(rows) else ""))
        keep = self.tbl_ng.currentRow()
        self.tbl_ng.setRowCount(len(rows))
        for i, r in enumerate(rows):
            pic = QtWidgets.QLabel()
            path = self.thumbs.get(r.sample_key, "")
            if path and os.path.exists(path):
                pm = QtGui.QPixmap(path)
                if not pm.isNull():
                    pic.setPixmap(pm.scaled(44, 44, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            pic.setAlignment(QtCore.Qt.AlignCenter)
            self.tbl_ng.setCellWidget(i, 0, pic)
            self.tbl_ng.setItem(i, 1, QtWidgets.QTableWidgetItem(r.sample_key))
            self.tbl_ng.setItem(i, 2, QtWidgets.QTableWidgetItem(
                f"{r.card_number} {r.card_label}" if r.card_number else r.node_id))
            self.tbl_ng.setItem(i, 3, QtWidgets.QTableWidgetItem(r.one_line))
        self.tbl_ng.setColumnWidth(0, 56)
        if 0 <= keep < len(rows):
            self.tbl_ng.selectRow(keep)

    def _pick(self, cid: str) -> None:
        self.picked = cid
        for k, w in self.widgets.items():
            w.set_selected(k == cid)
        self._show_settings(cid)

    def _show_settings(self, cid: str) -> None:
        c = next((x for x in self.cards if x.id == cid), None)
        if c is None:
            return
        self.lbl_settings.setText(f"{c.number}  {c.label} — 설정(읽기 전용)")
        rows = list(c.settings.items())
        self.tbl_settings.setRowCount(len(rows))
        for i, (_, s) in enumerate(rows):
            self.tbl_settings.setItem(i, 0, QtWidgets.QTableWidgetItem(str(s["label"])))
            self.tbl_settings.setItem(i, 1, QtWidgets.QTableWidgetItem(_show(s["value"])))

    def _show_detail(self) -> None:
        i = self.tbl_ng.currentRow()
        if not (0 <= i < len(self._ng_rows)):
            return
        r = self._ng_rows[i]
        where = f"{r.card_number} {r.card_label}" if r.card_number else r.node_id
        self.detail.setPlainText(f"건 {r.sample_key} · {where} ({r.node_id})\n\n{r.cause}"
                                 + (f"\n\n해결: {r.hint}" if r.hint else ""))

    # ── 실행 ────────────────────────────────────────────────────────────
    def _run(self, resume: Optional[bool] = None) -> None:
        res = self.editor.pipeline_start(resume, 0.0)
        ask = res.get("ask_resume")
        if ask:
            self._ask_resume(ask)
            return
        if not res.get("ok"):
            self.statusBar().showMessage(str(res.get("reason") or "실행하지 못했다"), 8000)
            if res.get("detail"):
                self.detail.setPlainText(str(res["detail"]))
            return
        self.statusBar().showMessage("실행 시작", 3000)
        self._poll.start()
        self.refresh()

    def _ask_resume(self, ask: Dict[str, Any]) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("굽다 만 것이 있습니다")
        box.setText(f"준비된 묶음 {ask['shards']}개 · 샘플 {ask['samples']}건이 이미 있습니다.")
        box.setInformativeText(f"{ask['out_dir']}\n\n이어 받으면 그 샘플은 다시 만들지 않습니다. "
                               "설정을 고쳤다면 처음부터 하는 편이 맞습니다.")
        keep = box.addButton("이어 받기", QtWidgets.QMessageBox.AcceptRole)
        fresh = box.addButton("처음부터", QtWidgets.QMessageBox.DestructiveRole)
        box.addButton("취소", QtWidgets.QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is keep:
            QtCore.QTimer.singleShot(0, lambda: self._run(True))
        elif box.clickedButton() is fresh:
            QtCore.QTimer.singleShot(0, lambda: self._run(False))

    def _stop(self) -> None:
        res = self.editor.run_stop()
        if not res.get("ok"):
            self.statusBar().showMessage(str(res.get("reason", "")), 4000)


def space_path(root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(root, rel))


def _show(v: Any) -> str:
    if v is None:
        return "(묶음 상자의 기본값)"
    if isinstance(v, dict):
        return ", ".join(f"{k}={x}" for k, x in v.items()) or "-"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def launch_line(path: str, extra_modules: tuple = ()) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("VLM Trainer")
    editor = Editor.open_line(path)
    editor.extra_modules = tuple(extra_modules)
    win = LineWindow(editor)
    win.show()
    return app.exec()
