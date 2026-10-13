"""5노드 화면 — Line 파일을 보고, 설정을 고치고, 실행한다.

왼쪽은 위에서 아래로 흐르는 카드 다섯 장(스테이션), 오른쪽 위는 고른 스테이션의 설정,
오른쪽 아래는 **항상 보이는 NG 표**다. NG 가 숨으면 라인 옆 배출함을 가린 것과 같다.

고칠 수 있는 것은 **설정 값뿐**이다. 상자를 더하거나 배선을 바꾸는 길은 없고, 그것은 "11노드로
내보내기"의 몫이다. 모든 편집은 `Editor.line_*` 가 Line 의 매핑을 고친 뒤 다시 펼쳐 컴파일하고,
통과해야 받아들인다 — 거부되면 원래 값이 남고 이유가 상태줄에 뜬다.

기본 모드는 기본 설정만, **전문가 모드**는 전부와 안쪽 상자의 값을 보여 준다. 전문가 모드는
창을 닫으면 꺼진다.

실행은 `vlmt pipeline <line.yaml>` 을 하위 프로세스로 띄우는 것이 전부다(버튼 하나가 CLI 명령 하나).
무엇을 그릴지의 계산은 `ui/line_model.py` 에 있다.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .. import tokens as T
from .. import line_model as M
from ..api import Editor
from ...engine import runner as runner_mod
from ...engine import samples as samples_mod
from .canvas import GraphCanvas
from .line_settings import _BAD, fill_table, make_editor, new_table, settings_rows
from .window import FONT, QSS

POLL_MS = 700

LAMP_COLOR = {
    M.PENDING: T.STATE["pending"], M.RUNNING: T.STATE["running"], M.OK: T.STATE["success"],
    M.NG: T.STATE["partial"], M.ERROR: T.STATE["failed"],
}
MUTED = f"color:{T.NODE['muted']}; font-size:11px;"


class CardWidget(QtWidgets.QFrame):
    clicked = QtCore.Signal(str)
    opened = QtCore.Signal(str)

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
        self.title.setText(f"{M.mark(c.number)}  {c.label}")
        color = LAMP_COLOR[c.lamp]
        # 색만으로 구분하지 않는다 — 램프 옆에 글자가 있다.
        self.lamp.setText(f'<span style="color:{color}">●</span> {c.lamp_text}')
        self.line2.setText(c.line2)
        self.hint.setText(c.hint)
        self.setToolTip(f"{c.label} — {c.hint}\n더블클릭: 안쪽 보기")

    def set_selected(self, on: bool) -> None:
        border = T.NODE["border_selected"] if on else T.NODE["border"]
        bg = T.NODE["bg_selected"] if on else T.NODE["bg"]
        self.setStyleSheet(f"QFrame#card {{ background:{bg}; border:1px solid {border}; border-radius:3px; }}"
                           "QFrame#card QLabel { background:transparent; }")

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:  # noqa: N802
        self.clicked.emit(self.cid)
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:  # noqa: N802
        self.opened.emit(self.cid)


class InnerView(QtWidgets.QDialog):
    """안쪽 보기 — 펼친 11노드 그래프를 읽기 전용으로. 고른 스테이션의 상자가 선택 색으로 보인다."""

    def __init__(self, cg: Any, members: List[str], title: str, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"안쪽 보기 — {title} (읽기 전용)")
        self.resize(1000, 800)
        self.setStyleSheet(QSS)
        lay = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("상자와 배선은 여기서 고치지 않습니다. 고치려면 `11노드로 내보내기`를 쓰세요.")
        note.setStyleSheet(MUTED)
        lay.addWidget(note)
        self.canvas = GraphCanvas(self)
        lay.addWidget(self.canvas, 1)
        self.canvas.load(cg, set(), None, compat={}, occupied=[], boundary=[], wired_from={})
        self.canvas.set_read_only(True)
        for nid in members:
            card = self.canvas.cards.get(nid)
            if card is not None:
                card.setSelected(True)
        QtCore.QTimer.singleShot(0, self.canvas.fit)


def _short_line(text: str) -> str:
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    return lines[0] if lines else ""


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
        self._running = False
        self._excluded = 0                    # 표의 `제외` 열 때문에 건너뛴 건수
        self._fix_key = ""                    # 고치기 칸에 올라와 있는 건
        self._fix_old: Dict[str, str] = {}

        self._build_toolbar()
        self._build_body()
        self.lbl_run = QtWidgets.QLabel()
        self.lbl_run.setObjectName("runline")
        self.statusBar().addPermanentWidget(self.lbl_run)
        self.lbl_info = QtWidgets.QLabel()
        self.statusBar().addWidget(self.lbl_info, 1)
        self.lbl_refuse = QtWidgets.QLabel()
        self.lbl_refuse.setStyleSheet(f"color:{T.STATE['failed']}; font-size:11px;")
        self.statusBar().addWidget(self.lbl_refuse, 2)

        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self.refresh)

        self.reload(confirm=False)

    # ── 구성 ────────────────────────────────────────────────────────────
    def _build_toolbar(self) -> None:
        tb = QtWidgets.QToolBar("실행")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.act_new = tb.addAction("새 5노드…", self._new_line)
        self.act_new.setToolTip("빈 폴더를 골라 새 5노드 프로젝트를 만듭니다. 표는 `데이터` 에서 넣습니다.")
        tb.addSeparator()
        self.act_run = tb.addAction("실행", self._run)
        self.act_run.setToolTip("준비 → 사전 준비 → 학습 → 검증 순으로 끝까지 갑니다.")
        self.act_stop = tb.addAction("중단", self._stop)
        self.act_stop.setEnabled(False)
        tb.addSeparator()
        self.act_save = tb.addAction("저장", self._save)
        self.act_save.setShortcut(QtGui.QKeySequence.Save)
        self.act_data = tb.addAction("데이터", self._open_data)
        self.act_data.setToolTip("표 파일, 열 지정, 정답 항목, 학습/검증 나누기를 정합니다.")
        self.act_reread = tb.addAction("다시 읽기", lambda: self.reload())
        self.act_reread.setToolTip("표(CSV)와 Line 파일을 다시 읽습니다. 표를 고쳤다면 눌러 주세요.")
        self.act_verdict = tb.addAction("검증 결과", self._open_verdict)
        self.act_verdict.setToolTip("검증 사진에 모델이 낸 답을 정답과 나란히 보고 OK / NG 를 정합니다.")
        tb.addSeparator()
        self.act_expert = tb.addAction("전문가 모드")
        self.act_expert.setCheckable(True)
        self.act_expert.setToolTip("켜면 모든 설정과 안쪽 상자의 값이 보입니다. 창을 닫으면 꺼집니다.")
        self.act_expert.toggled.connect(lambda _on: self._show_settings(self.picked))
        self.act_inner = tb.addAction("안쪽 보기", lambda: self._open_inner(self.picked))
        self.act_inner.setToolTip("고른 스테이션 안쪽의 상자와 배선을 읽기 전용으로 봅니다(카드 더블클릭도 같습니다).")
        self.act_export = tb.addAction("11노드로 내보내기", self._export)
        self.act_export.setToolTip("상자와 배선을 직접 고칠 때 씁니다. 내보낸 뒤에는 5노드와 연결이 끊깁니다.")

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
        self.tbl_settings = new_table()
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
        # 오류는 한 줄 요약(몇 번 스테이션인지 포함)이 먼저 보이고, 네 가지 상세는 접혀 있다.
        self.lbl_sum = QtWidgets.QLabel("")
        self.lbl_sum.setWordWrap(True)
        self.lbl_sum.setStyleSheet(f"color:{T.STATE['failed']};")
        bl.addWidget(self.lbl_sum)
        self.btn_more = QtWidgets.QToolButton()
        self.btn_more.setText("상세 보기")
        self.btn_more.setCheckable(True)
        self.btn_more.setVisible(False)
        self.btn_more.toggled.connect(lambda on: (self.detail.setVisible(on), self.btn_more.setText("상세 접기" if on else "상세 보기")))
        bl.addWidget(self.btn_more)
        self.detail = QtWidgets.QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMaximumHeight(150)
        self.detail.setVisible(False)
        bl.addWidget(self.detail)
        self.lbl_fix = QtWidgets.QLabel("고치려면 NG 한 건을 누르세요. 표(CSV)를 직접 고쳐도 됩니다 — 고친 뒤 `다시 읽기`.")
        self.lbl_fix.setStyleSheet(MUTED)
        bl.addWidget(self.lbl_fix)
        self.fix_box = QtWidgets.QGroupBox("이 건 고치기")
        fl = QtWidgets.QVBoxLayout(self.fix_box)
        self.tbl_fix = QtWidgets.QTableWidget(0, 2)
        self.tbl_fix.setHorizontalHeaderLabels(["열", "값"])
        self.tbl_fix.horizontalHeader().setStretchLastSection(True)
        self.tbl_fix.verticalHeader().setVisible(False)
        self.tbl_fix.setMaximumHeight(150)
        fl.addWidget(self.tbl_fix)
        self.lbl_fixmsg = QtWidgets.QLabel()
        self.lbl_fixmsg.setStyleSheet(MUTED)
        self.lbl_fixmsg.setWordWrap(True)
        fl.addWidget(self.lbl_fixmsg)
        hb = QtWidgets.QHBoxLayout()
        self.btn_fix = QtWidgets.QPushButton("저장 후 다시 읽기")
        self.btn_fix.setToolTip("고친 칸만 표(CSV)에 씁니다. 처음 고칠 때 원본을 `표.bak.csv` 로 남깁니다.")
        self.btn_fix.clicked.connect(self._fix_save)
        self.btn_skip = QtWidgets.QPushButton("이 건 제외")
        self.btn_skip.setToolTip("표의 `제외` 열에 x 를 적습니다. 행을 지우지 않으며, x 를 지우면 되살아납니다.")
        self.btn_skip.clicked.connect(lambda: self._fix_exclude(True))
        hb.addWidget(self.btn_fix)
        hb.addWidget(self.btn_skip)
        hb.addStretch(1)
        fl.addLayout(hb)
        self.fix_box.setVisible(False)
        bl.addWidget(self.fix_box)
        right.addWidget(bottom)
        right.setSizes([320, 500])

    # ── 읽기 ────────────────────────────────────────────────────────────
    def reload(self, confirm: bool = True) -> None:
        """Line 과 표를 다시 읽는다. 저장하지 않은 변경이 있으면 사라지므로 먼저 묻는다."""
        if confirm and self.editor.dirty:
            ans = QtWidgets.QMessageBox.question(
                self, "저장하지 않은 변경", "다시 읽으면 저장하지 않은 변경이 사라집니다. 계속할까요?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if ans != QtWidgets.QMessageBox.Yes:
                return
        try:
            old = self.editor
            new = Editor.open_line(old.path)
            # 돌고 있는 실행의 손잡이를 넘긴다. 새로 만든 편집기가 빈 손이면 다시 읽는 순간
            # 실행 중인 것이 화면에서 사라진다.
            for f in ("proc", "run_id", "progress_path", "train_progress_path", "console_path",
                      "phase", "stopped", "launched_at", "finished_at", "extra_modules"):
                setattr(new, f, getattr(old, f))
            self.editor = new
            self.desc = self.editor.line_describe()
        except Exception as exc:  # 스펙이 어긋난 채 열린 창이 죽지 않게 한다
            self.lbl_info.setText("읽지 못했습니다: " + str(exc).splitlines()[0])
            self._set_detail(str(exc))
            return
        self._title()
        self._load_thumbs()
        self.refresh()
        self._show_settings(self.picked)
        if self._fix_key:
            self._show_fix(self._fix_key)      # 표가 바뀌었으니 고치기 칸도 새 값으로

    def _title(self) -> None:
        cg = self.editor.compiled
        name = self.desc.get("name") or self.desc.get("id") or os.path.basename(self.editor.path)
        self.setWindowTitle(f"{name} — VLM Trainer (5노드)" + ("  *저장 안 됨" if self.editor.dirty else ""))
        self.lbl_info.setText(
            f"{name} · 스테이션 {len(self.desc.get('stations') or [])} · "
            f"노드 {len(cg.nodes) if cg else 0} · {cg.spec_hash if cg else ''}")

    def _load_thumbs(self) -> None:
        """NG 표의 사진. 건 id → 사진 경로. 표를 읽지 못하면 사진 없이 간다."""
        self.thumbs = {}
        self._excluded = 0
        cg = self.editor.compiled
        if cg is None:
            return
        try:
            space = samples_mod.load(cg.sample_space, os.path.dirname(self.editor.path))
        except Exception:
            return
        col = next((n.params.get("column", "image_path") for n in cg.nodes.values()
                    if n.ref.startswith("source.image")), "image_path")
        self._excluded = space.excluded
        for r in space.rows:
            self.thumbs[str(r[space.key])] = space_path(space.root, str(r.get(col, "")))

    def refresh(self) -> None:
        cg = self.editor.compiled
        if cg is None:
            # 새 프로젝트는 표가 없어 컴파일되지 않는다. 이유를 보이고, 실행은 막는다.
            self.act_run.setEnabled(False)
            self.act_save.setEnabled(False)
            self._set_detail("아직 실행할 수 없습니다. 위의 `데이터` 에서 사진·질문·정답 표를 넣으세요."
                                     + chr(10) * 2 + "지금 막힌 이유:" + chr(10) + self.editor.error)
            return
        run = self.editor.run_state()
        snap = self.editor._read_progress()
        rep = runner_mod.report_from_snapshot(snap) if snap else None
        self.cards = M.cards(self.desc, cg, rep, run)
        self._paint_cards()
        quarantine = list((snap or {}).get("quarantine") or [])
        self._paint_ng(M.ng_rows(self.cards, quarantine), int((snap or {}).get("quarantine_total") or len(quarantine)))

        running = bool(run.get("running"))
        self._running = running
        self.act_run.setEnabled(not running)
        self.act_stop.setEnabled(running)
        # 돌고 있는 동안은 설정을 못 고친다. 실행은 디스크의 Line 을 도는데 편집은 메모리를 바꾼다 —
        # 둘이 갈라지면 화면과 다른 것이 돌아간다.
        self.tbl_settings.setEnabled(not running)
        self.act_save.setEnabled(not running)
        self.act_data.setEnabled(not running)
        self.fix_box.setEnabled(not running)
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
            self._set_detail(str(run.get("console") or ""))
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
                w.opened.connect(self._open_inner)
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
        self.lbl_ng.setText(f"NG {total}건" + (f" (표시 {len(rows)}건)" if total > len(rows) else "")
                            + (f" · 제외 {self._excluded}건" if self._excluded else ""))
        self.lbl_ng.setToolTip("제외한 건은 표의 `제외` 열에 x 가 적힌 행입니다. 되살리려면 x 를 지우고 `다시 읽기`.")
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
                f"{M.mark(r.card_number)} {r.card_label}" if r.card_number else r.node_id))
            self.tbl_ng.setItem(i, 3, QtWidgets.QTableWidgetItem(r.one_line))
        self.tbl_ng.setColumnWidth(0, 56)
        if 0 <= keep < len(rows):
            self.tbl_ng.selectRow(keep)

    def _pick(self, cid: str) -> None:
        self.picked = cid
        for k, w in self.widgets.items():
            w.set_selected(k == cid)
        self._show_settings(cid)

    # ── 설정 ────────────────────────────────────────────────────────────
    def _station(self, cid: str) -> Optional[Dict[str, Any]]:
        return next((s for s in self.desc.get("stations") or [] if s["id"] == cid), None)

    def _show_settings(self, cid: str) -> None:
        st = self._station(cid)
        if st is None:
            return
        expert = self.act_expert.isChecked()
        rows = settings_rows(self.editor, st, expert)

        self.lbl_settings.setText(f"{M.mark(st['number'])}  {st['label']} — 설정" + ("  [전문가 모드]" if expert else ""))
        t = self.tbl_settings
        fill_table(t, rows, self._edit_cb)
        t.setEnabled(not self._running)

    def _edit_cb(self, setter: Callable[[Any], Dict[str, Any]], old: Any) -> Callable[[Any], None]:
        def done(v: Any) -> None:
            if v is _BAD:
                QtCore.QTimer.singleShot(0, lambda: (self._refuse("JSON 이 아닙니다 — 예: [448, 448]"),
                                                     self._show_settings(self.picked)))
                return
            if v == old:
                return
            # 값을 보낸 위젯은 표를 다시 그리는 순간 지워진다. 그 위젯의 신호 처리가 끝난 뒤에 바꾼다.
            QtCore.QTimer.singleShot(0, lambda: self._apply(setter, v))

        return done

    def _apply(self, setter: Callable[[Any], Dict[str, Any]], value: Any) -> None:
        res = setter(value)
        if res.get("ok"):
            self.lbl_refuse.setText("")
            self.desc = self.editor.line_describe()
        else:
            self._refuse(res.get("reason", "거부됨"))
            self._set_detail(str(res.get("detail") or ""))
        self._title()
        self.refresh()
        self._show_settings(self.picked)   # 거부됐어도 다시 그려 원래 값으로 되돌린다

    def _refuse(self, why: str) -> None:
        self.lbl_refuse.setText("거부: " + why)

    def _set_detail(self, text: str, where: str = "") -> None:
        """한 줄 요약은 바로 보이고 상세는 접어 둔다. `where` 가 없으면 고른 스테이션을 번호로 붙인다."""
        text = str(text or "")
        if not where:
            st = self._station(self.picked or "")
            where = f"{M.mark(st['number'])} {st['label']}" if st else ""
        first = _short_line(text)
        self.lbl_sum.setText(f"{where} — {first}" if where and first else first)
        self.detail.setPlainText(text)
        self.btn_more.setVisible(bool(text.strip()))
        if not text.strip():
            self.btn_more.setChecked(False)

    def _show_detail(self) -> None:
        i = self.tbl_ng.currentRow()
        if not (0 <= i < len(self._ng_rows)):
            return
        r = self._ng_rows[i]
        where = f"{M.mark(r.card_number)} {r.card_label}" if r.card_number else r.node_id
        # 요약 줄에는 원인만, 건 이름과 노드는 상세 머리에 둔다.
        nn = chr(10) * 2
        self._set_detail(r.cause + f"{nn}건 {r.sample_key} · {r.node_id}" + (f"{nn}해결: {r.hint}" if r.hint else ""), where)
        self._show_fix(r.sample_key)

    # ── NG 한 건 고치기 ─────────────────────────────────────────────────
    def _show_fix(self, key: str) -> None:
        """표의 그 행을 칸마다 보여 준다. CSV 가 아니면 이유만 보이고 고칠 수 없다."""
        self._fix_key, self._fix_old = key, {}
        self.fix_box.setVisible(bool(key))
        if not key:
            return
        res = self.editor.line_row(key)
        ok = bool(res.get("ok"))
        vals = res.get("values") or {}
        self._fix_old = dict(vals)
        self.tbl_fix.setRowCount(len(vals))
        for i, (col, v) in enumerate(vals.items()):
            a = QtWidgets.QTableWidgetItem(col)
            a.setFlags(QtCore.Qt.ItemIsEnabled)
            b = QtWidgets.QTableWidgetItem(v)
            if col == res.get("key"):                  # 건 이름은 NG 목록과 이어 주는 열이라 고치지 않는다
                b.setFlags(QtCore.Qt.ItemIsEnabled)
            self.tbl_fix.setItem(i, 0, a)
            self.tbl_fix.setItem(i, 1, b)
        self.lbl_fixmsg.setText("고친 건은 다시 실행하면 NG 목록에서 사라집니다." if ok else str(res.get("detail") or res.get("reason", "")))
        self.btn_fix.setEnabled(ok and not self._running)
        self.btn_skip.setEnabled(ok and not self._running)
        self.tbl_fix.setEnabled(ok and not self._running)

    def _fix_save(self) -> None:
        edits = {self.tbl_fix.item(i, 0).text(): self.tbl_fix.item(i, 1).text() for i in range(self.tbl_fix.rowCount())}
        edits = {c: v for c, v in edits.items() if v != self._fix_old.get(c)}
        if not edits:
            self.statusBar().showMessage("바뀐 칸이 없습니다", 4000)
            return
        res = self.editor.line_fix_row(self._fix_key, edits)
        if not res.get("ok"):
            self._refuse(res.get("reason", "저장하지 못했습니다"))
            self._set_detail(str(res.get("detail") or ""))
            return
        self.lbl_refuse.setText("")
        self.statusBar().showMessage(f"{self._fix_key}: {', '.join(res['changed'])} 칸을 고쳤습니다 (원본은 .bak.csv)", 6000)
        self.reload()

    def _fix_exclude(self, on: bool, confirm: bool = True) -> None:
        if confirm:
            ans = QtWidgets.QMessageBox.question(
                self, "이 건 제외", f"건 {self._fix_key} 을(를) 학습·검증에서 뺍니다. 표에서 지우지는 않습니다. 계속할까요?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if ans != QtWidgets.QMessageBox.Yes:
                return
        res = self.editor.line_exclude_row(self._fix_key, on)
        if not res.get("ok"):
            self._refuse(res.get("reason", "저장하지 못했습니다"))
            self._set_detail(str(res.get("detail") or ""))
            return
        self.lbl_refuse.setText("")
        self.reload()

    # ── 저장·안쪽·내보내기 ──────────────────────────────────────────────
    def _save(self) -> None:
        res = self.editor.line_save()
        if res.get("ok"):
            self.statusBar().showMessage("저장했습니다", 4000)
        else:
            self._refuse(res.get("reason", "저장하지 못했습니다"))
            self._set_detail(str(res.get("detail") or ""))
        self._title()

    def _open_data(self) -> None:
        from .data_dialog import DataDialog

        dlg = DataDialog(self.editor, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.desc = self.editor.line_describe()
            self.lbl_refuse.setText("")
            self._title()
            self._load_thumbs()
            self.refresh()
            self._show_settings(self.picked)
            self.statusBar().showMessage("데이터 설정을 적용했습니다 — 저장하면 파일에 남습니다", 6000)

    def _new_line(self) -> None:
        from ...spec import scaffold

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "새 프로젝트를 만들 폴더(없으면 만들어 두세요)")
        if not folder:
            return
        try:
            path, _made = scaffold.new_line(folder)
        except Exception as exc:
            self._refuse(str(exc).splitlines()[0])
            self._set_detail(str(exc))
            return
        ed = Editor.open_line(path)
        ed.extra_modules = self.editor.extra_modules
        win = LineWindow(ed)
        self._others = getattr(self, "_others", []) + [win]  # 닫히지 않게 붙들어 둔다
        win.show()
        QtCore.QTimer.singleShot(0, win._open_data)

    def _open_verdict(self) -> None:
        from .verdict_dialog import VerdictDialog

        VerdictDialog(self.thumbs, getattr(self.editor, "run_id", "") or "", self).exec()

    def _open_inner(self, cid: str) -> None:
        st = self._station(cid or "")
        if st is None:
            self.statusBar().showMessage("스테이션을 먼저 고르세요", 3000)
            return
        dlg = InnerView(self.editor.compiled, st["members"], f"{M.mark(st['number'])} {st['label']}", self)
        dlg.exec()

    def _export(self) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("11노드로 내보내기")
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setText("11노드 프로젝트로 내보냅니다.")
        box.setInformativeText(
            "내보낸 파일은 이 5노드 파일과 연결이 끊깁니다. 한 번 내보내면 5노드 화면으로 돌아올 수 없고, "
            "이후에는 내보낸 파일을 직접 편집해야 합니다.\n상자와 배선을 직접 고칠 때만 쓰세요.\n\n"
            f"만들 파일: {os.path.basename(self.editor.line_export_path())}")
        go = box.addButton("내보내기", QtWidgets.QMessageBox.DestructiveRole)
        box.addButton("취소", QtWidgets.QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is not go:
            return
        res = self.editor.line_export()
        if res.get("ok"):
            self.statusBar().showMessage(f"내보냈습니다: {res['path']}", 8000)
        else:
            self._refuse(res.get("reason", "내보내지 못했습니다"))
            self._set_detail(str(res.get("detail") or ""))

    # ── 실행 ────────────────────────────────────────────────────────────
    def _run(self, resume: Optional[bool] = None) -> None:
        res = self.editor.pipeline_start(resume, 0.0)
        ask = res.get("ask_resume")
        if ask:
            self._ask_resume(ask)
            return
        if not res.get("ok"):
            self._refuse(str(res.get("reason") or "실행하지 못했습니다"))
            if res.get("detail"):
                self._set_detail(str(res["detail"]))
            return
        self.lbl_refuse.setText("")
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

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self.editor.dirty:
            ans = QtWidgets.QMessageBox.question(
                self, "저장하지 않은 변경", "저장하지 않은 변경이 있습니다. 저장할까요?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
            if ans == QtWidgets.QMessageBox.Cancel:
                ev.ignore()
                return
            if ans == QtWidgets.QMessageBox.Save:
                self._save()
                if self.editor.dirty:
                    ev.ignore()
                    return
        ev.accept()


def space_path(root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(root, rel))


def launch_line(path: str, extra_modules: tuple = ()) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("VLM Trainer")
    editor = Editor.open_line(path)
    editor.extra_modules = tuple(extra_modules)
    win = LineWindow(editor)
    win.show()
    if editor.compiled is None:  # 새 프로젝트: 표부터 넣게 한다
        QtCore.QTimer.singleShot(0, win._open_data)
    return app.exec()
