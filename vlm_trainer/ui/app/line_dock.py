"""5노드 NG 표 — 화면 아래에 항상 보이는 배출함.

NG 건을 사진과 함께 보이고, 한 건을 누르면 한 줄 요약(몇 번 스테이션인지 포함)과 상세, 그리고 그 건을
표(CSV)에서 고치거나 제외하는 칸이 열린다. 고치는 일은 전부 `Editor.line_*` 가 한다 — 여기는 그 호출을
위젯에 연결할 뿐이다.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from .. import line_model as M
from .. import tokens as T
from ...engine import samples as samples_mod

MUTED = f"color:{T.NODE['muted']}; font-size:11px;"


def space_path(root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(root, rel))


def load_thumbs(editor: Any) -> Tuple[Dict[str, str], int]:
    """(건 id → 사진 경로, 표의 `제외` 열 때문에 건너뛴 건수). 표를 읽지 못하면 사진 없이 간다."""
    cg = editor.compiled
    if cg is None:
        return {}, 0
    try:
        space = samples_mod.load(cg.sample_space, os.path.dirname(editor.path))
    except Exception:
        return {}, 0
    col = next((n.params.get("column", "image_path") for n in cg.nodes.values()
                if n.ref.startswith("source.image")), "image_path")
    return {str(r[space.key]): space_path(space.root, str(r.get(col, ""))) for r in space.rows}, space.excluded


def _short_line(text: str) -> str:
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    return lines[0] if lines else ""


class NgDock(QtWidgets.QDockWidget):
    reread = QtCore.Signal()               # 표를 고쳤으니 다시 읽어 달라
    refused = QtCore.Signal(str, dict)     # (무엇을 하려다, 거부 결과)

    def __init__(self, parent: Any = None) -> None:
        super().__init__("NG", parent)
        self.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.editor: Any = None
        self.thumbs: Dict[str, str] = {}
        self._rows: List[M.NgRow] = []
        self._excluded = 0
        self._running = False
        self._fix_key = ""
        self._fix_old: Dict[str, str] = {}

        body = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(body)
        bl.setContentsMargins(8, 6, 8, 8)
        self.lbl_ng = QtWidgets.QLabel("NG 0건")
        self.lbl_ng.setStyleSheet("font-weight:600;")
        bl.addWidget(self.lbl_ng)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bl.addWidget(split, 1)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.tbl_ng = QtWidgets.QTableWidget(0, 4)
        self.tbl_ng.setHorizontalHeaderLabels(["사진", "건", "스테이션", "원인"])
        self.tbl_ng.horizontalHeader().setStretchLastSection(True)
        self.tbl_ng.verticalHeader().setVisible(False)
        self.tbl_ng.verticalHeader().setDefaultSectionSize(48)
        self.tbl_ng.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_ng.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_ng.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_ng.itemSelectionChanged.connect(self._show_detail)
        ll.addWidget(self.tbl_ng, 1)
        # 오류는 한 줄 요약(몇 번 스테이션인지 포함)이 먼저, 네 가지 상세는 접혀 있다.
        self.lbl_sum = QtWidgets.QLabel("")
        self.lbl_sum.setWordWrap(True)
        self.lbl_sum.setStyleSheet(f"color:{T.STATE['failed']};")
        ll.addWidget(self.lbl_sum)
        self.btn_more = QtWidgets.QToolButton()
        self.btn_more.setText("상세 보기")
        self.btn_more.setCheckable(True)
        self.btn_more.setVisible(False)
        self.btn_more.toggled.connect(
            lambda on: (self.detail.setVisible(on), self.btn_more.setText("상세 접기" if on else "상세 보기")))
        ll.addWidget(self.btn_more)
        self.detail = QtWidgets.QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMaximumHeight(120)
        self.detail.setVisible(False)
        ll.addWidget(self.detail)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.lbl_fix = QtWidgets.QLabel("고치려면 NG 한 건을 누르세요. 표(CSV)를 직접 고쳐도 됩니다 — 고친 뒤 `다시 읽기`.")
        self.lbl_fix.setWordWrap(True)
        self.lbl_fix.setStyleSheet(MUTED)
        rl.addWidget(self.lbl_fix)
        self.fix_box = QtWidgets.QGroupBox("이 건 고치기")
        fl = QtWidgets.QVBoxLayout(self.fix_box)
        self.tbl_fix = QtWidgets.QTableWidget(0, 2)
        self.tbl_fix.setHorizontalHeaderLabels(["열", "값"])
        self.tbl_fix.horizontalHeader().setStretchLastSection(True)
        self.tbl_fix.verticalHeader().setVisible(False)
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
        rl.addWidget(self.fix_box, 1)
        split.addWidget(right)
        split.setSizes([620, 420])
        self.setWidget(body)

    # ── 채우기 ──────────────────────────────────────────────────────────
    def bind(self, editor: Any) -> None:
        """편집기를 (다시) 물린다. 표가 바뀌었을 수 있으니 사진도 새로 읽고, 열려 있던 고치기 칸도 새 값으로."""
        self.editor = editor
        self.thumbs, self._excluded = load_thumbs(editor)
        if self._fix_key:
            self._show_fix(self._fix_key)

    def paint(self, rows: List[M.NgRow], total: int) -> None:
        self._rows = rows
        self.lbl_ng.setText(f"NG {total}건" + (f" (표시 {len(rows)}건)" if total > len(rows) else "")
                            + (f" · 제외 {self._excluded}건" if self._excluded else ""))
        self.lbl_ng.setToolTip("제외한 건은 표의 `제외` 열에 x 가 적힌 행입니다. 되살리려면 x 를 지우고 `다시 읽기`.")
        keep = self.tbl_ng.currentRow()
        self.tbl_ng.blockSignals(True)          # 다시 그리는 것은 사람이 고른 것이 아니다
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
        self.tbl_ng.blockSignals(False)

    def set_running(self, on: bool) -> None:
        """돌고 있는 동안은 표를 못 고친다 — 실행은 디스크의 표를 읽는 중이다."""
        self._running = on
        if self._fix_key:
            self._show_fix(self._fix_key)

    def set_detail(self, text: str, where: str = "") -> None:
        text = str(text or "")
        first = _short_line(text)
        self.lbl_sum.setText(f"{where} — {first}" if where and first else first)
        self.detail.setPlainText(text)
        self.btn_more.setVisible(bool(text.strip()))
        if not text.strip():
            self.btn_more.setChecked(False)

    def _show_detail(self) -> None:
        i = self.tbl_ng.currentRow()
        if not (0 <= i < len(self._rows)):
            return
        r = self._rows[i]
        where = f"{M.mark(r.card_number)} {r.card_label}" if r.card_number else r.node_id
        nn = chr(10) * 2
        self.set_detail(r.cause + f"{nn}건 {r.sample_key} · {r.node_id}" + (f"{nn}해결: {r.hint}" if r.hint else ""), where)
        self._show_fix(r.sample_key)

    # ── NG 한 건 고치기 ─────────────────────────────────────────────────
    def _show_fix(self, key: str) -> None:
        """표의 그 행을 칸마다 보여 준다. CSV 가 아니면 이유만 보이고 고칠 수 없다."""
        self._fix_key, self._fix_old = key, {}
        self.fix_box.setVisible(bool(key))
        if not key or self.editor is None:
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
        can = ok and not self._running
        self.btn_fix.setEnabled(can)
        self.btn_skip.setEnabled(can)
        self.tbl_fix.setEnabled(can)

    def _fix_save(self) -> None:
        edits = {self.tbl_fix.item(i, 0).text(): self.tbl_fix.item(i, 1).text() for i in range(self.tbl_fix.rowCount())}
        edits = {c: v for c, v in edits.items() if v != self._fix_old.get(c)}
        if not edits:
            self.lbl_fixmsg.setText("바뀐 칸이 없습니다.")
            return
        res = self.editor.line_fix_row(self._fix_key, edits)
        if not res.get("ok"):
            self.refused.emit("이 건 고치기", res)
            return
        self.lbl_fixmsg.setText(f"{', '.join(res['changed'])} 칸을 고쳤습니다 (원본은 .bak.csv)")
        self.reread.emit()

    def _fix_exclude(self, on: bool, confirm: bool = True) -> None:
        if confirm:
            ans = QtWidgets.QMessageBox.question(
                self, "이 건 제외", f"건 {self._fix_key} 을(를) 학습·검증에서 뺍니다. 표에서 지우지는 않습니다. 계속할까요?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if ans != QtWidgets.QMessageBox.Yes:
                return
        res = self.editor.line_exclude_row(self._fix_key, on)
        if not res.get("ok"):
            self.refused.emit("이 건 제외", res)
            return
        self.reread.emit()
