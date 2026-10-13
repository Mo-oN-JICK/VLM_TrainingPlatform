"""검증 결과 대화상자 — 모델이 낸 답을 정답과 나란히 놓고 사람이 OK/NG 를 정한다.

기계의 O/X 는 항목별 보조 표시이고, 점수 합계는 내지 않는다. 판정은 건 단위이며 누를 때마다
`judgments.jsonl` 에 한 줄씩 덧붙는다(`vlmt judge` 와 같은 함수). 다시 열어도 유지된다.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from PySide6 import QtCore, QtGui, QtWidgets

from .. import tokens as T
from ...answer import judge
from .window import QSS

GREEN, RED = T.STATE["success"], T.STATE["failed"]


def list_runs() -> List[str]:
    """검증까지 끝난 실행 id, 최근 것이 위."""
    root = "runs"
    if not os.path.isdir(root):
        return []
    found = [(os.path.getmtime(os.path.join(root, d, "infer", "answers.jsonl")), d) for d in os.listdir(root)
             if os.path.exists(os.path.join(root, d, "infer", "answers.jsonl"))]
    return [d for _, d in sorted(found, reverse=True)]


class VerdictDialog(QtWidgets.QDialog):
    def __init__(self, thumbs: Dict[str, str], current: str = "", parent: Any = None) -> None:
        super().__init__(parent)
        self.thumbs = thumbs
        self.setWindowTitle("검증 결과 — 모델 답을 보고 OK / NG 를 정합니다")
        self.resize(980, 720)
        self.setStyleSheet(QSS)
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("실행"))
        self.cb = QtWidgets.QComboBox()
        runs = list_runs()
        self.cb.addItems(runs)
        if current in runs:
            self.cb.setCurrentText(current)
        self.cb.currentTextChanged.connect(self._fill)
        top.addWidget(self.cb)
        top.addStretch(1)
        self.lbl_sum = QtWidgets.QLabel("")
        top.addWidget(self.lbl_sum)
        lay.addLayout(top)
        self.lbl_err = QtWidgets.QLabel("")
        self.lbl_err.setWordWrap(True)
        self.lbl_err.setStyleSheet(f"color:{RED};")
        lay.addWidget(self.lbl_err)
        self.tbl = QtWidgets.QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["사진", "건", "항목별 (모델 답 / 정답)", "판정"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        h = self.tbl.horizontalHeader()
        h.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        lay.addWidget(self.tbl, 1)
        note = QtWidgets.QLabel("O/X 는 참고용입니다. 최종 판정은 사람이 건마다 OK / NG 로 내립니다. 점수 합계는 내지 않습니다.")
        note.setStyleSheet(f"color:{T.NODE['muted']}; font-size:11px;")
        lay.addWidget(note)
        self._fill(self.cb.currentText())

    def _fill(self, run: str) -> None:
        self.tbl.setRowCount(0)
        self.lbl_err.setText("")
        if not run:
            self.lbl_err.setText("검증까지 끝난 실행이 없습니다. 실행을 마친 뒤 다시 여세요.")
            self.lbl_sum.setText("")
            return
        self.run_dir = os.path.join("runs", run)
        try:
            got = judge.review(self.run_dir)
        except Exception as exc:
            self.lbl_err.setText(str(exc))
            return
        self._steps = got["steps"]
        for it in got["items"]:
            self._add(it)
        self._sum()

    def _add(self, it: Dict[str, Any]) -> None:
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        self.tbl.setRowHeight(r, 78)
        lb = QtWidgets.QLabel()
        pm = QtGui.QPixmap(self.thumbs.get(it["id"], ""))
        if not pm.isNull():
            lb.setPixmap(pm.scaled(72, 72, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            lb.setText("사진 없음")
        self.tbl.setCellWidget(r, 0, lb)
        self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(it["id"]))
        if it["empty"]:
            html = f"<span style='color:{RED}'>모델이 답하지 않았습니다 (빈 답)</span>"
        else:
            rows = []
            for f in it["fields"]:
                mark = f"<b style='color:{GREEN}'>O 일치</b>" if f["match"] else f"<b style='color:{RED}'>X 다름</b>"
                far = f"  <b style='color:{RED}'>범위 밖</b>" if f["out_of_range"] else ""
                rows.append(f"{f['step']}: {f['answer'] if f['answer'] is not None else '(없음)'} / {f['expected']}  {mark}{far}")
            html = "<br>".join(rows)
            if it["broken"]:
                html += f"<br><span style='color:{RED}'>형식 어긋남: {', '.join(it['broken'])} 항목이 답에 없습니다</span>"
        txt = QtWidgets.QLabel(html)
        txt.setTextFormat(QtCore.Qt.RichText)
        self.tbl.setCellWidget(r, 2, txt)
        box = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(box)
        hl.setContentsMargins(4, 4, 4, 4)
        btns = {}
        for v, name in (("ok", "OK"), ("ng", "NG")):
            b = QtWidgets.QPushButton(name)
            b.setCheckable(True)
            b.setStyleSheet(f"QPushButton:checked {{ background:{GREEN if v == 'ok' else RED}; color:#111; font-weight:bold; }}")
            b.setChecked(it["verdict"] == v)
            b.clicked.connect(lambda _c, i=it["id"], vv=v, bs=btns: self._set(i, vv, bs))
            hl.addWidget(b)
            btns[v] = b
        self.tbl.setCellWidget(r, 3, box)

    def _set(self, sample_id: str, verdict: str, btns: Dict[str, Any]) -> None:
        """이미 켜진 버튼을 다시 누르면 판정을 지운다(미판정)."""
        now = btns[verdict].isChecked()
        v = verdict if now else "none"
        try:
            judge.judge(self.run_dir, sample_id, v)
        except Exception as exc:
            self.lbl_err.setText(str(exc))
            btns[verdict].setChecked(not now)
            return
        for k, b in btns.items():
            b.setChecked(k == v)
        self._sum()

    def _sum(self) -> None:
        c = judge.review(self.run_dir)["count"]
        self.lbl_sum.setText(f"OK {c['ok']} · NG {c['ng']} · 미판정 {c['none']}")
