"""5노드 설정 칸 — 스테이션 하나의 설정 값을 표로 보이고 고친다.

`LineWindow` 와 기존 편집기 화면(`EditorWindow`)의 설정 칸이 같은 것을 쓴다. 값을 고치는 일은 전부
`Editor.line_set*` 이 하고, 여기는 그 호출을 위젯에 연결할 뿐이다.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from .. import line_model as M


_BAD = object()   # JSON 이 아닌 입력. 그대로 보내지 않고 거부로 돌린다


def make_editor(value: Any, choices: List[str], on_done: Callable[[Any], None]) -> QtWidgets.QWidget:
    """값의 종류에 맞는 입력 위젯. 값이 바뀌어 **입력이 끝났을 때만** 알린다 — 글자마다 알리면
    타이핑하는 동안 컴파일이 돌고 중간 값이 전부 거부된다."""
    if choices:
        w = QtWidgets.QComboBox()
        w.addItems([str(c) for c in choices])
        if str(value) not in [str(c) for c in choices]:
            w.insertItem(0, str(value))
        w.setCurrentText(str(value))
        w.activated.connect(lambda _i: on_done(w.currentText()))
        return w
    if isinstance(value, bool):
        w = QtWidgets.QCheckBox()
        w.setChecked(value)
        w.clicked.connect(lambda on: on_done(bool(on)))
        return w
    if isinstance(value, int):
        w = QtWidgets.QSpinBox()
        w.setRange(-10**9, 10**9)
        w.setValue(value)
        w.editingFinished.connect(lambda: on_done(w.value()))
        return w
    if isinstance(value, float):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(-1e9, 1e9)
        w.setDecimals(6)
        w.setValue(value)
        w.editingFinished.connect(lambda: on_done(w.value()))
        return w
    w = QtWidgets.QLineEdit()
    if isinstance(value, str):
        w.setText(value)
        w.editingFinished.connect(lambda: on_done(w.text()))
    else:  # 목록·표·비어 있음 — JSON 으로 적는다
        w.setText(json.dumps(value, ensure_ascii=False))
        w.setToolTip("JSON 으로 적습니다. 예: [448, 448]")

        def done() -> None:
            try:
                on_done(json.loads(w.text()))
            except ValueError:
                on_done(_BAD)

        w.editingFinished.connect(done)
    return w

def settings_rows(editor: Any, st: Dict[str, Any], expert: bool) -> List[tuple]:
    """(종류, 이름, 값, 선택지, 적용 함수). 기본 모드는 기본 항목만, 전문가는 전부와 안쪽 상자 값까지."""
    cid = st["id"]
    rows: List[tuple] = []
    if st.get("amount") and (expert or st["amount"]["level"] == "basic"):
        a = st["amount"]
        rows.append(("amount", a["label"] + ("" if a["value"] else " (직접 입력 중)"),
                     a["value"] or "직접 입력", a["choices"],
                     lambda v, sid=cid: editor.line_set_amount(sid, v)))
    for name, s in st["settings"].items():
        if s["level"] != "basic" and not expert:
            continue
        rows.append(("setting", s["label"] + ("" if s["level"] == "basic" else "  · 전문가"),
                     s["value"], s["choices"], lambda v, sid=cid, n=name: editor.line_set(sid, n, v)))
    if expert:
        for b in M.boxes(st, editor.compiled, os.path.dirname(editor.path)):
            rows.append(("box", f"{b.label}  ({b.id})", None, [], None))
            for pname, pval in b.params:
                rows.append(("param", pname, pval, [],
                             lambda v, nid=b.id, pn=pname: editor.line_set_override(nid, pn, v)))
    return rows


def new_table() -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, 2)
    t.setHorizontalHeaderLabels(["설정", "값"])
    t.horizontalHeader().setStretchLastSection(True)
    t.setColumnWidth(0, 200)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    t.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    return t


def fill_table(t: QtWidgets.QTableWidget, rows: List[tuple],
               wrap: Callable[[Callable[[Any], Dict[str, Any]], Any], Callable[[Any], None]]) -> None:
    """`wrap(적용 함수, 지금 값)` 이 입력이 끝났을 때 불릴 함수를 돌려준다."""
    # 칸 위젯은 setRowCount 로 비워야 사라진다. 그냥 덮으면 지난 그림의 입력칸이 남는다.
    t.clearSpans()
    t.setRowCount(0)
    t.setRowCount(len(rows))
    for i, (kind, label, value, choices, setter) in enumerate(rows):
        if kind == "box":
            item = QtWidgets.QTableWidgetItem(label)
            f = item.font()
            f.setBold(True)
            item.setFont(f)
            t.setItem(i, 0, item)
            t.setSpan(i, 0, 1, 2)
            t.removeCellWidget(i, 1)
            continue
        t.setItem(i, 0, QtWidgets.QTableWidgetItem(label))
        t.setCellWidget(i, 1, make_editor(value, choices, wrap(setter, value)))
