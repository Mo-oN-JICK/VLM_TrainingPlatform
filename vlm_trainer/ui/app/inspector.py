"""노드 설정 패널 — 고른 상자의 값을 고친다.

값이 바뀌는 순간 `api.set_param()` 으로 보낸다. 거부되면 원래 값으로 되돌리고 이유를
보여 준다 — 화면에만 남은 값은 다음에 여는 사람을 속인다.

다만 **글자를 치는 칸은 포커스가 떠날 때** 보낸다. 한 글자마다 컴파일을 돌리면
"44" 를 치는 중간의 "4" 에서 거부 창이 뜬다.

접힌 Procedure 는 **노출 파라미터만**, 그리고 **상자 주소**로 보낸다. 안쪽 노드
(`p_prep/n_resize`)는 프로시저 파일에 있어 프로젝트가 건드릴 수 있는 것이 아니다.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from .. import tokens as T
from ..api import param_meta

MUTED = "color:#6F7478; font-size:11px;"


class _Mark(QtWidgets.QLabel):
    """파라미터 옆 표식. 왜 이 값이 특별한지 한 글자로 말한다."""

    def __init__(self, text: str, color: str, tip: str) -> None:
        super().__init__(text)
        self.setToolTip(tip)
        self.setStyleSheet(
            f"color:{color}; border:1px solid {color}; border-radius:2px;"
            "padding:0 3px; font-size:9px;")


class Inspector(QtWidgets.QDockWidget):
    paramChanged = QtCore.Signal(str, str, object)   # 상자 id, 파라미터, 값
    boundaryToggled = QtCore.Signal(str, bool)
    expandToggled = QtCore.Signal(str)
    removeRequested = QtCore.Signal(str)
    focusRequested = QtCore.Signal(str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__("노드 설정", parent)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setWidget(self._scroll)
        self._editor: Any = None
        self.show_empty()

    def bind(self, editor: Any) -> None:
        self._editor = editor

    # ── 비어 있을 때 ────────────────────────────────────────────────────
    def show_empty(self) -> None:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        msg = QtWidgets.QLabel("가운데 그림에서 상자를 하나 누르면\n그 상자의 설정이 여기에 나옵니다.")
        msg.setWordWrap(True)
        msg.setStyleSheet(MUTED)
        lay.addWidget(msg)
        lay.addStretch(1)
        self._scroll.setWidget(w)

    # ── 상자 하나 ───────────────────────────────────────────────────────
    def show_node(self, nid: str, shown: Any, cg: Any, state: Dict[str, Any]) -> None:
        """`shown` 은 캔버스가 그린 것과 같은 상자다 — 접혔으면 Procedure 하나."""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        lay.addWidget(self._header(nid, shown))

        rows = (self._proc_rows(nid, shown, cg) if shown.inner
                else self._node_rows(nid, cg, state))
        if rows:
            lay.addWidget(rows)
        else:
            empty = QtWidgets.QLabel("고칠 값이 없는 노드입니다.")
            empty.setStyleSheet(MUTED)
            lay.addWidget(empty)

        lay.addWidget(self._wiring(nid, shown, cg))
        lay.addWidget(self._actions(nid, shown, state))
        lay.addStretch(1)
        self._scroll.setWidget(w)

    def _header(self, nid: str, shown: Any) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        title = QtWidgets.QLabel(shown.label or nid)
        title.setStyleSheet("color:#FFFFFF; font-size:14px; font-weight:600;")
        title.setWordWrap(True)
        sub = QtWidgets.QLabel(f"{nid}   ·   {shown.ref}")
        sub.setStyleSheet(MUTED)
        sub.setWordWrap(True)
        lay.addWidget(title)
        lay.addWidget(sub)
        if shown.hint:
            hint = QtWidgets.QLabel(shown.hint)
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#A8B0B6; font-size:11px; padding-top:3px;")
            lay.addWidget(hint)
        return box

    # ── 파라미터 ────────────────────────────────────────────────────────
    def _node_rows(self, nid: str, cg: Any, state: Dict[str, Any]) -> Optional[QtWidgets.QWidget]:
        n = cg.nodes.get(nid)
        if n is None:
            return None
        overlaid = set(state.get("overlaid") or ())
        form = _Form("파라미터")
        for m in param_meta(n.ref, n.params):
            marks = []
            if f"{nid}:{m['name']}" in overlaid:
                marks.append(_Mark("레시피", T.STATE["cached"],
                                   "레시피가 덮고 있다 — 고치면 오버레이가 바뀌고 프로젝트 스펙은 그대로다"))
            elif m["overridable"]:
                marks.append(_Mark("recipe", "#6F7478", "Parameter Recipe 가 덮을 수 있다"))
            if m["type_affecting"]:
                marks.append(_Mark("type", T.NODE["accent"], "바꾸면 배선 타입이 다시 검사된다"))
            form.add(m["name"], self._widget(nid, m), marks)
        return form if form.count else None

    def _proc_rows(self, pid: str, shown: Any, cg: Any) -> Optional[QtWidgets.QWidget]:
        """접힌 Procedure. 노출 파라미터만 보이고, 주소는 상자다."""
        proc = next((p for p in cg.procedures if p["id"] == pid), None)
        if proc is None:
            return None
        form = _Form("노출 파라미터")
        for name, target in (proc.get("exposed_params") or {}).items():
            inner_id, inner_param = target
            inner = cg.nodes.get(inner_id)
            if inner is None:
                continue
            value = inner.params.get(inner_param)
            meta = {"name": name, "value": value,
                    "kind": _kind_of(value), "overridable": False, "type_affecting": False}
            tip = _Mark(inner_id.split("/")[-1], "#6F7478",
                        f"안쪽 {inner_id}.{inner_param} 로 간다")
            form.add(name, self._widget(pid, meta), [tip])
        note = QtWidgets.QLabel(
            f"안에 노드 {shown.inner}개가 들어 있습니다. "
            "상자를 우클릭해 펼치면 안쪽이 보입니다.")
        note.setWordWrap(True)
        note.setStyleSheet(MUTED)
        form.layout().addWidget(note)
        return form if form.count else None

    def _widget(self, nid: str, m: Dict[str, Any]) -> QtWidgets.QWidget:
        name, kind, value = m["name"], m["kind"], m["value"]
        send = lambda v, p=name: self.paramChanged.emit(nid, p, v)  # noqa: E731

        if kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(send)                 # 즉시
            return w

        if kind == "number":
            if isinstance(value, bool) or isinstance(value, int):
                w = QtWidgets.QSpinBox()
                w.setRange(-10_000_000, 10_000_000)
                w.setValue(int(value))
            else:
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(6)
                w.setRange(-1e9, 1e9)
                w.setValue(float(value))
            w.setKeyboardTracking(False)            # 타이핑 도중에는 보내지 않는다
            w.editingFinished.connect(lambda ww=w: send(ww.value()))
            return w

        if kind == "json":
            w = _JsonEdit(json.dumps(value, ensure_ascii=False))
            w.committed.connect(send)               # 포커스 아웃 또는 Enter
            return w

        w = _TextEdit(str(value))
        w.committed.connect(send)
        return w

    # ── 연결과 조작 ─────────────────────────────────────────────────────
    def _wiring(self, nid: str, shown: Any, cg: Any) -> QtWidgets.QWidget:
        grp = _Form("연결")
        n = cg.nodes.get(nid)
        wired = n.inputs if n is not None else {}
        for port in sorted(shown.inputs):
            src = wired.get(port, "")
            lbl = QtWidgets.QLabel(src or "아직 비어 있음")
            lbl.setStyleSheet("color:#A8B0B6;" if src else "color:#5F6468;")
            lbl.setWordWrap(True)
            grp.add("← " + port, lbl, [])
        for port in sorted(shown.outputs):
            outs = [e.dst for e in cg.edges if e.src == f"{nid}:{port}"]
            lbl = QtWidgets.QLabel(", ".join(outs) if outs else "아직 아무 데도 안 감")
            lbl.setStyleSheet("color:#A8B0B6;" if outs else "color:#5F6468;")
            lbl.setWordWrap(True)
            grp.add("→ " + port, lbl, [])
        return grp

    def _actions(self, nid: str, shown: Any, state: Dict[str, Any]) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("이 상자 다루기")
        lay = QtWidgets.QVBoxLayout(box)
        lay.setSpacing(5)

        chk = QtWidgets.QCheckBox("물질화 경계")
        chk.setToolTip("여기까지 미리 굽고, 학습은 그 산출물만 읽는다")
        bset = set(state.get("boundary") or ())
        chk.setChecked(nid in bset or any(i in bset for i in shown.state_ids))
        chk.toggled.connect(lambda on: self.boundaryToggled.emit(nid, on))
        lay.addWidget(chk)

        row = QtWidgets.QHBoxLayout()
        if shown.inner:
            b = QtWidgets.QPushButton("펼치기 / 접기")
            b.clicked.connect(lambda: self.expandToggled.emit(nid))
            row.addWidget(b)
        look = QtWidgets.QPushButton("캔버스에서 보기")
        look.clicked.connect(lambda: self.focusRequested.emit(nid))
        row.addWidget(look)
        rm = QtWidgets.QPushButton("삭제")
        rm.clicked.connect(lambda: self.removeRequested.emit(nid))
        row.addWidget(rm)
        lay.addLayout(row)
        return box


def _kind_of(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, (list, tuple, dict)):
        return "json"
    return "text"


class _Form(QtWidgets.QGroupBox):
    """이름 + 표식 + 입력칸 한 줄씩. 이름 폭을 맞춰 두어야 눈이 세로로 흐른다."""

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._lay = QtWidgets.QVBoxLayout(self)
        self._lay.setSpacing(6)
        self.count = 0

    def add(self, name: str, widget: QtWidgets.QWidget, marks: List[QtWidgets.QWidget]) -> None:
        row = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(4)
        lbl = QtWidgets.QLabel(name)
        lbl.setStyleSheet("color:#D8DCDF; font-size:11px;")
        head.addWidget(lbl)
        for m in marks:
            head.addWidget(m)
        head.addStretch(1)
        lay.addLayout(head)
        lay.addWidget(widget)

        self._lay.addWidget(row)
        self.count += 1


class _TextEdit(QtWidgets.QLineEdit):
    """포커스가 떠날 때, 또는 Enter 에 보낸다. 값이 그대로면 보내지 않는다."""

    committed = QtCore.Signal(object)

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._last = text
        self.editingFinished.connect(self._commit)

    def _commit(self) -> None:
        if self.text() != self._last:
            self._last = self.text()
            self.committed.emit(self.text())

    def revert(self, text: str) -> None:
        self._last = text
        self.setText(text)


class _JsonEdit(_TextEdit):
    """리스트·사전 값. 사람이 치는 중간은 JSON 이 아닌 게 정상이라,
    떠날 때 한 번만 따져 보고 아니면 그 자리에서 말한다."""

    def _commit(self) -> None:
        if self.text() == self._last:
            return
        try:
            value = json.loads(self.text())
        except json.JSONDecodeError:
            QtWidgets.QToolTip.showText(
                self.mapToGlobal(QtCore.QPoint(0, self.height())),
                "JSON 이 아니다. 예: [448, 448]  또는  {\"a\": 1}")
            self.setText(self._last)
            return
        self._last = self.text()
        self.committed.emit(value)
