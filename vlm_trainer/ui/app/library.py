"""노드 라이브러리 도크 — 끌어다 캔버스에 놓는다.

목록은 `api.library()`가 준다. 이 파일은 그것을 분류별로 묶어 보여 주고,
드래그가 시작되면 노드 타입 문자열 하나를 실어 보낼 뿐이다.
"""

from __future__ import annotations

from typing import Any, Dict, List

from PySide6 import QtCore, QtGui, QtWidgets

from .. import tokens as T
from .canvas import MIME

KIND_MARK = {"input": "I", "processing": "P", "output": "O"}


def _node_tip(n: Dict[str, Any]) -> str:
    """호버하면 뜨는 글. **무엇을 하는 상자이고, 무엇을 꽂아야 하는가.**

    전에는 타입 이름과 포트 **이름만** 보여 줬다(`입력: image, regions`). 이름만으로는
    무엇을 꽂아야 하는지 알 수 없다 — 그래서 타입 설명을 함께 싣는다.
    """
    help_ = n.get("port_help") or {}
    kind = KIND_MARK.get(n["kind"], "")
    head = (n.get("label") or n["type"]) + (f"  <small>{kind}</small>" if kind else "")
    rows = [f"<b style='font-size:13px'>{head}</b>"]
    if n.get("hint"):
        rows.append(n["hint"])
    if n.get("scenario"):
        rows.append(f"<br><i>{n['scenario']}</i>")

    for side, names, title in (("in", n["inputs"], "받는 것"),
                               ("out", n["outputs"], "내보내는 것")):
        if not names:
            continue
        rows.append(f"<br><b>{title}</b>")
        for name in names:
            tip = help_.get(f"{side}:{name}", "")
            # 포트 설명의 첫 줄(= 사람 말로 된 종류)만 꺼내 온다. 전문은 캔버스에서
            # 그 포트에 직접 호버하면 나온다.
            word = tip.split("</b>")[0].split("<b>")[-1] if tip else ""
            rows.append(f"· {name}" + (f" — {word}" if word else ""))

    rows.append(f"<br><span style='color:#6F7478'>{n['type']}</span>")
    return "<div style='max-width:340px'>" + "<br>".join(rows) + "</div>"


class LibraryTree(QtWidgets.QTreeWidget):
    """분류 트리. 드래그를 시작하면 `application/x-vlmt-node` 에 타입을 싣는다."""

    def __init__(self, nodes: List[Dict[str, Any]], parent: Any = None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.setIndentation(12)
        self.setAlternatingRowColors(False)
        self._fill(nodes)

    def _fill(self, nodes: List[Dict[str, Any]]) -> None:
        by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for n in nodes:
            by_cat.setdefault(n["category"], []).append(n)
        for cat in sorted(by_cat):
            top = QtWidgets.QTreeWidgetItem(self, [f"{cat}   {len(by_cat[cat])}"])
            top.setFlags(QtCore.Qt.ItemIsEnabled)
            top.setForeground(0, QtGui.QBrush(QtGui.QColor("#C6CCD1")))
            dot = QtGui.QPixmap(8, 8)
            dot.fill(QtGui.QColor(T.category_color(cat)))
            top.setIcon(0, QtGui.QIcon(dot))
            for n in sorted(by_cat[cat], key=lambda d: d.get("label") or d["type"]):
                # **한국어 이름을 앞에 둔다.** 타입 이름(`source.field`)은 처음 보는
                # 사람에게 아무 뜻도 없고, 정렬 기준으로도 쓸모가 없다.
                it = QtWidgets.QTreeWidgetItem(top, [n.get("label") or n["type"]])
                it.setData(0, QtCore.Qt.UserRole, n["type"])
                it.setToolTip(0, _node_tip(n))
            top.setExpanded(False)

    def startDrag(self, actions: Any) -> None:
        it = self.currentItem()
        ref = it.data(0, QtCore.Qt.UserRole) if it is not None else None
        if not ref:
            return
        mime = QtCore.QMimeData()
        mime.setData(MIME, QtCore.QByteArray(ref.encode("utf-8")))
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime)
        drag.exec(QtCore.Qt.CopyAction)


class LibraryDock(QtWidgets.QDockWidget):
    def __init__(self, nodes: List[Dict[str, Any]], parent: Any = None) -> None:
        super().__init__("노드 라이브러리", parent)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(5)

        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("노드 검색")
        self.search.setClearButtonEnabled(True)
        lay.addWidget(self.search)

        self.tree = LibraryTree(nodes)
        lay.addWidget(self.tree, 1)

        hint = QtWidgets.QLabel("끌어서 캔버스에 놓으면 추가됩니다")
        hint.setStyleSheet("color:#6F7478; font-size:11px;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self.setWidget(body)
        self.search.textChanged.connect(self._filter)

    def _filter(self, text: str) -> None:
        q = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            shown = 0
            for j in range(top.childCount()):
                ch = top.child(j)
                hit = (not q) or q in ch.text(0).lower() or q in (
                    ch.data(0, QtCore.Qt.UserRole) or "").lower()
                ch.setHidden(not hit)
                shown += int(hit)
            top.setHidden(shown == 0)
            # 검색 중에는 펼쳐 둔다. 접힌 채로 걸러 봐야 결과가 안 보인다
            top.setExpanded(bool(q) and shown > 0)
