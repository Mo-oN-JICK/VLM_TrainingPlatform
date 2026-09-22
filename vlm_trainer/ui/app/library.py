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
            for n in sorted(by_cat[cat], key=lambda d: d["type"]):
                # 버전(@1.0.0)은 접어 둔다 — 고를 때 필요한 정보가 아니다
                label = n["type"].split("@")[0]
                it = QtWidgets.QTreeWidgetItem(top, [f"[{KIND_MARK.get(n['kind'], '?')}] {label}"])
                it.setData(0, QtCore.Qt.UserRole, n["type"])
                it.setToolTip(0, f"{n['type']}\n{n.get('summary', '')}\n\n"
                                 f"입력: {', '.join(n['inputs']) or '없음'}\n"
                                 f"출력: {', '.join(n['outputs']) or '없음'}")
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
