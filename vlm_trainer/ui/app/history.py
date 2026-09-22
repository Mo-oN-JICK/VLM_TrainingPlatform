"""History 와 Recipe 도크.

둘 다 `api.state()` 가 준 것을 그대로 그린다. 판단은 하지 않는다.

History 는 되감을 수 있는 시점만 목록에 둔다 — 누를 수 있는 것과 없는 것이 섞이면
목록이 신뢰를 잃는다. `api` 가 이미 본문이 남은 시점만 실어 보낸다.
"""

from __future__ import annotations

from typing import Any, Dict, List

from PySide6 import QtCore, QtGui, QtWidgets

from .. import tokens as T

MUTED = "color:#6F7478; font-size:11px;"


class HistoryDock(QtWidgets.QDockWidget):
    rewindRequested = QtCore.Signal(int)

    def __init__(self, parent: Any = None) -> None:
        super().__init__("이력", parent)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(5)

        self.list = QtWidgets.QListWidget()
        self.list.setAlternatingRowColors(False)
        self.list.itemDoubleClicked.connect(self._pick)
        lay.addWidget(self.list, 1)

        self.diff = QtWidgets.QPlainTextEdit()
        self.diff.setReadOnly(True)
        self.diff.setMaximumHeight(190)
        self.diff.setStyleSheet(
            "font-family: Consolas, ui-monospace, monospace; font-size:10.5px;")
        lay.addWidget(self.diff)

        hint = QtWidgets.QLabel("두 번 누르면 그 시점으로 되감습니다")
        hint.setStyleSheet(MUTED)
        lay.addWidget(hint)

        self.setWidget(body)
        self.list.currentRowChanged.connect(self._show_diff)
        self._rows: List[Dict[str, Any]] = []

    def update_view(self, history: List[Dict[str, Any]]) -> None:
        self._rows = list(history)
        self.list.blockSignals(True)
        self.list.clear()
        cur = 0
        for i, h in enumerate(history):
            mark = "●" if h["current"] else ("·" if h["past"] else " ")
            it = QtWidgets.QListWidgetItem(f"{mark}  {h['at']}   {h['label']}")
            it.setData(QtCore.Qt.UserRole, h["index"])
            it.setToolTip(f"{h['when']}\n{h['spec_hash']}")
            if h["current"]:
                it.setForeground(QtGui.QBrush(QtGui.QColor(T.NODE["accent"])))
                cur = i
            elif h["past"]:
                # 지난 세션의 시점. 되감을 수는 있지만 이번에 한 일은 아니다
                it.setForeground(QtGui.QBrush(QtGui.QColor("#6F7478")))
            self.list.addItem(it)
        self.list.blockSignals(False)
        self.list.setCurrentRow(cur)
        self._show_diff(cur)

    def _show_diff(self, row: int) -> None:
        if 0 <= row < len(self._rows):
            self.diff.setPlainText("\n".join(self._rows[row].get("diff") or [])
                                   or "(이 시점에는 바뀐 줄이 없다)")
        else:
            self.diff.setPlainText("")

    def _pick(self, it: QtWidgets.QListWidgetItem) -> None:
        self.rewindRequested.emit(int(it.data(QtCore.Qt.UserRole)))


class RecipeDock(QtWidgets.QDockWidget):
    """Parameter Recipe — 값 몇 개만 덮어쓰는 오버레이.

    **프로젝트 스펙은 바뀌지 않는다.** 그래서 여기서 고친 값은 저장을 눌러도
    project.yaml 로 가지 않고 recipes.yaml 로 간다.
    """

    selectRecipe = QtCore.Signal(object)      # 번호, 또는 None(벗기기)
    setValue = QtCore.Signal(str, object)     # 경로, 값
    storeRequested = QtCore.Signal(object, str)
    dropPath = QtCore.Signal(str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__("레시피", parent)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        body = QtWidgets.QWidget()
        self._lay = QtWidgets.QVBoxLayout(body)
        self._lay.setContentsMargins(8, 8, 8, 8)
        self._lay.setSpacing(6)
        self.setWidget(body)
        self._built: List[QtWidgets.QWidget] = []

    def update_view(self, recipe: Dict[str, Any]) -> None:
        for w in self._built:
            w.setParent(None)
        self._built.clear()

        def put(w: QtWidgets.QWidget) -> None:
            self._lay.addWidget(w)
            self._built.append(w)

        if not recipe.get("path"):
            lbl = QtWidgets.QLabel("이 프로젝트에는 레시피 파일이 없습니다.")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(MUTED)
            put(lbl)
            return

        pick = QtWidgets.QComboBox()
        pick.addItem("(적용 안 함)", None)
        for r in recipe.get("recipes") or []:
            pick.addItem(f"{r['id']}. {r.get('name') or '이름 없음'}", r["id"])
        applied = recipe.get("applied")
        idx = pick.findData(applied)
        pick.setCurrentIndex(idx if idx >= 0 else 0)
        pick.currentIndexChanged.connect(
            lambda _i, c=pick: self.selectRecipe.emit(c.currentData()))
        put(pick)

        axes = recipe.get("overlay") or []
        if axes:
            grp = QtWidgets.QGroupBox("덮고 있는 값")
            lay = QtWidgets.QVBoxLayout(grp)
            for ax in axes:
                row = QtWidgets.QHBoxLayout()
                lbl = QtWidgets.QLabel(ax.get("path", ""))
                lbl.setStyleSheet("color:#D8DCDF; font-size:11px;")
                row.addWidget(lbl, 1)
                cut = QtWidgets.QPushButton("빼기")
                cut.setFixedWidth(46)
                cut.clicked.connect(lambda _c=False, p=ax.get("path", ""): self.dropPath.emit(p))
                row.addWidget(cut)
                box = QtWidgets.QWidget()
                box.setLayout(row)
                lay.addWidget(box)
            put(grp)

        if recipe.get("dirty"):
            warn = QtWidgets.QLabel("고친 값이 레시피 파일에 아직 담기지 않았습니다.")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color:{T.STATE['partial']}; font-size:11px;")
            put(warn)
            save = QtWidgets.QPushButton("레시피에 담기")
            save.clicked.connect(lambda: self.storeRequested.emit(applied, ""))
            put(save)

        status = recipe.get("status") or ""
        if status:
            lbl = QtWidgets.QLabel(status)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(MUTED)
            put(lbl)

        self._lay.addStretch(1)
