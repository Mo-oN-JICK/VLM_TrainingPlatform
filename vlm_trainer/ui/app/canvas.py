"""노드 그래프 캔버스 — QGraphicsView.

배치는 `ui/layout.py`가 준 것을 그대로 쓴다. 여기서 좌표를 다시 셈하지 않는다 —
그 순간 앱과 `vlmt view`가 같은 그래프를 다르게 그리기 시작한다.

색은 `ui/tokens.py`. 이 파일이 아는 것은 "어떻게 그리는가"와 "무엇을 눌렀는가"뿐이다.
**그래프를 바꾸는 판단은 하지 않는다** — 시그널로 올려 보내고, `api.py`가 게이트를
통과시킨 뒤에야 화면을 다시 그린다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from .. import layout as L
from .. import tokens as T
from .. import words as W
from ...core.node import NodeKind

FONT = "Malgun Gothic"        # 한글이 나오는 Windows 기본 폰트
ZOOM_MIN, ZOOM_MAX = 0.25, 4.0

Z_WIRE, Z_CARD, Z_CHIP, Z_LIVE = 1, 2, 3, 20
MIME = "application/x-vlmt-node"


def _c(hex_: str) -> QtGui.QColor:
    return QtGui.QColor(hex_)


class PortChip(QtWidgets.QGraphicsRectItem):
    """카드 위아래에 붙는 포트 칩. 배선을 잡고 놓는 자리다.

    **카드의 자식**이다. 카드를 옮기면 Qt가 알아서 따라오게 하려는 것이다 —
    좌표를 손으로 동기화하면 언젠가 어긋난다.
    """

    def __init__(self, node_id: str, port: str, is_input: bool,
                 rect: QtCore.QRectF, color: str, type_label: str,
                 parent: QtWidgets.QGraphicsItem) -> None:
        super().__init__(rect, parent)
        self.node_id, self.port, self.is_input = node_id, port, is_input
        self.setBrush(QtGui.QBrush(_c(color)))
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.setZValue(Z_CHIP)
        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip(f"{node_id}:{port}   {type_label}")
        self._type_label = type_label

    @property
    def ref(self) -> str:
        return f"{self.node_id}:{self.port}"

    def set_dimmed(self, on: bool) -> None:
        """배선을 끄는 동안, 꽂을 수 없는 자리는 죽인다.
        못 놓는 곳을 밝은 채로 두면 놓아 보고 거부당한 뒤에야 알게 된다."""
        self.setOpacity(0.2 if on else 1.0)

    def anchor(self) -> QtCore.QPointF:
        """배선이 닿는 점. 입력은 칩의 위쪽, 출력은 아래쪽 — 수직 흐름 규약이다."""
        r = self.sceneBoundingRect()
        return QtCore.QPointF(r.center().x(), r.top() if self.is_input else r.bottom())

    def paint(self, p: QtGui.QPainter, opt: Any, w: Any = None) -> None:
        super().paint(p, opt, w)
        r = self.rect()
        avail = int(r.width()) - 6
        p.setPen(QtGui.QPen(_c("#E8EDF0")))
        p.setFont(QtGui.QFont(FONT, 6))
        fm = QtGui.QFontMetrics(p.font())
        p.drawText(QtCore.QRectF(r.left() + 3, r.top() + 1, r.width() - 6, 10),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   fm.elidedText(self._type_label, QtCore.Qt.ElideRight, avail))
        p.setFont(QtGui.QFont(FONT, 7, QtGui.QFont.Bold))
        fm = QtGui.QFontMetrics(p.font())
        p.drawText(QtCore.QRectF(r.left() + 3, r.top() + 11, r.width() - 6, 12),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   fm.elidedText(self.port, QtCore.Qt.ElideRight, avail))


class NodeCard(QtWidgets.QGraphicsRectItem):
    """상자 하나. 접힌 Procedure면 안쪽 노드 여럿을 대표한다.

    좌표는 `pos()`가 들고 rect 는 (0,0,w,h)다. 그래야 카드를 옮길 때 자식 칩이 따라온다.
    """

    def __init__(self, shown: L.Shown, w: int, h: int) -> None:
        super().__init__(QtCore.QRectF(0, 0, w, h))
        self.shown = shown
        self.state = "pending"
        self.state_extra = ""
        self.boundary = False
        # 카드는 배치 좌표보다 입력 칩 높이만큼 아래에 그려진다. 옮긴 자리를 저장할 때
        # 이 값을 빼지 않으면 움직일 때마다 그만큼씩 아래로 밀려난다.
        self.top_h = 0
        self.setZValue(Z_CARD)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCursor(QtCore.Qt.SizeAllCursor)
        self.setToolTip(f"{shown.id}\n{shown.ref}\n{shown.summary}")

    # ── 이동 ────────────────────────────────────────────────────────────
    def itemChange(self, change: Any, value: Any) -> Any:
        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self.scene():
            # 격자에 붙인다. 손으로 옮긴 상자가 미세하게 어긋나 있으면 그래프가
            # 정돈돼 보이지 않는다 — 웹판과 같은 눈금(GRID 16)이다.
            g = float(L.GRID)
            value = QtCore.QPointF(round(value.x() / g) * g, max(0.0, round(value.y() / g) * g))
        elif change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            views = self.scene().views() if self.scene() else []
            if views:
                views[0].reroute_for(self.shown.id)
        return super().itemChange(change, value)

    # ── 그리기 ──────────────────────────────────────────────────────────
    def paint(self, p: QtGui.QPainter, opt: Any, w: Any = None) -> None:
        r = self.rect()
        sel = self.isSelected()
        border = (T.NODE["border_selected"] if sel
                  else T.STATE.get(self.state, T.NODE["border"]))
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        p.setBrush(QtGui.QBrush(_c(T.NODE["bg_selected"] if sel else T.NODE["bg"])))
        p.setPen(QtGui.QPen(_c(border), 2 if sel or self.state == "running" else 1))
        p.drawRect(r)
        p.fillRect(QtCore.QRectF(r.left() + 1, r.top() + 1, 3, r.height() - 2),
                   _c(T.category_color(self.shown.category)))
        if self.boundary:
            # 물질화 경계 — 여기까지 미리 굽는다. 점선으로 표시한다
            p.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
            p.setPen(QtGui.QPen(_c(T.STATE["cached"]), 1, QtCore.Qt.DashLine))
            p.drawRect(r.adjusted(-4, -4, 4, 4))
        p.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        self._paint_head(p, r)
        self._paint_rows(p, r)

    def _paint_head(self, p: QtGui.QPainter, r: QtCore.QRectF) -> None:
        p.setFont(QtGui.QFont(FONT, 9, QtGui.QFont.Bold))
        fm = QtGui.QFontMetrics(p.font())
        name = self.shown.label or self.shown.id
        right = 92 if self.shown.inner else 72
        p.setPen(QtGui.QPen(_c(T.NODE["title"])))
        p.drawText(QtCore.QRectF(r.left() + 10, r.top() + 4, r.width() - right, L.HEAD_H - 6),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   fm.elidedText(name, QtCore.Qt.ElideRight, int(r.width()) - right))

        x = r.right() - 10
        if self.state_extra:
            p.setFont(QtGui.QFont(FONT, 7))
            fm2 = QtGui.QFontMetrics(p.font())
            wid = min(fm2.horizontalAdvance(self.state_extra), 120)
            p.setPen(QtGui.QPen(_c("#8A9196")))
            p.drawText(QtCore.QRectF(x - wid, r.top() + 5, wid, 14),
                       QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                       fm2.elidedText(self.state_extra, QtCore.Qt.ElideRight, wid))
            x -= wid + 6
        p.setBrush(QtGui.QBrush(_c(T.STATE.get(self.state, "#4A4A4A"))))
        p.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        p.drawEllipse(QtCore.QRectF(x - 7, r.top() + 9, 6, 6))

        if self.shown.inner:
            p.setPen(QtGui.QPen(_c(T.NODE["accent"])))
            p.setFont(QtGui.QFont(FONT, 7))
            p.drawText(QtCore.QRectF(r.right() - 92, r.top() + 5, 26, 14),
                       QtCore.Qt.AlignCenter, f"P·{self.shown.inner}")

    def _paint_rows(self, p: QtGui.QPainter, r: QtCore.QRectF) -> None:
        """입력 / 처리 / 출력. 포트 이름이 아니라 **그 포트가 무엇인지**를 적는다."""
        y = r.top() + L.HEAD_H + 4
        s = self.shown

        def row(kind: str, text: str, color: Optional[str]) -> None:
            nonlocal y
            if color:
                p.setBrush(QtGui.QBrush(_c(color)))
                p.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                p.drawRect(QtCore.QRectF(r.left() + 10, y + 4, 6, 6))
            p.setFont(QtGui.QFont(FONT, 7))
            p.setPen(QtGui.QPen(_c("#6F7478")))
            p.drawText(QtCore.QRectF(r.left() + 21, y, 34, L.ROW_H),
                       QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, kind)
            p.setFont(QtGui.QFont(FONT, 7.5))
            p.setPen(QtGui.QPen(_c("#A8B0B6" if color or kind == "처리" else "#5F6468")))
            avail = int(r.width()) - 66
            fm = QtGui.QFontMetrics(p.font())
            p.drawText(QtCore.QRectF(r.left() + 56, y, avail, L.ROW_H),
                       QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                       fm.elidedText(text, QtCore.Qt.ElideRight, avail))
            y += L.ROW_H

        if s.inputs:
            for i, (name, t) in enumerate(s.inputs.items()):
                _, root, is_list = L._type_label(t)
                row("입력" if len(s.inputs) == 1 else f"입력 {i + 1}",
                    L._port_doc(s, "입력", name) or name, T.port_color(root, is_list))
        else:
            row("입력", "없음", None)

        row("처리", s.hint or s.label or s.ref, None)

        if s.outputs:
            for i, (name, t) in enumerate(s.outputs.items()):
                _, root, is_list = L._type_label(t)
                row("출력" if len(s.outputs) == 1 else f"출력 {i + 1}",
                    L._port_doc(s, "출력", name) or name, T.port_color(root, is_list))
        else:
            row("출력", "없음", None)


class Wire(QtWidgets.QGraphicsPathItem):
    """배선 하나. 색은 **소스 포트 타입 색**이다 — Mech-Vision 의 관측된 규칙이다."""

    def __init__(self, src: PortChip, dst: PortChip, color: str) -> None:
        super().__init__()
        self.src, self.dst = src, dst
        self._base = _c(color)
        self.setPen(QtGui.QPen(self._base, 2))
        self.setZValue(Z_WIRE)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{src.ref}  →  {dst.ref}\n우클릭으로 끊는다")
        self.reroute()

    def reroute(self) -> None:
        a, b = self.src.anchor(), self.dst.anchor()
        path = QtGui.QPainterPath(a)
        dy = max(28.0, abs(b.y() - a.y()) * 0.45)
        path.cubicTo(a.x(), a.y() + dy, b.x(), b.y() - dy, b.x(), b.y())
        self.setPath(path)

    def highlight(self, on: bool) -> None:
        """선택한 상자에 닿는 배선만 밝힌다. 배선이 카드 밑을 지나가면 어디서
        어디로 가는지 따라가기 어렵다 — 그때 이것이 답을 준다."""
        self.setPen(QtGui.QPen(self._base.lighter(160) if on else self._base, 3 if on else 2))
        self.setZValue(Z_LIVE if on else Z_WIRE)

    def shape(self) -> QtGui.QPainterPath:
        """얇은 곡선을 맞히기 쉽게 두껍게 잡는다. 2px 선을 정확히 누르라고 하면
        우클릭으로 끊는 기능이 있으나 마나다."""
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(12)
        return stroker.createStroke(self.path())


class GraphCanvas(QtWidgets.QGraphicsView):
    """팬·줌·선택·배선·이동. 바꾸는 일은 전부 시그널로 올려 보낸다."""

    selected = QtCore.Signal(str)                 # 상자 id. 빈 문자열이면 해제
    connectRequested = QtCore.Signal(str, str)    # 출발 "노드:포트", 도착 "노드:포트"
    disconnectRequested = QtCore.Signal(str)      # 도착 "노드:포트"
    movedNode = QtCore.Signal(str, int, int)
    removeRequested = QtCore.Signal(str)
    expandRequested = QtCore.Signal(str)
    boundaryRequested = QtCore.Signal(str, bool)
    addRequested = QtCore.Signal(str, int, int)   # 노드 타입, 놓은 자리

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setFrameStyle(0)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.SmartViewportUpdate)
        self.setBackgroundBrush(QtGui.QBrush(_c(T.SURFACE["canvas"])))
        self.setAcceptDrops(True)

        self.cards: Dict[str, NodeCard] = {}
        self.chips: Dict[str, PortChip] = {}
        self.wires: List[Wire] = []
        self.compat: Dict[str, Dict[str, str]] = {}
        self.occupied: set = set()
        # 포트 호버 설명이 쓰는 것. "지금 무엇이 꽂혀 있나" 와 "마지막으로 흐른 값".
        self.wired_from: Dict[str, str] = {}
        self.port_values: Dict[str, str] = {}

        self._panning = False
        self._pan_from = QtCore.QPoint()
        self._wire_from: Optional[PortChip] = None
        self._rubber: Optional[QtWidgets.QGraphicsPathItem] = None
        self._drag_start: Dict[str, Tuple[float, float]] = {}
        self._loading = False       # 다시 그리는 중에는 아이템을 만지지 않는다
        self._scene.selectionChanged.connect(self._on_selection)

    # ── 격자 ────────────────────────────────────────────────────────────
    def drawBackground(self, p: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        """격자를 아이템이 아니라 배경으로 그린다. 장면에 선을 수천 개 넣으면
        선택·충돌 계산이 전부 그만큼 느려진다."""
        super().drawBackground(p, rect)
        step = L.GRID * 4
        p.setPen(QtGui.QPen(_c("#202020"), 0))
        lines = []
        x = int(rect.left()) - int(rect.left()) % step
        while x < rect.right():
            lines.append(QtCore.QLineF(x, rect.top(), x, rect.bottom()))
            x += step
        y = int(rect.top()) - int(rect.top()) % step
        while y < rect.bottom():
            lines.append(QtCore.QLineF(rect.left(), y, rect.right(), y))
            y += step
        p.drawLines(lines)

    # ── 채우기 ──────────────────────────────────────────────────────────
    def load(self, cg: Any, expanded: Any = (), fixed: Optional[Dict] = None,
             compat: Optional[Dict] = None, occupied: Any = (),
             boundary: Any = (), wired_from: Optional[Dict] = None) -> None:
        keep = self._selected_id()
        # **보던 자리와 배율을 기억한다.** 아래에서 scene 을 통째로 비우고 다시 채우는데,
        # 그대로 두면 Qt 가 경계를 다시 셈하면서 화면이 튄다 — 줌을 맞춰 놓고 노드를
        # 하나 추가하면 비율이 깨져 보이던 것이 이것이다.
        had = bool(self.cards)
        keep_zoom = self.transform().m11()
        keep_center = self.mapToScene(self.viewport().rect().center())
        # **지우기 전에** 파이썬 쪽 참조를 비운다. `scene.clear()` 는 C++ 객체를
        # 없애는데, 그동안 selectionChanged 가 날아와 죽은 배선을 만지면 그대로 죽는다.
        self._loading = True
        self._scene.blockSignals(True)
        self.cards.clear()
        self.chips.clear()
        self.wires.clear()
        self._scene.clear()
        self._scene.blockSignals(False)
        self._scene.setSceneRect(QtCore.QRectF())   # 비운 뒤 경계도 함께 비운다
        self.compat = compat or {}
        self.occupied = set(occupied or ())
        bset = set(boundary or ())
        self.wired_from = dict(wired_from or {})

        shown, edges = L.fold(cg, expanded)
        placed = L._layout(shown, edges, fixed)

        for nid, pl in placed.items():
            s = shown[nid]
            top_h = 0 if s.kind is NodeKind.INPUT else L.CHIP_H
            card = NodeCard(s, pl.w, pl.h)
            card.top_h = top_h
            card.boundary = nid in bset or any(i in bset for i in s.state_ids)
            card.setPos(pl.x, pl.y + top_h)
            self._scene.addItem(card)
            self.cards[nid] = card

            # 칩은 카드의 자식이다. 좌표는 카드 기준 — 입력은 위(-CHIP_H), 출력은 아래
            if s.kind is not NodeKind.INPUT:
                for name, cx, cy, cw in L._chips(s.inputs, 0, -L.CHIP_H, pl.w):
                    self._add_chip(card, nid, name, s.inputs[name], cx, cy, cw, True)
            for name, cx, cy, cw in L._chips(s.outputs, 0, pl.h, pl.w):
                self._add_chip(card, nid, name, s.outputs[name], cx, cy, cw, False)

        for a, b in edges:
            src, dst = self.chips.get(f"out:{a}"), self.chips.get(f"in:{b}")
            if src is None or dst is None:
                continue
            nid, port = a.split(":", 1)
            _, root, is_list = L._type_label(shown[nid].outputs[port])
            wire = Wire(src, dst, T.wire_color(root, is_list))
            self._scene.addItem(wire)
            self.wires.append(wire)

        # **경계를 넉넉히 잡는다.** 상자에 딱 맞추면 노드 하나를 추가할 때마다 경계가
        # 바뀌고, 그때마다 스크롤바가 생겼다 사라지며 화면이 출렁인다.
        bounds = self._scene.itemsBoundingRect()
        margin = max(L.PAD, int(max(bounds.width(), bounds.height()) * 0.25))
        self._scene.setSceneRect(bounds.adjusted(-margin, -margin, margin, margin))

        # 보던 자리와 배율을 되돌린다. 처음 그리는 것(`had=False`)은 창이 `fit()` 을
        # 따로 부르므로 건드리지 않는다.
        if had:
            self.zoom_to(keep_zoom)
            self.centerOn(keep_center)

        self._loading = False
        if keep in self.cards:
            self.cards[keep].setSelected(True)

    def _add_chip(self, card: NodeCard, nid: str, name: str, port_type: Any,
                  x: int, y: int, w: int, is_input: bool) -> None:
        base, root, is_list = L._type_label(port_type)
        chip = PortChip(nid, name, is_input, QtCore.QRectF(x, y, w, L.CHIP_H - 4),
                        T.port_color(root, is_list), f"<{base}>", card)
        # 호버하면 타입 표기가 아니라 **사람 말**이 뜬다. 타입은 맨 아래 한 줄로 남는다.
        docs = card.shown.in_docs if is_input else card.shown.out_docs
        chip.setToolTip(L.port_help(
            port_type, docs.get(name, ""),
            wired=self.wired_from.get(f"{nid}:{name}", ""),
            value=self.port_values.get(f"{nid}:{name}", ""),
            required=not getattr(port_type, "optional", False),
            is_input=is_input,
        ))
        self.chips[("in:" if is_input else "out:") + f"{nid}:{name}"] = chip

    def reroute_for(self, nid: str) -> None:
        if self._loading:
            return
        for w in self.wires:
            if nid in (w.src.node_id, w.dst.node_id):
                w.reroute()

    def paint_states(self, states: Dict[str, Dict[str, str]]) -> None:
        for nid, card in self.cards.items():
            st = states.get(nid) or {}
            card.state = st.get("state", "pending")
            card.state_extra = st.get("extra", "")
            card.update()

    # ── 팬과 줌 ─────────────────────────────────────────────────────────
    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        """휠 = 줌. 마우스 아래 지점을 고정점으로 삼는다 — 보던 곳이 달아나지 않는다."""
        now = self.transform().m11()
        target = min(ZOOM_MAX, max(ZOOM_MIN, now * (1.0015 ** ev.angleDelta().y())))
        if abs(target - now) > 1e-6:
            self.scale(target / now, target / now)

    def fit(self) -> None:
        r = self._scene.itemsBoundingRect()
        if r.isEmpty():
            return
        self.fitInView(r.adjusted(-20, -20, 20, 20), QtCore.Qt.KeepAspectRatio)
        if self.transform().m11() > 1.0:      # 작은 그래프를 확대까지 하지는 않는다
            self.resetTransform()

    def resizeEvent(self, ev: Any) -> None:
        """창이 좁아지면 배율을 같이 줄인다.

        **창을 줄였는데 배율이 그대로면 상자가 화면 밖으로 잘려 나간다.** 실제로
        그랬다 — 창 폭을 900 으로 줄이자 노드 상자의 왼쪽 절반이 사라졌다.

        키울 때는 건드리지 않는다. 자리가 넓어졌다고 멋대로 확대하면, 보려고 맞춰 둔
        배율이 창을 만질 때마다 달라진다.
        """
        super().resizeEvent(ev)
        if self._loading or not self.cards:
            return
        old, new = ev.oldSize(), ev.size()
        if not old.isValid() or new.width() >= old.width():
            return
        bounds = self._scene.itemsBoundingRect()
        if bounds.isEmpty():
            return
        # 지금 배율로 상자들이 창에 안 들어가면, 들어갈 만큼만 줄인다.
        need = (bounds.width() + 2 * L.PAD) * self.transform().m11()
        if need > new.width():
            self.zoom_to(self.transform().m11() * new.width() / need)

    def zoom_to(self, factor: float) -> None:
        now = self.transform().m11()
        target = min(ZOOM_MAX, max(ZOOM_MIN, factor))
        if now > 1e-9 and abs(target - now) > 1e-6:
            self.scale(target / now, target / now)

    def free_spot(self, x: int, y: int, w: int = 0, h: int = 0) -> Any:
        """`(x, y)` 에서 시작해 **비어 있는** 가장 가까운 자리를 찾는다.

        놓은 자리에 그대로 두면 기존 상자 위에 겹치고, 겹친 둘 다 못 읽게 된다.
        아래로 한 칸씩 내려가며 찾고, 그래도 없으면 오른쪽 열로 넘어간다 —
        이 캔버스는 위에서 아래로 흐르므로 세로가 먼저다.
        """
        w = w or L.CARD_W
        h = h or (L.HEAD_H + L.ROW_H * 4)
        taken = [c.sceneBoundingRect().adjusted(-L.GRID, -L.GRID, L.GRID, L.GRID)
                 for c in self.cards.values()]

        def clear(px: float, py: float) -> bool:
            r = QtCore.QRectF(px, py, w, h)
            return not any(t.intersects(r) for t in taken)

        step = L.GRID * 2
        for col in range(12):
            px = x + col * (w + L.PAD)
            for row in range(60):
                py = y + row * step
                if clear(px, py):
                    return int(px), int(py)
        return int(x), int(y)

    def center_on_node(self, nid: str) -> None:
        if nid in self.cards:
            self.centerOn(self.cards[nid])

    # ── 마우스 ──────────────────────────────────────────────────────────
    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.MiddleButton or (
            ev.button() == QtCore.Qt.LeftButton and ev.modifiers() & QtCore.Qt.ControlModifier
        ):
            self._panning = True
            self._pan_from = ev.position().toPoint()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            return

        item = self.itemAt(ev.position().toPoint())
        if ev.button() == QtCore.Qt.LeftButton and isinstance(item, PortChip) and not item.is_input:
            self._begin_wire(item)
            return
        if ev.button() == QtCore.Qt.LeftButton and isinstance(item, NodeCard):
            self._drag_start[item.shown.id] = (item.pos().x(), item.pos().y())
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._panning:
            d = ev.position().toPoint() - self._pan_from
            self._pan_from = ev.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            return
        if self._wire_from is not None:
            self._drag_wire(self.mapToScene(ev.position().toPoint()))
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._panning:
            self._panning = False
            self.unsetCursor()
            return
        if self._wire_from is not None:
            self._end_wire(self.mapToScene(ev.position().toPoint()))
            return
        super().mouseReleaseEvent(ev)
        for nid, (x0, y0) in list(self._drag_start.items()):
            card = self.cards.get(nid)
            if card is None:
                continue
            if abs(card.pos().x() - x0) > 0.5 or abs(card.pos().y() - y0) > 0.5:
                # 배치 좌표로 되돌려 보낸다 — 그려진 자리가 아니라
                self.movedNode.emit(nid, int(card.pos().x()),
                                    int(card.pos().y()) - card.top_h)
        self._drag_start.clear()

    # ── 배선 끌기 ───────────────────────────────────────────────────────
    def _begin_wire(self, chip: PortChip) -> None:
        """출력 칩에서 시작한다. 꽂을 수 있는 입력만 남기고 나머지는 죽인다.

        호환 표는 `api.compat_matrix()`가 컴파일 결과에서 미리 계산해 둔 것이다.
        여기서 타입을 따지지 않는다 — 판단은 한 곳에만 있어야 한다.
        """
        self._wire_from = chip
        row = self.compat.get(chip.ref, {})
        for c in self.chips.values():
            if c.is_input:
                c.set_dimmed(row.get(c.ref, "꽂을 수 없는 자리") != "")
        self._rubber = self._scene.addPath(
            QtGui.QPainterPath(), QtGui.QPen(_c(T.NODE["accent"]), 2, QtCore.Qt.DashLine))
        self._rubber.setZValue(Z_LIVE)

    def _drag_wire(self, to: QtCore.QPointF) -> None:
        if self._rubber is None or self._wire_from is None:
            return
        a = self._wire_from.anchor()
        path = QtGui.QPainterPath(a)
        dy = max(24.0, abs(to.y() - a.y()) * 0.45)
        path.cubicTo(a.x(), a.y() + dy, to.x(), to.y() - dy, to.x(), to.y())
        self._rubber.setPath(path)

    def _end_wire(self, at: QtCore.QPointF) -> None:
        src = self._wire_from
        self._wire_from = None
        if self._rubber is not None:
            self._scene.removeItem(self._rubber)
            self._rubber = None
        for c in self.chips.values():
            c.set_dimmed(False)
        if src is None:
            return

        target = next((it for it in self._scene.items(at)
                       if isinstance(it, PortChip) and it.is_input), None)
        if target is None:
            return
        why = self.compat.get(src.ref, {}).get(target.ref, "호환 표에 없는 자리다")
        if why:
            # 놓기 전에 막고 이유를 말한다. 게이트는 api.py 에도 있지만, 왜 안 되는지는
            # 배선을 놓는 그 순간에 필요하다 — 거부 메시지를 나중에 읽어서는 늦다.
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"꽂을 수 없다 — {why}")
            return
        self.connectRequested.emit(src.ref, target.ref)

    # ── 우클릭 ──────────────────────────────────────────────────────────
    def contextMenuEvent(self, ev: QtGui.QContextMenuEvent) -> None:
        """메뉴를 띄우고, **닫힌 뒤에** 무엇을 골랐는지 본다.

        동작을 메뉴 항목에 직접 매달면 그 콜백이 `exec()` 의 이벤트 루프 안에서 돈다.
        거기서 그래프를 다시 그리면 이 메뉴를 띄운 아이템이 지워진 채로 루프가 풀리고,
        Qt 가 없는 것을 만지다 죽는다. 고른 것만 받아 두고 밖에서 처리한다.
        """
        item = self.itemAt(ev.pos())
        menu = QtWidgets.QMenu(self)
        acts: Dict[QtGui.QAction, Any] = {}

        def put(text: str, fn: Any) -> None:
            acts[menu.addAction(text)] = fn

        if isinstance(item, PortChip) and item.is_input and item.ref in self.occupied:
            put("이 배선 끊기", lambda r=item.ref: self.disconnectRequested.emit(r))
        elif isinstance(item, Wire):
            put(f"배선 끊기   {item.src.ref} → {item.dst.ref}",
                lambda r=item.dst.ref: self.disconnectRequested.emit(r))
        elif isinstance(item, (NodeCard, PortChip)):
            card = item if isinstance(item, NodeCard) else item.parentItem()
            nid, on = card.shown.id, card.boundary
            if card.shown.inner:
                put("펼치기 / 접기", lambda n=nid: self.expandRequested.emit(n))
            put(f"{W.BOUNDARY} 에서 빼기" if on else f"{W.BOUNDARY} 로 지정",
                lambda n=nid, o=on: self.boundaryRequested.emit(n, not o))
            menu.addSeparator()
            put("노드 삭제", lambda n=nid: self.removeRequested.emit(n))
        else:
            put("전체 맞춤", self.fit)

        if menu.isEmpty():
            return
        chosen = menu.exec(ev.globalPos())      # 여기서 메뉴가 닫힌다
        fn = acts.get(chosen)
        if fn is not None:
            fn()

    # ── 라이브러리에서 끌어다 놓기 ──────────────────────────────────────
    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent) -> None:
        if ev.mimeData().hasFormat(MIME):
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev: QtGui.QDragMoveEvent) -> None:
        if ev.mimeData().hasFormat(MIME):
            ev.acceptProposedAction()

    def dropEvent(self, ev: QtGui.QDropEvent) -> None:
        ref = bytes(ev.mimeData().data(MIME)).decode("utf-8")
        at = self.mapToScene(ev.position().toPoint())
        g = float(L.GRID)
        # 놓은 지점을 카드의 **머리**가 아니라 배치 좌표로 본다. 입력 칩이 붙는 노드는
        # 그만큼 위로 자리를 잡아야 마우스가 있던 곳에 카드가 온다.
        self.addRequested.emit(ref, int(round(at.x() / g) * g),
                               max(0, int(round(at.y() / g) * g) - L.CHIP_H))
        ev.acceptProposedAction()

    # ── 선택 ────────────────────────────────────────────────────────────
    def _selected_id(self) -> str:
        picked = [i for i in self._scene.selectedItems() if isinstance(i, NodeCard)]
        return picked[0].shown.id if picked else ""

    def _on_selection(self) -> None:
        if self._loading:
            return
        nid = self._selected_id()
        for w in self.wires:
            w.highlight(bool(nid) and nid in (w.src.node_id, w.dst.node_id))
        self.selected.emit(nid)
