"""실행 화면 — 진행 폴링과 그 결과를 보여 주는 도크.

`api.run_state()` 를 0.7초마다 읽는다. 웹판의 `vlmtRunPoll` 과 같은 주기이고,
읽는 것도 같은 함수다 — 실행을 지켜보는 통로는 하나여야 한다.

**실행 자체는 하위 프로세스다.** 편집기는 그 진행 파일을 읽기만 한다. 버튼 하나가
CLI 명령 하나이고, 여러 명령을 엮어 돌리지 않는다 — 그렇게 하는 순간 CLI 에 없는
경로가 하나 생긴다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .. import tokens as T

POLL_MS = 700       # 웹판과 같은 주기
TICK_MS = 250       # 경과 시간을 화면이 스스로 세는 간격
MUTED = "color:#6F7478; font-size:11px;"


def fmt_ms(ms: float) -> str:
    """밀리초를 사람이 읽는 단위로. `render._ms` · CLI `_took` 과 같은 규칙이다 —
    같은 값이 창과 터미널에서 다르게 보이면 둘 중 하나가 틀린 것처럼 읽힌다."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    total = int(ms / 1000)
    m, sec = divmod(total, 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    return f"{m // 60}h {m % 60:02d}m"


class RunDock(QtWidgets.QDockWidget):
    """진행·격리·콘솔. 실행 중에만 쓸모가 있어 평소에는 접어 둔다."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__("실행", parent)
        self.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.BottomDockWidgetArea)
        body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.setFormat("%v / %m")
        lay.addWidget(self.bar)

        self.line = QtWidgets.QLabel("아직 아무것도 돌지 않았습니다.")
        self.line.setWordWrap(True)
        self.line.setStyleSheet(MUTED)
        lay.addWidget(self.line)

        self.quarantine = QtWidgets.QListWidget()
        self.quarantine.setMaximumHeight(150)
        lay.addWidget(QtWidgets.QLabel("격리된 샘플"))
        lay.addWidget(self.quarantine)

        self.previews = QtWidgets.QWidget()
        self._pv_lay = QtWidgets.QVBoxLayout(self.previews)
        self._pv_lay.setContentsMargins(0, 0, 0, 0)
        self._pv_lay.setSpacing(4)
        self.pv_label = QtWidgets.QLabel("Debug Output")
        lay.addWidget(self.pv_label)
        lay.addWidget(self.previews)
        self.previews.setVisible(False)
        self.pv_label.setVisible(False)
        self._pv_items: List[QtWidgets.QWidget] = []

        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        self.console.setStyleSheet(
            "font-family: Consolas, ui-monospace, monospace; font-size:10.5px;")
        lay.addWidget(QtWidgets.QLabel("콘솔 (실패했을 때만)"))
        lay.addWidget(self.console)

        self.setWidget(body)

    def update_view(self, st: Dict[str, Any]) -> None:
        total, done = int(st.get("total") or 0), int(st.get("processed") or 0)
        self.bar.setMaximum(max(total, 1))
        self.bar.setValue(min(done, max(total, 1)))

        q = st.get("quarantine") or []
        self.quarantine.clear()
        for item in q:
            self.quarantine.addItem(
                f"{item.get('sample_key', '')} @{item.get('node_id', '')}: {item.get('cause', '')}")
        extra = st.get("quarantine_total") or 0
        if extra > len(q):
            self.quarantine.addItem(f"… 외 {extra - len(q)}건")

        self._show_previews(st.get("previews") or {})

        console = st.get("console") or ""
        self.console.setPlainText(console)
        self.console.setVisible(bool(console))

    def _show_previews(self, previews: Dict[str, Any]) -> None:
        """Debug Output 규약 — 토글이 꺼져 있으면 애초에 만들어지지 않는다.
        여기서 감추는 것이 아니라, 받을 것이 없는 것이다."""
        for w in self._pv_items:
            w.setParent(None)
        self._pv_items.clear()
        self.previews.setVisible(bool(previews))
        self.pv_label.setVisible(bool(previews))
        if not previews:
            return
        for nid, p in previews.items():
            box = QtWidgets.QGroupBox(f"{nid}   {p.get('kind', '')}")
            v = QtWidgets.QVBoxLayout(box)
            raw = self.win_preview(p.get("image", ""))
            if raw:
                pm = QtGui.QPixmap()
                pm.loadFromData(raw)
                img = QtWidgets.QLabel()
                img.setPixmap(pm.scaledToWidth(260, QtCore.Qt.SmoothTransformation))
                v.addWidget(img)
            text = (p.get("text") or "").strip()
            if text:
                t = QtWidgets.QPlainTextEdit(text)
                t.setReadOnly(True)
                t.setMaximumHeight(110)
                v.addWidget(t)
            self._pv_lay.addWidget(box)
            self._pv_items.append(box)

    # 미리보기 파일은 `api.preview_file()` 만 내준다 — 경로를 그대로 읽으면
    # 이 창이 아무 파일이나 여는 문이 된다.
    def bind_preview_reader(self, fn: Any) -> None:
        self._read_preview = fn

    def win_preview(self, url: str) -> Optional[bytes]:
        fn = getattr(self, "_read_preview", None)
        if fn is None or not url:
            return None
        return fn(url.rsplit("/", 1)[-1])


class RunWatcher(QtCore.QObject):
    """진행 파일을 읽어 카드와 상태표시줄을 칠한다.

    화면이 스스로 초를 세는 이유: 폴링이 0.7초마다라 서버 값만 쓰면 초가 툭툭 끊긴다.
    받은 값을 기준점으로 삼고 그 사이를 메운다.
    """

    def __init__(self, window: Any) -> None:
        super().__init__(window)
        self.win = window
        self._base_ms = 0.0
        self._base_at = 0.0
        self._eta_ms = 0.0
        self._live = False
        self._follow_last = ""
        self._last_phase = ""

        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self.poll)

        self._tick = QtCore.QTimer(self)
        self._tick.setInterval(TICK_MS)
        self._tick.timeout.connect(self._paint_clock)
        self._tick.start()

    def start(self) -> None:
        self._follow_last = ""
        self._poll.start()
        self.poll()

    # ── 한 번 읽기 ──────────────────────────────────────────────────────
    def poll(self) -> None:
        st = self.win.editor.run_state()
        win = self.win

        win.canvas.paint_states(st.get("states") or {})
        win.run_dock.update_view(st)

        running = bool(st.get("running"))
        self._base_ms = float(st.get("elapsed_ms") or 0.0)
        self._base_at = QtCore.QDateTime.currentMSecsSinceEpoch()
        self._eta_ms = float(st.get("eta_ms") or 0.0)
        self._live = running
        self._head = self._headline(st)
        self._paint_clock()

        win.set_running(running)
        if running and st.get("active"):
            self._follow(str(st["active"]))

        if not running:
            self._poll.stop()
            self._finish(st)

    def _headline(self, st: Dict[str, Any]) -> str:
        kind, run_id = st.get("kind") or "", st.get("run_id") or ""
        train = st.get("train") or {}
        if kind == "train" and train.get("step"):
            body = (f"train {train.get('stage', '')} step {train['step']} "
                    f"loss {float(train.get('loss') or 0):.4f}")
        elif kind and kind != "run":
            body = kind
        else:
            body = f"{st.get('processed', 0)}/{st.get('total', 0)}"
        q = st.get("quarantine_total") or 0
        return f"{run_id} {body}" + (f" · 격리 {q}" if q else "")

    def _paint_clock(self) -> None:
        """폴링 사이를 메운다. 끝났으면 멈춘 값이어야 한다 — 다 끝난 작업의 숫자가
        계속 올라가면 그것은 경과 시간이 아니라 시계다."""
        if not getattr(self, "_head", ""):
            return
        gone = (QtCore.QDateTime.currentMSecsSinceEpoch() - self._base_at) if self._live else 0
        text = f"{self._head} · {fmt_ms(self._base_ms + gone)}"
        if self._live and self._eta_ms:
            text += f" · 남은 ~{fmt_ms(max(0.0, self._eta_ms - gone))}"
        self.win.lbl_run.setText(text)

    def _follow(self, active: str) -> None:
        """실행 중인 상자가 바뀔 때만 화면을 옮긴다. 폴링마다 옮기면 사람이 다른 곳을
        보려 할 때마다 끌려 돌아온다. 이미 보이면 가만히 둔다."""
        if active == self._follow_last or not self.win.act_follow.isChecked():
            return
        self._follow_last = active
        card = self.win.canvas.cards.get(active)
        if card is None:
            return
        view = self.win.canvas
        if view.viewport().rect().contains(
            view.mapFromScene(card.sceneBoundingRect().center())
        ):
            return
        view.centerOn(card)

    def _finish(self, st: Dict[str, Any]) -> None:
        tail = ("중지" if st.get("stopped") else
                "중단" if st.get("aborted") else
                f"실패 (exit {st.get('exit')})" if st.get("exit") else "끝")
        self.win.statusBar().showMessage(
            f"{st.get('kind') or '작업'} {tail} · {fmt_ms(self._base_ms)}", 8000)
        if st.get("aborted"):
            QtWidgets.QMessageBox.warning(self.win, "중단됨", str(st["aborted"]))
        elif st.get("exit") and not st.get("stopped"):
            box = QtWidgets.QMessageBox(self.win)
            box.setWindowTitle("실행 실패")
            box.setIcon(QtWidgets.QMessageBox.Critical)
            box.setText(f"{st.get('kind') or '작업'} 가 exit {st['exit']} 로 끝났다")
            box.setDetailedText(str(st.get("console") or ""))
            box.exec()
