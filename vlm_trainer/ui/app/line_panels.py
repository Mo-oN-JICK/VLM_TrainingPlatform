"""오른쪽 설정 칸의 ① 이미지 · ② 질문 형태 · ③ 정답 형태 패널.

사진이 ① 에서 들어오고, ② 에서 질문이 붙고, ③ 에서 정답이 붙는 흐름을 패널 순서 그대로 보여 준다.
값을 고치는 일은 전부 `Editor.line_*` 가 하고(통과해야 받아들인다), 여기는 그 호출을 위젯에 연결할 뿐이다.

* ① 입력 방식(폴더 / 표). 폴더는 `FolderDialog` 에서 폴더·질문 문장·정답 항목을 한 번에 받는다 — 셋이
  다 있어야 컴파일이 통과한다(Q74). 첫 사진의 원본과 **모델이 보는 크기**를 나란히, 실제 리사이즈 노드로 그린다.
* ② 질문 문장. `{image}` 는 맨 앞이나 맨 뒤. 실제로 들어갈 질문을 한 줄 미리 보인다.
* ③ 정답 항목 표와 건별 정답 표. 정답은 `저장` 때 표에 쓴다(Q71). 표가 CSV 가 아니면 보기 전용이다.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import line_data as LD
from .. import tokens as T
from ...answer import table as answer_table
from ...core.errors import SpecError
from ...engine import folder_table

KINDS = list(answer_table.KINDS)          # 목록 · 숫자 · 문장
MUTED = f"color:{T.NODE['muted']}; font-size:11px;"
RED = f"color:{T.STATE['failed']}; font-size:11px;"
PREVIEW = 150                             # ① 미리보기 그림 한 변(px)
BIG = 260                                 # ③ 고른 건의 사진 한 변(px)


# ── 그림 ────────────────────────────────────────────────────────────────
def _pixmap(arr: np.ndarray, side: int) -> QtGui.QPixmap:
    arr = np.ascontiguousarray(arr[..., :3], dtype=np.uint8)
    h, w = arr.shape[:2]
    img = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(img).scaled(side, side, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


def _read_rgb(path: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:
        return None


def model_view(arr: np.ndarray, size: Any, keep_aspect: Any, mode: Any) -> np.ndarray:
    """학습에 쓰는 바로 그 리사이즈 노드로 줄인다 — 화면과 학습이 어긋나지 않게(Q63)."""
    from ...nodes.imaging import ImageResize, ResizeParams

    p = ResizeParams(size=tuple(int(x) for x in (size or (448, 448))), mode=str(mode or "bilinear"),
                     keep_aspect=True if keep_aspect is None else bool(keep_aspect))
    return ImageResize().run(None, p, image=arr)["image"]


def _label(text: str = "", style: str = MUTED) -> QtWidgets.QLabel:
    lb = QtWidgets.QLabel(text)
    lb.setWordWrap(True)
    lb.setStyleSheet(style)
    return lb


def _first_image(win: Any) -> str:
    return next(iter(win.ng_dock.thumbs.values()), "") if win.ng_dock is not None else ""


# ── 정답 항목 표 ────────────────────────────────────────────────────────
class ItemTable(QtWidgets.QWidget):
    """정답 항목(항목 · 이름 · 종류 · 허용값). 표 방식이면 `표의 열` 칸이 붙는다."""

    def __init__(self, header: Optional[List[str]] = None, parent: Any = None) -> None:
        super().__init__(parent)
        self.header = header                       # None 이면 `표의 열` 칸이 없다(폴더 방식)
        cols = ["항목", "이름", "종류", "허용값 (목록은 |, 숫자는 0~100)"] + (["표의 열"] if header is not None else [])
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tbl = QtWidgets.QTableWidget(0, len(cols))
        self.tbl.setHorizontalHeaderLabels(cols)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().setVisible(False)
        for c, w in enumerate((90, 90, 70, 170)):
            self.tbl.setColumnWidth(c, w)
        self.tbl.setMinimumHeight(90)
        self.tbl.setMaximumHeight(140)
        lay.addWidget(self.tbl)
        bar = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("행 추가")
        self.btn_add.clicked.connect(lambda: self.add_row())
        self.btn_del = QtWidgets.QPushButton("행 삭제")
        self.btn_del.clicked.connect(lambda: self.tbl.removeRow(self.tbl.currentRow()) if self.tbl.currentRow() >= 0 else None)
        bar.addWidget(self.btn_add)
        bar.addWidget(self.btn_del)
        bar.addStretch(1)
        lay.addLayout(bar)

    def add_row(self, row: Optional[Dict[str, str]] = None, column: str = "") -> None:
        row = row or {"항목": "", "이름": "", "종류": "목록", "허용값": ""}
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        self.tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(row.get("항목", "")))
        self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(row.get("이름", "")))
        kind = QtWidgets.QComboBox()
        kind.addItems(KINDS)
        kind.setCurrentText(row.get("종류", "목록"))
        self.tbl.setCellWidget(r, 2, kind)
        self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(row.get("허용값", "")))
        if self.header is not None:
            cb = QtWidgets.QComboBox()
            cb.addItem("(고르세요)", "")
            for c in self.header:
                cb.addItem(c, c)
            i = cb.findData(column) if column else 0
            cb.setCurrentIndex(i if i >= 0 else 0)
            self.tbl.setCellWidget(r, 4, cb)

    def rows(self) -> List[Dict[str, str]]:
        t = lambda r, c: (self.tbl.item(r, c).text() if self.tbl.item(r, c) else "").strip()
        return [{"항목": t(r, 0), "이름": t(r, 1), "종류": self.tbl.cellWidget(r, 2).currentText(), "허용값": t(r, 3)}
                for r in range(self.tbl.rowCount())]

    def columns(self) -> Dict[str, str]:
        if self.header is None:
            return {}
        return {row["항목"]: self.tbl.cellWidget(r, 4).currentData() or "" for r, row in enumerate(self.rows())}

    def lock(self) -> None:
        """표로 나타낼 수 없는 항목이 있는 정답 형태 — 고쳐 저장하면 그 내용이 사라진다."""
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for r in range(self.tbl.rowCount()):
            for c in range(2, self.tbl.columnCount()):
                w = self.tbl.cellWidget(r, c)
                if w is not None:
                    w.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_del.setEnabled(False)


# ── 폴더 고르기 ─────────────────────────────────────────────────────────
class FolderDialog(QtWidgets.QDialog):
    """폴더·질문 문장·정답 항목을 한 번에 받는다. 셋이 다 있어야 컴파일이 통과한다(Q74 a)."""

    def __init__(self, editor: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self.editor = editor
        self.spec_dir = os.path.dirname(editor.path)
        self.setWindowTitle("사진 폴더 고르기")
        self.resize(640, 520)
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        row = QtWidgets.QHBoxLayout()
        self.ed_folder = QtWidgets.QLineEdit(str((editor.line_data or {}).get("folder") or ""))
        self.ed_folder.editingFinished.connect(self._looked)
        btn = QtWidgets.QPushButton("찾기…")
        btn.clicked.connect(self._pick)
        row.addWidget(self.ed_folder, 1)
        row.addWidget(btn)
        form.addRow("사진 폴더", row)
        self.lbl_found = _label()
        form.addRow("", self.lbl_found)
        self.ed_q = QtWidgets.QLineEdit()
        self.ed_q.setPlaceholderText("예: {image} 공의 색상은?")
        form.addRow("질문 문장", self.ed_q)
        form.addRow("", _label("모든 사진에 같은 질문이 붙습니다. {image} 는 사진 자리이고 맨 앞이나 맨 뒤에만 둡니다."))
        self.sp_val = QtWidgets.QSpinBox()
        self.sp_val.setRange(1, 50)
        self.sp_val.setValue(20)
        self.sp_val.setSuffix(" %")
        form.addRow("검증 비율", self.sp_val)
        lay.addLayout(form)
        lay.addWidget(QtWidgets.QLabel("정답 항목 — 모델이 답할 칸"))
        self.items = ItemTable()
        current = LD.form(editor.line_data, self.spec_dir)["schema"]["rows"]
        for r in current or [None]:
            self.items.add_row(r)
        lay.addWidget(self.items, 1)
        lay.addWidget(_label("하위 폴더가 있으면 그 이름들을 첫 항목의 허용값으로 미리 적어 둡니다. 정답은 ③ 에서 넣습니다."))
        self.err = QtWidgets.QPlainTextEdit()
        self.err.setReadOnly(True)
        self.err.setMaximumHeight(90)
        self.err.setStyleSheet(RED)
        lay.addWidget(self.err)
        bb = QtWidgets.QHBoxLayout()
        bb.addStretch(1)
        ok = QtWidgets.QPushButton("적용")
        ok.setDefault(True)
        ok.clicked.connect(self._apply)
        no = QtWidgets.QPushButton("닫기")
        no.clicked.connect(self.reject)
        bb.addWidget(ok)
        bb.addWidget(no)
        lay.addLayout(bb)
        self.result: Dict[str, Any] = {}
        if self.ed_folder.text():
            self._looked()

    def _pick(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "사진 폴더", self.spec_dir)
        if d:
            self.ed_folder.setText(d)
            self._looked()

    def _looked(self) -> None:
        full = LD.resolve(self.spec_dir, self.ed_folder.text().strip())
        try:
            items, skipped = folder_table.scan(full)
        except SpecError as exc:
            self.lbl_found.setText(str(exc).splitlines()[0])
            self.lbl_found.setStyleSheet(RED)
            return
        subs = sorted({folder_table.folder_name(k) for k, _ in items} - {""})
        self.lbl_found.setText(f"사진 {len(items)}장" + (f" · 하위 폴더 {len(subs)}개" if subs else "")
                               + (f" · 건너뛴 파일 {skipped}개" if skipped else ""))
        self.lbl_found.setStyleSheet(MUTED)
        t = self.items.tbl
        if subs and t.rowCount() and not (t.item(0, 3) and t.item(0, 3).text().strip()):
            t.item(0, 3).setText(answer_table.SEP.join(subs))
            t.cellWidget(0, 2).setCurrentText("목록")

    def _apply(self) -> None:
        res = self.editor.line_use_folder(self.ed_folder.text().strip(), self.ed_q.text(), self.items.rows(),
                                          val_pct=self.sp_val.value())
        if res.get("ok"):
            self.result = res
            self.accept()
        else:
            self.err.setPlainText(str(res.get("detail") or res.get("reason")))


# ── ① 이미지 ────────────────────────────────────────────────────────────
def image_panel(win: Any, st: Dict[str, Any]) -> QtWidgets.QWidget:
    ed = win.editor
    folder = str((ed.line_data or {}).get("folder") or "")
    box = QtWidgets.QGroupBox("입력")
    # 어두운 바탕에서는 기본 라디오 점이 안 보인다 — 어느 방식인지 안 보이면 안 된다
    box.setStyleSheet("QRadioButton::indicator { width:12px; height:12px; border:1px solid #6F7478; border-radius:7px; }"
                      f"QRadioButton::indicator:checked {{ background:{T.NODE['accent']}; border-color:{T.NODE['accent']}; }}")
    lay = QtWidgets.QVBoxLayout(box)
    radios = QtWidgets.QHBoxLayout()
    rb_folder = QtWidgets.QRadioButton("사진 폴더")
    rb_table = QtWidgets.QRadioButton("표 (CSV · 엑셀)")
    (rb_folder if folder else rb_table).setChecked(True)
    radios.addWidget(rb_folder)
    radios.addWidget(rb_table)
    radios.addStretch(1)
    lay.addLayout(radios)

    def open_folder() -> None:
        dlg = FolderDialog(ed, win)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            c = dlg.result.get("counts") or {}
            win.statusBar().showMessage(f"사진 {c.get('total', 0)}장을 표(data.csv)로 만들었습니다 — 정답은 ③ 에서 넣습니다", 8000)
            win.ng_dock.bind(ed)
        win._later(win.reload_graph)      # 취소해도 다시 그려 라디오를 되돌린다

    rb_folder.clicked.connect(lambda: open_folder() if not folder else None)
    rb_table.clicked.connect(lambda: (win._open_data(), win._later(win.reload_graph)) if folder else None)

    if folder:
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_label(f"폴더: {folder}", ""), 1)
        b = QtWidgets.QPushButton("폴더 고르기…")
        b.clicked.connect(open_folder)
        row.addWidget(b)
        lay.addLayout(row)
        sync = ed.folder_sync or {}
        if not sync.get("ok", True):
            lay.addWidget(_label(str(sync.get("reason", "")), RED))
        vrow = QtWidgets.QHBoxLayout()
        vrow.addWidget(QtWidgets.QLabel("검증 비율"))
        sp = QtWidgets.QSpinBox()
        sp.setRange(1, 50)
        sp.setSuffix(" %")
        splits = dict((ed.line_data.get("sample_space") or {}).get("splits") or {})
        sp.setValue(max(1, min(50, LD.val_pct(splits))))
        old = sp.value()
        sp.editingFinished.connect(lambda: win._later(lambda: win._line_apply(ed.line_set_val_pct, sp.value()))
                                   if sp.value() != old else None)
        vrow.addWidget(sp)
        vrow.addStretch(1)
        lay.addLayout(vrow)
    else:
        b = QtWidgets.QPushButton("표 고르기 · 열 지정…")
        b.clicked.connect(win._open_data)
        lay.addWidget(b)
    ss = ed.line_data.get("sample_space") or {}
    if ed.compiled is None:     # 아직 폴더도 표도 안 고른 새 프로젝트 — 셀 것이 없다
        lay.addWidget(_label("사진 폴더나 표를 고르면 여기에 건수와 첫 사진이 보입니다."))
        win.preview_pair = False
        return box
    try:
        n = LD.split_counts(os.path.dirname(ed.path), str(ss.get("index", "")), str(ss.get("key", "")),
                            dict(ss.get("splits") or {}))
        lay.addWidget(_label(f"학습 {n.get('train', 0)}건 · 검증 {n.get('val', 0)}건"))
    except Exception as exc:
        lay.addWidget(_label("건수를 셀 수 없습니다: " + str(exc).splitlines()[0], RED))

    # 첫 사진: 원본과 모델이 보는 크기
    path = _first_image(win)
    arr = _read_rgb(path) if path else None
    if arr is not None:
        s = st["settings"]
        out = model_view(arr, s.get("size", {}).get("value"), s.get("keep_aspect", {}).get("value"),
                         s.get("mode", {}).get("value"))
        pics = QtWidgets.QGridLayout()
        for col, (a, cap) in enumerate(((arr, f"원본\n{arr.shape[1]}×{arr.shape[0]}"),
                                        (out, f"모델이 보는 크기\n{out.shape[1]}×{out.shape[0]}"))):
            pic = QtWidgets.QLabel()
            pic.setPixmap(_pixmap(a, PREVIEW))
            pic.setAlignment(QtCore.Qt.AlignCenter)
            pics.addWidget(pic, 0, col)
            pics.addWidget(_label(cap), 1, col, QtCore.Qt.AlignHCenter)
        lay.addLayout(pics)
        lay.addWidget(_label(f"첫 사진: {os.path.basename(path)}"))
    win.preview_pair = arr is not None    # 테스트가 본다
    return box


# ── ② 질문 형태 ─────────────────────────────────────────────────────────
def question_panel(win: Any, st: Dict[str, Any]) -> QtWidgets.QWidget:
    ed = win.editor
    s = st["settings"]
    tpl = str(s.get("template", {}).get("value") or "")
    policy = str(s.get("policy", {}).get("value") or "prepend")
    box = QtWidgets.QGroupBox("질문 문장")
    lay = QtWidgets.QVBoxLayout(box)
    text = LD.question_text(tpl, policy)
    le = QtWidgets.QLineEdit(text)
    le.setObjectName("question")
    le.editingFinished.connect(lambda: win._later(lambda: win._line_apply(ed.line_set_question, le.text()))
                               if le.text().strip() != text else None)
    lay.addWidget(le)
    lay.addWidget(_label("{image} 는 사진 자리이고 맨 앞이나 맨 뒤에만 둡니다. 표의 열 값을 넣으려면 {열 이름} 을 씁니다."))
    got = ed.line_answers() if ed.compiled is not None else {}
    first = (got.get("rows") or [{}])[0].get("values", {}) if got.get("ok") else {}
    lay.addWidget(_label("실제 들어갈 질문:  " + LD.question_for(tpl, policy, first), ""))
    return box


# ── ③ 정답 형태 ─────────────────────────────────────────────────────────
def answer_panel(win: Any, st: Dict[str, Any]) -> QtWidgets.QWidget:
    ed = win.editor
    spec_dir = os.path.dirname(ed.path)
    folder = bool((ed.line_data or {}).get("folder"))
    form = LD.form(ed.line_data, spec_dir)
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)

    # 정답 항목
    g1 = QtWidgets.QGroupBox("정답 항목 — 모델이 답할 칸")
    l1 = QtWidgets.QVBoxLayout(g1)
    header = None
    if not folder:
        try:
            header = LD.read_header(spec_dir, form["index"])["columns"]
        except SpecError:
            header = []
    items = ItemTable(header)
    for r in form["schema"]["rows"]:
        items.add_row(r, form["answers"].get(r["항목"], ""))
    l1.addWidget(items)
    if form["schema"]["unsupported"]:
        items.lock()
        l1.addWidget(_label("표로 편집할 수 없는 항목이 있어 읽기 전용입니다 — " + ", ".join(form["schema"]["unsupported"]), RED))
    else:
        b = QtWidgets.QPushButton("항목 적용")
        b.setToolTip("정답 항목을 바꿉니다. 폴더 방식이면 새 항목의 칸을 표에 덧붙입니다.")
        b.clicked.connect(lambda: win._later(lambda: win._line_apply(
            lambda v: ed.line_set_schema(*v), (items.rows(), items.columns()))))
        l1.addWidget(b, 0, QtCore.Qt.AlignLeft)
    win.items_table = items
    lay.addWidget(g1)

    # 건별 정답
    g2 = QtWidgets.QGroupBox("건별 정답 — 사진 · 질문 · 정답")
    l2 = QtWidgets.QVBoxLayout(g2)
    big = QtWidgets.QLabel("줄을 고르면 사진이 여기에 크게 보입니다")
    big.setAlignment(QtCore.Qt.AlignCenter)
    big.setMinimumHeight(BIG // 2)
    big.setStyleSheet(MUTED)
    l2.addWidget(big)
    got = ed.line_answers() if ed.compiled is not None else {"ok": False, "reason": "아직 컴파일되지 않았습니다"}
    labels = {r["항목"]: (r["이름"] or r["항목"]) for r in form["schema"]["rows"]}
    tpl = str(LD.station(ed.line_data, "question").get("params", {}).get("template", "")) if LD.station(ed.line_data, "question") else ""
    policy = str((LD.station(ed.line_data, "question") or {}).get("params", {}).get("policy", "prepend"))
    if got.get("ok"):
        cols = got["cols"]
        t = QtWidgets.QTableWidget(len(got["rows"]), 3 + len(cols))
        t.setObjectName("answers")
        inv = {v: k for k, v in form["answers"].items()}
        t.setHorizontalHeaderLabels(["Index", "건", "Q"] + [f"A. {labels.get(inv.get(c, c), c)}" for c in cols])
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        t.blockSignals(True)
        for i, row in enumerate(got["rows"]):
            for c, text in enumerate((str(i + 1), row["key"], "Q. " + LD.question_for(tpl, policy, row["values"]))):
                it = QtWidgets.QTableWidgetItem(text)
                it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                t.setItem(i, c, it)
            for j, col in enumerate(cols):
                it = QtWidgets.QTableWidgetItem(row["values"].get(col, ""))
                if not got["editable"]:
                    it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                t.setItem(i, 3 + j, it)
        t.blockSignals(False)
        t.setColumnWidth(0, 44)
        t.setColumnWidth(1, 110)
        t.setColumnWidth(2, 170)
        keys = [r["key"] for r in got["rows"]]

        def changed(it: QtWidgets.QTableWidgetItem) -> None:
            if it.column() >= 3:
                ed.line_set_answer(keys[it.row()], cols[it.column() - 3], it.text().strip())
                win.setWindowTitle(win._title())

        def picked() -> None:
            r = t.currentRow()
            p = win.ng_dock.thumbs.get(keys[r], "") if 0 <= r < len(keys) and win.ng_dock is not None else ""
            arr = _read_rgb(p) if p else None
            if arr is None:
                big.setText("사진을 읽을 수 없습니다" if p else "")
                return
            big.setPixmap(_pixmap(arr, BIG))

        t.itemChanged.connect(changed)
        t.itemSelectionChanged.connect(picked)
        t.setMinimumHeight(220)
        l2.addWidget(t, 1)
        win.answers_table = t
        if not got["editable"]:
            l2.addWidget(_label("보기 전용입니다 — " + got["why"].splitlines()[0]
                                + " 엑셀에서 `다른 이름으로 저장` 으로 CSV UTF-8 로 저장해 쓰세요.", RED))
        else:
            l2.addWidget(_label("입력한 정답은 `저장` 을 누르면 표에 씁니다. 빈 정답은 실행 때 ③ 에서 NG 로 나갑니다."))
    else:
        l2.addWidget(_label(str(got.get("reason", "")), RED))

    bar = QtWidgets.QHBoxLayout()
    b_csv = QtWidgets.QPushButton("CSV로 열기")
    b_csv.setToolTip("표를 엑셀로 엽니다. 수백 건을 넣을 때 씁니다. 엑셀에서 고친 뒤 `다시 읽기`.")
    b_csv.clicked.connect(lambda: open_table(win))
    bar.addWidget(b_csv)
    if folder and form["schema"]["rows"]:
        cb = QtWidgets.QComboBox()
        for r in form["schema"]["rows"]:
            cb.addItem(labels[r["항목"]], r["항목"])
        b_fill = QtWidgets.QPushButton("하위 폴더 이름으로 채우기")
        b_fill.setToolTip("빈 정답 칸만 사진이 든 하위 폴더 이름으로 채웁니다. 적힌 값은 덮지 않습니다.")
        b_fill.clicked.connect(lambda: fill_from_folder(win, cb.currentData()))
        bar.addWidget(cb)
        bar.addWidget(b_fill)
    bar.addStretch(1)
    l2.addLayout(bar)
    lay.addWidget(g2, 1)
    return w


def fill_from_folder(win: Any, item: str) -> None:
    res = win.editor.line_fill_from_folder(item)
    if not res.get("ok"):
        win._refuse("하위 폴더 이름으로 채우기", res)
        return
    win.statusBar().showMessage(f"{res['filled']}건 채움 · 이미 적힌 {res['kept']}건 그대로"
                                + (f" · 허용값에 없는 폴더 {res['not_allowed']}건" if res["not_allowed"] else "")
                                + " — `저장` 하면 표에 씁니다", 8000)
    win._later(win.reload_graph)


def open_table(win: Any) -> None:
    """표를 엑셀(기본 프로그램)로 연다. 저장 안 한 정답이 있으면 엑셀에 옛 값이 보이므로 먼저 묻는다."""
    ed = win.editor
    if ed.dirty:
        ans = QtWidgets.QMessageBox.question(
            win, "저장하지 않은 변경", "저장하지 않은 변경이 있어 표에 아직 없습니다. 저장한 뒤 열까요?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel)
        if ans != QtWidgets.QMessageBox.Save:
            return
        win.on_save()
        if ed.dirty:
            return
    got = ed.line_answers()
    if not got.get("ok"):
        win._refuse("CSV로 열기", got)
        return
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(got["path"]))
