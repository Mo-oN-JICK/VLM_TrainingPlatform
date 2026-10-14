"""데이터 대화상자 — 표 파일, 열 지정, 정답 항목, 학습/검증 나누기를 한 번에 정한다.

사진·질문·정답 세 가지를 표(CSV) 한 장으로 넣는다. 이 창은 Line 의 값을 고치는 입구일 뿐이고,
**적용은 `Editor.line_apply_data` 한 번**이다. 표 파일을 바꾸면 열 이름이 전부 어긋나므로 하나씩
고치면 중간 상태가 거부된다 — 그래서 모아서 보내고, 통과해야 받아들인다. 거부되면 이유가 이 창
안에 보이고 아무것도 바뀌지 않는다.

"짐작"은 없다. 열은 같은 이름이 있을 때만 미리 골라 두고, 없으면 빈 칸으로 둬서 사용자가 고르게 한다.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from .. import line_data as LD
from .. import tokens as T
from ...answer import table as answer_table
from ...core.errors import SpecError
from ...engine import xlsx
from ...spec import table as spec_table
from .window import QSS

BLANK = "(고르세요)"
KEEP_TEXT = "(직접 작성한 문장 유지)"
KINDS = list(answer_table.KINDS)          # 목록 · 숫자 · 문장
COLS = ["항목", "이름", "종류", "허용값 (목록은 | 로, 숫자는 0~100)", "표의 열"]
RED = f"color:{T.STATE['failed']};"
MUTED = f"color:{T.NODE['muted']}; font-size:11px;"


class DataDialog(QtWidgets.QDialog):
    def __init__(self, editor: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self.editor = editor
        self.spec_dir = os.path.dirname(editor.path)
        self.form = LD.form(editor.line_data, self.spec_dir)
        self._custom = self.form["question_column"] is None and bool(self.form["template"])
        self._locked = False
        self.header: List[str] = []
        self._want_sheet = ""
        self.setWindowTitle("데이터 — 사진, 질문, 정답을 표로 넣습니다")
        self.resize(900, 860)
        # 어두운 바탕에서는 기본 라디오 버튼의 점이 안 보인다 — 어느 것이 골라졌는지 안 보이면 안 된다.
        self.setStyleSheet(QSS + "QRadioButton::indicator { width:12px; height:12px; border:1px solid #6F7478; border-radius:7px; }"
                           f"QRadioButton::indicator:checked {{ background:{T.NODE['accent']}; border-color:{T.NODE['accent']}; }}")
        self._build()
        self._fill()

    # ── 구성 ────────────────────────────────────────────────────────────
    def _build(self) -> None:
        lay = QtWidgets.QVBoxLayout(self)

        # ① 표와 열
        g1 = QtWidgets.QGroupBox("1  표와 열 지정 — 한 행이 한 건입니다")
        f1 = QtWidgets.QFormLayout(g1)
        row = QtWidgets.QHBoxLayout()
        self.ed_path = QtWidgets.QLineEdit()
        self.ed_path.editingFinished.connect(self._reread_table)
        btn = QtWidgets.QPushButton("찾기…")
        btn.clicked.connect(self._pick_file)
        row.addWidget(self.ed_path, 1)
        row.addWidget(btn)
        f1.addRow("표 파일", row)
        self.cb_sheet = QtWidgets.QComboBox()      # 엑셀(.xlsx)일 때만 보인다
        self.cb_sheet.currentIndexChanged.connect(lambda _i: self._reread_table())
        f1.addRow("시트", self.cb_sheet)
        self.f1 = f1
        self.lbl_table = QtWidgets.QLabel()
        self.lbl_table.setStyleSheet(MUTED)
        f1.addRow("", self.lbl_table)
        self.cb_key = QtWidgets.QComboBox()
        self.cb_image = QtWidgets.QComboBox()
        self.cb_question = QtWidgets.QComboBox()
        f1.addRow("건 구분 열", self.cb_key)
        f1.addRow("사진 열", self.cb_image)
        f1.addRow("질문 열", self.cb_question)
        lay.addWidget(g1)

        # ② 정답 항목
        g2 = QtWidgets.QGroupBox("2  정답 항목 — 모델이 답할 칸")
        v2 = QtWidgets.QVBoxLayout(g2)
        self.tbl = QtWidgets.QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setColumnWidth(0, 130)
        self.tbl.setColumnWidth(1, 120)
        self.tbl.setColumnWidth(2, 80)
        self.tbl.setColumnWidth(3, 260)
        v2.addWidget(self.tbl)
        bar = QtWidgets.QHBoxLayout()
        self.btns: List[QtWidgets.QPushButton] = []
        for text, fn in (("행 추가", self._add_row), ("행 삭제", self._del_row),
                         ("▲ 위로", lambda: self._move(-1)), ("▼ 아래로", lambda: self._move(1))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            bar.addWidget(b)
            self.btns.append(b)
        bar.addStretch(1)
        v2.addLayout(bar)
        self.lbl_schema = QtWidgets.QLabel()
        self.lbl_schema.setWordWrap(True)
        self.lbl_schema.setStyleSheet(MUTED)
        v2.addWidget(self.lbl_schema)
        lay.addWidget(g2, 1)

        # ③ 나누기
        g3 = QtWidgets.QGroupBox("3  학습용 / 검증용 나누기 — 검증용은 학습에 쓰지 않은 건만 씁니다")
        f3 = QtWidgets.QGridLayout(g3)
        self.rb_random = QtWidgets.QRadioButton("자동 — 행마다 무작위로 나눕니다")
        self.rb_group = QtWidgets.QRadioButton("자동 — 그룹 열이 같은 건은 같은 쪽으로")
        self.rb_column = QtWidgets.QRadioButton("표의 구분 열을 그대로 씁니다 (값: train / val)")
        self.cb_group = QtWidgets.QComboBox()
        self.cb_split = QtWidgets.QComboBox()
        f3.addWidget(self.rb_random, 0, 0)
        f3.addWidget(self.rb_group, 1, 0)
        f3.addWidget(self.cb_group, 1, 1)
        f3.addWidget(self.rb_column, 2, 0)
        f3.addWidget(self.cb_split, 2, 1)
        self.sp_val = QtWidgets.QSpinBox()
        self.sp_val.setRange(1, 50)
        self.sp_val.setSuffix(" %")
        self.sp_seed = QtWidgets.QSpinBox()
        self.sp_seed.setRange(0, 2_000_000_000)
        f3.addWidget(QtWidgets.QLabel("검증 비율 (자동일 때)"), 3, 0)
        f3.addWidget(self.sp_val, 3, 1)
        f3.addWidget(QtWidgets.QLabel("시드 (같은 값이면 같게 나뉩니다)"), 4, 0)
        f3.addWidget(self.sp_seed, 4, 1)
        self.lbl_counts = QtWidgets.QLabel()
        f3.addWidget(self.lbl_counts, 5, 0, 1, 2)
        lay.addWidget(g3)
        for w in (self.rb_random, self.rb_group, self.rb_column):
            w.toggled.connect(self._queue_preview)
        for w in (self.cb_group, self.cb_split, self.cb_key):
            w.currentIndexChanged.connect(self._queue_preview)
        self.sp_val.valueChanged.connect(self._queue_preview)
        self.sp_seed.valueChanged.connect(self._queue_preview)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self._preview)

        # ④ 아래
        self.lbl_err = QtWidgets.QPlainTextEdit()
        self.lbl_err.setReadOnly(True)
        self.lbl_err.setMaximumHeight(110)
        self.lbl_err.setStyleSheet(RED)
        self.lbl_err.setPlaceholderText("적용이 거부되면 이유와 고치는 방법이 여기에 보입니다.")
        lay.addWidget(self.lbl_err)
        bottom = QtWidgets.QHBoxLayout()
        self.btn_tpl = QtWidgets.QPushButton("표 양식 만들기…")
        self.btn_tpl.setToolTip("지금 정한 열 이름으로 빈 표(CSV)의 첫 줄을 저장합니다. 엑셀에서 열 수 있습니다.")
        self.btn_tpl.clicked.connect(self._template)
        self.btn_apply = QtWidgets.QPushButton("적용")
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self._apply)
        btn_close = QtWidgets.QPushButton("닫기")
        btn_close.clicked.connect(self.reject)
        bottom.addWidget(self.btn_tpl)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_apply)
        bottom.addWidget(btn_close)
        lay.addLayout(bottom)

    # ── 채우기 ──────────────────────────────────────────────────────────
    def _fill(self) -> None:
        f = self.form
        path, self._want_sheet = xlsx.split_sheet(f["index"])
        self.ed_path.setText(path)
        self._reread_table(keep=f)
        schema = f["schema"]
        self._locked = bool(schema["unsupported"])
        for r in schema["rows"]:
            self._add_row(r, f["answers"].get(r["항목"], ""))
        if self._locked:
            self.lbl_schema.setText("표로 편집할 수 없는 항목이 있어 이 구역은 읽기 전용입니다 — "
                                    + ", ".join(schema["unsupported"]) + "\n(표로 고쳐 저장하면 이 내용이 사라집니다. 표의 열 지정은 바꿀 수 있습니다.)")
            self.lbl_schema.setStyleSheet(RED)
            for b in self.btns:
                b.setEnabled(False)
            self._lock_cells(True)
        elif schema["rows"]:
            # 정답 항목은 ③ 정답 형태 패널에서 고친다(Q64). 여기서는 표의 열만 고른다 — 표를 바꾸면
            # 열 이름이 달라지므로 그것까지 막으면 바꾼 표를 적용할 길이 없다.
            self.lbl_schema.setText("정답 항목은 오른쪽 ③ 정답 형태에서 고칩니다. 여기서는 항목마다 표의 열만 고릅니다.")
            for b in self.btns:
                b.setEnabled(False)
            self._lock_cells(True)
        else:
            src = f["schema_path"] or "(없음)"
            self.lbl_schema.setText(f"지금 정답 형태 파일: {src}\n저장하면 이 프로젝트 전용 파일 "
                                    f"`{os.path.basename(LD.owned_schema_path(f['schema_path'], str(self.editor.line_data.get('id') or '')))}` "
                                    "로 만들고 원래 파일은 그대로 둡니다.")
        sp = f["split"]
        {"random": self.rb_random, "group": self.rb_group, "column": self.rb_column}[sp["mode"]].setChecked(True)
        self.sp_val.setValue(max(1, min(50, sp["val_pct"])))
        self.sp_seed.setValue(sp["seed"])
        self._set_combo(self.cb_group, sp["group"])
        self._set_combo(self.cb_split, sp["column"])
        self._preview()

    def _combo_items(self, cb: QtWidgets.QComboBox, keep: str = "") -> None:
        cur = keep or (cb.currentData() or "")
        cb.blockSignals(True)
        cb.clear()
        cb.addItem(BLANK, "")
        for c in self.header:
            cb.addItem(c, c)
        cb.blockSignals(False)
        self._set_combo(cb, cur)

    @staticmethod
    def _set_combo(cb: QtWidgets.QComboBox, value: str) -> None:
        i = cb.findData(value) if value else 0
        cb.blockSignals(True)
        cb.setCurrentIndex(i if i >= 0 else 0)   # 표에 없는 이름은 고르지 않은 것으로 둔다
        cb.blockSignals(False)

    def _sheets(self) -> None:
        """엑셀이면 시트 목록을 채워 보이고, 아니면 감춘다. 전에 고른 시트가 있으면 그대로 둔다."""
        path = self.ed_path.text().strip()
        full = LD.resolve(self.spec_dir, path)
        names: List[str] = []
        if path.lower().endswith(".xlsx") and os.path.exists(full):
            try:
                names = xlsx.sheet_names(full)
            except SpecError:
                names = []          # 못 여는 이유는 아래 표 읽기가 말한다
        want = self.cb_sheet.currentData() or self._want_sheet
        self.cb_sheet.blockSignals(True)
        self.cb_sheet.clear()
        for n in names:
            self.cb_sheet.addItem(n, n)
        self.cb_sheet.setCurrentIndex(max(0, self.cb_sheet.findData(want)))
        self.cb_sheet.blockSignals(False)
        self.cb_sheet.setVisible(bool(names))
        self.f1.labelForField(self.cb_sheet).setVisible(bool(names))

    def _index(self) -> str:
        """표 파일 경로. 엑셀이면 `표.xlsx#시트` 로 시트까지 담는다."""
        path, sheet = self.ed_path.text().strip(), self.cb_sheet.currentData()
        return f"{path}#{sheet}" if sheet and self.cb_sheet.count() else path

    def _reread_table(self, keep: Optional[Dict[str, Any]] = None) -> None:
        self._sheets()
        index = self._index()
        try:
            h = LD.read_header(self.spec_dir, index)
        except SpecError as exc:
            self.header = []
            self.lbl_table.setText(str(exc).splitlines()[0])
            self.lbl_table.setStyleSheet(RED)
        else:
            self.header = h["columns"]
            self.lbl_table.setText(f"{h['count']}건 · 열 {len(h['columns'])}개: {', '.join(h['columns'])}")
            self.lbl_table.setStyleSheet(MUTED)
        k = keep or {}
        for cb, val in ((self.cb_key, k.get("key")), (self.cb_image, k.get("image_column"))):
            self._combo_items(cb, val if val is not None else "")
        # 질문 열: 직접 쓴 문장이 있으면 "그대로 둔다"가 첫 선택이다. 표를 바꿔도 사라지지 않는다.
        first = self.cb_question.count() == 0
        prev = (None if self._custom else (k.get("question_column") or "")) if first else self.cb_question.currentData()
        self._combo_items(self.cb_question, prev or "")
        if self._custom:
            self.cb_question.insertItem(0, KEEP_TEXT, None)
            if prev is None:
                self.cb_question.setCurrentIndex(0)
        for cb in (self.cb_group, self.cb_split):
            self._combo_items(cb)
        for r in range(self.tbl.rowCount()):
            w = self.tbl.cellWidget(r, 4)
            if isinstance(w, QtWidgets.QComboBox):
                self._combo_items(w)
        self._queue_preview()

    def _pick_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "표 파일 고르기", self.spec_dir, "표 (*.csv *.xlsx *.jsonl);;모든 파일 (*)")
        if path:
            try:
                rel = os.path.relpath(path, self.spec_dir)
            except ValueError:      # 다른 드라이브면 상대 경로가 없다
                rel = path
            self.ed_path.setText(rel.replace("\\", "/"))
            self._reread_table()

    # ── 정답 항목 표 ────────────────────────────────────────────────────
    def _add_row(self, row: Any = None, column: str = "") -> None:
        row = row if isinstance(row, dict) else {"항목": "", "이름": "", "종류": "목록", "허용값": ""}
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        for c, key in enumerate(("항목", "이름")):
            self.tbl.setItem(r, c, QtWidgets.QTableWidgetItem(row.get(key, "")))
        kind = QtWidgets.QComboBox()
        kind.addItems(KINDS)
        kind.setCurrentText(row.get("종류", "목록"))
        self.tbl.setCellWidget(r, 2, kind)
        self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(row.get("허용값", "")))
        col = QtWidgets.QComboBox()
        self._combo_items(col, column)
        self.tbl.setCellWidget(r, 4, col)
        if getattr(self, "_locked", False):
            self._lock_cells(True)

    def _lock_cells(self, on: bool) -> None:
        for r in range(self.tbl.rowCount()):
            for c in (0, 1, 3):
                it = self.tbl.item(r, c)
                if it is not None:
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable if on else it.flags() | QtCore.Qt.ItemIsEditable)
            self.tbl.cellWidget(r, 2).setEnabled(not on)

    def _del_row(self) -> None:
        r = self.tbl.currentRow()
        if r >= 0:
            self.tbl.removeRow(r)

    def _move(self, d: int) -> None:
        r = self.tbl.currentRow()
        t = r + d
        if r < 0 or not (0 <= t < self.tbl.rowCount()):
            return
        a, b = self._row(r), self._row(t)
        for i, (data, col) in ((r, b), (t, a)):
            self._set_row(i, data, col)
        self.tbl.setCurrentCell(t, 0)

    def _row(self, r: int) -> tuple:
        text = lambda c: (self.tbl.item(r, c).text() if self.tbl.item(r, c) else "").strip()
        return ({"항목": text(0), "이름": text(1), "종류": self.tbl.cellWidget(r, 2).currentText(), "허용값": text(3)},
                self.tbl.cellWidget(r, 4).currentData() or "")

    def _set_row(self, r: int, row: Dict[str, str], column: str) -> None:
        self.tbl.item(r, 0).setText(row["항목"])
        self.tbl.item(r, 1).setText(row["이름"])
        self.tbl.cellWidget(r, 2).setCurrentText(row["종류"])
        self.tbl.item(r, 3).setText(row["허용값"])
        self._set_combo(self.tbl.cellWidget(r, 4), column)

    # ── 나누기 ──────────────────────────────────────────────────────────
    def _mode(self) -> str:
        return "column" if self.rb_column.isChecked() else "group" if self.rb_group.isChecked() else "random"

    def _splits(self) -> Dict[str, Any]:
        return LD.make_splits(self._mode(), column=self.cb_split.currentData() or "",
                              group=self.cb_group.currentData() or "", val_pct=self.sp_val.value(),
                              seed=self.sp_seed.value())

    def _queue_preview(self, *_a: Any) -> None:
        if hasattr(self, "_timer"):
            self._timer.start()

    def _preview(self) -> None:
        try:
            n = LD.split_counts(self.spec_dir, self._index(), self.cb_key.currentData() or "", self._splits())
        except Exception as exc:
            self.lbl_counts.setText("미리 보기: " + str(exc).splitlines()[0])
            self.lbl_counts.setStyleSheet(RED)
            return
        self.lbl_counts.setText("미리 보기: " + " · ".join(f"{k} {v}건" for k, v in sorted(n.items())))
        self.lbl_counts.setStyleSheet(MUTED if n.get("val") else RED)
        if not n.get("val"):
            self.lbl_counts.setText(self.lbl_counts.text() + "  — 검증용이 0건입니다")

    # ── 모으기·적용 ─────────────────────────────────────────────────────
    def _collect(self) -> Dict[str, Any]:
        """대화상자의 칸을 한 묶음으로. 빈 칸은 **짐작으로 채우지 않고** 여기서 거부한다."""
        rows, answers = [], {}
        for r in range(self.tbl.rowCount()):
            row, col = self._row(r)
            rows.append(row)
            if not col:
                raise SpecError(f"정답 항목 {r + 1}번째 행({row['항목'] or '이름 없음'})의 표의 열이 비었습니다.\n"
                                "  안 잡혔다면: 정답 칸이 비어 학습됩니다.\n  표에서 이 항목의 정답이 든 열을 고르세요.")
            answers[row["항목"]] = col
        need = {"건 구분 열": self.cb_key, "사진 열": self.cb_image}
        if self.cb_question.currentData() is not None:
            need["질문 열"] = self.cb_question
        for name, cb in need.items():
            if not cb.currentData():
                raise SpecError(f"{name}을 고르지 않았습니다.\n  안 잡혔다면: 이름이 틀린 채 첫 샘플에서 멈춥니다.\n  표의 열 중에서 고르세요.")
        mode = self._mode()
        if mode == "column" and not self.cb_split.currentData():
            raise SpecError("구분 열을 고르지 않았습니다.\n  안 잡혔다면: 어느 건이 검증용인지 정해지지 않습니다.\n  표의 구분 열을 고르세요.")
        if mode == "group" and not self.cb_group.currentData():
            raise SpecError("그룹 열을 고르지 않았습니다.\n  안 잡혔다면: 같은 그룹이 학습과 검증에 나뉘어 들어갑니다.\n  그룹 열을 고르세요.")
        f = dict(self.form)
        f.update({"index": self._index(), "key": self.cb_key.currentData(),
                  "image_column": self.cb_image.currentData(), "question_column": self.cb_question.currentData(),
                  "answers": answers, "splits": self._splits()})
        return {"form": f, "rows": rows}

    def _apply(self) -> None:
        self.lbl_err.setPlainText("")
        try:
            got = self._collect()
        except SpecError as exc:
            self.lbl_err.setPlainText(str(exc))
            return
        schema = None
        if not self._locked:
            s = self.form["schema"]
            schema = {"rows": got["rows"], "id": s["id"], "render": s["render"], "max_tokens": s["max_tokens"],
                      "step_order": s["step_order"], "vocabulary": s["vocabulary"]}
        res = self.editor.line_apply_data(got["form"], schema=schema)
        if res.get("ok"):
            self.accept()
        else:
            self.lbl_err.setPlainText(str(res.get("detail") or res.get("reason")))

    def _template(self) -> None:
        """지금 칸의 열 이름으로 빈 표의 첫 줄을 저장한다. 표가 아직 없는 사람이 시작하는 자리다."""
        try:
            got = self._collect_for_template()
            cols = spec_table.header(**got)
        except SpecError as exc:
            self.lbl_err.setPlainText(str(exc))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "표 양식 저장", os.path.join(self.spec_dir, "표_양식.csv"),
                                                        "CSV (*.csv)")
        if not path:
            return
        with open(path, "wb") as fh:
            fh.write(spec_table.template_csv(cols))
        self.lbl_err.setPlainText("")
        self.lbl_counts.setText(f"표 양식을 저장했습니다: {path}  (열: {', '.join(cols)})")

    def _collect_for_template(self) -> Dict[str, Any]:
        answers = []
        for r in range(self.tbl.rowCount()):
            row, col = self._row(r)
            answers.append(col or row["항목"])
        mode = self._mode()
        return {"key": self.cb_key.currentData() or "sample_id", "image": self.cb_image.currentData() or "image_path",
                "question": self.cb_question.currentData() or "question", "answers": answers,
                "split": (self.cb_split.currentData() or "split") if mode == "column" else "",
                "group": (self.cb_group.currentData() or "group") if mode == "group" else ""}
