"""데이터 입력 — 표 읽기, 열 지정, 나누기, 정답 항목 표를 한 번에 적용한다.

화면(`LineWindow` 의 데이터 대화상자)이 부르는 것과 같은 함수를 확인한다. 화면에서만 되는
입력 경로는 없다.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys

import pytest
import yaml

from vlm_trainer.answer import table as answer_table
from vlm_trainer.core import registry
from vlm_trainer.core.compiler import compile_project
from vlm_trainer.core.errors import SpecError
from vlm_trainer.ui import line_data as LD
from vlm_trainer.ui.api import Editor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOL = os.path.join(ROOT, "solutions")


@pytest.fixture(autouse=True)
def _nodes():
    registry.load_builtin_nodes()


@pytest.fixture
def work(tmp_path):
    dst = tmp_path / "solutions"
    shutil.copytree(SOL, dst, ignore=shutil.ignore_patterns("images", "runs", "edit_history*", "layout.yaml", "series"))
    proj = dst / "vlm_open" / "projects" / "01_open"
    # 한국어 열 이름의 CSV. 사진 파일은 안 읽으므로 경로만 맞다.
    src = dst / "vlm_open" / "data" / "open_parts" / "index.jsonl"
    rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = dst / "vlm_open" / "data" / "open_parts" / "new.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["번호", "사진", "질문", "부품", "방향", "묶음", "구분", "메모"])
        for i, r in enumerate(rows):
            w.writerow([r["sample_id"], r["image_path"], r["question"], r["part_type"], r["orientation"],
                        f"g{i // 4}", r["split_hint"], f"m{i}"])
    return str(proj / "line.yaml"), str(proj)


def _form(ed, spec_dir):
    return LD.form(ed.line_data, spec_dir)


# ── 스키마 ↔ 표 ─────────────────────────────────────────────────────────
def test_a_schema_goes_to_rows_and_back_without_losing_anything_the_table_can_hold():
    got = answer_table.rows_from_schema(os.path.join(SOL, "vlm_open", "schemas", "answer.yaml"))
    assert got["unsupported"] == [] and got["max_tokens"] == 48
    assert [(r["항목"], r["종류"]) for r in got["rows"]] == [("part_type", "목록"), ("orientation", "목록")]
    assert got["rows"][1]["허용값"] == "가로|세로|정사각"
    rebuilt = answer_table.build(got["rows"], schema_id=got["id"], render=got["render"], max_tokens=got["max_tokens"])
    orig = yaml.safe_load(open(os.path.join(SOL, "vlm_open", "schemas", "answer.yaml"), encoding="utf-8"))
    assert [(s["id"], s["type"], s["values"]) for s in rebuilt["steps"]] == \
        [(s["id"], s["type"], s["values"]) for s in orig["steps"]]
    assert rebuilt["constraints"] == orig["constraints"]


def test_a_schema_with_things_a_table_cannot_hold_says_so():
    got = answer_table.rows_from_schema(os.path.join(SOL, "dummy_ecg", "schemas", "answer.yaml"))
    assert got["unsupported"], "object/근거 문장이 있는데 표로 편집해도 된다고 한다"
    assert any("object" in x or "근거" in x for x in got["unsupported"])


def test_the_cli_prints_the_rows_as_a_csv_the_builder_reads_back(tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
    out = tmp_path / "rows.csv"
    r = subprocess.run([sys.executable, "-m", "vlm_trainer.cli.main", "schema-rows",
                        os.path.join(SOL, "vlm_open", "schemas", "answer.yaml"), "--out", str(out)],
                       env=env, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    rows = answer_table.read_rows(str(out))
    assert [x["항목"] for x in rows] == ["part_type", "orientation"]
    assert out.read_bytes().startswith("﻿".encode("utf-8"))

    r = subprocess.run([sys.executable, "-m", "vlm_trainer.cli.main", "schema-rows",
                        os.path.join(SOL, "dummy_ecg", "schemas", "answer.yaml")],
                       env=env, capture_output=True, text=True, encoding="utf-8")
    assert "표로 나타낼 수 없는 것" in r.stderr


# ── 표와 나누기 ─────────────────────────────────────────────────────────
def test_the_header_lists_the_columns_the_table_really_has(work):
    line, spec_dir = work
    h = LD.read_header(spec_dir, "../../data/open_parts/new.csv")
    assert h["columns"] == ["번호", "사진", "질문", "부품", "방향", "묶음", "구분", "메모"] and h["count"] == 110
    assert "image_path" in LD.read_header(spec_dir, "../../data/open_parts/index.jsonl")["columns"]
    with pytest.raises(SpecError, match="표 파일이 없다"):
        LD.read_header(spec_dir, "../../data/none.csv")
    with pytest.raises(SpecError, match="정해지지 않았다"):
        LD.read_header(spec_dir, "")


def test_the_split_preview_counts_each_mode_and_every_row_lands_somewhere(work):
    _, spec_dir = work
    idx = "../../data/open_parts/new.csv"
    by_col = LD.split_counts(spec_dir, idx, "번호", LD.make_splits("column", column="구분"))
    assert by_col == {"train": 100, "val": 10}
    auto = LD.split_counts(spec_dir, idx, "번호", LD.make_splits("random", val_pct=20, seed=3))
    assert sum(auto.values()) == 110 and 0 < auto["val"] < 55
    grp = LD.split_counts(spec_dir, idx, "번호", LD.make_splits("group", group="묶음", val_pct=20, seed=3))
    assert sum(grp.values()) == 110
    with pytest.raises(SpecError, match="그룹"):
        LD.split_counts(spec_dir, idx, "번호", LD.make_splits("group", group="없는열"))
    with pytest.raises(SpecError, match="없는 값"):
        LD.make_splits("whatever")


def test_the_form_reads_what_the_line_holds_now(work):
    line, spec_dir = work
    f = _form(Editor.open_line(line), spec_dir)
    assert f["image_column"] == "image_path" and f["question_column"] == "question" and f["key"] == "sample_id"
    assert f["answers"] == {"part_type": "part_type", "orientation": "orientation"}
    assert f["split"]["mode"] == "column" and f["split"]["column"] == "split_hint"
    assert [r["항목"] for r in f["schema"]["rows"]] == ["part_type", "orientation"]


# ── 한 번에 적용 ────────────────────────────────────────────────────────
def _apply_form(ed, spec_dir, **over):
    f = _form(ed, spec_dir)
    f.update({"index": "../../data/open_parts/new.csv", "key": "번호", "image_column": "사진",
              "question_column": "질문", "answers": {"part_type": "부품", "orientation": "방향"},
              "splits": LD.make_splits("column", column="구분")})
    f.update(over)
    return f


def test_switching_the_whole_table_applies_in_one_step_and_survives_a_save(work):
    line, spec_dir = work
    ed = Editor.open_line(line)
    # 하나씩 고치면 중간 상태가 거부된다 — 이것이 한 번에 적용하는 이유다
    assert not ed.line_set("s1", "image_column", "사진")["ok"]
    r = ed.line_apply_data(_apply_form(ed, spec_dir))
    assert r["ok"], r
    assert ed.dirty and ed.compiled.sample_space["index"].endswith("new.csv")
    assert ed.compiled.nodes["n_img"].params["column"] == "사진"
    assert ed.compiled.nodes["n_cols"].params["columns"] == ["질문", "부품", "방향"]
    assert ed.line_data["stations"][1]["params"]["template"] == "{질문}"
    assert ed.line_data["stations"][2]["params"]["mapping"] == {"part_type": "부품", "orientation": "방향"}

    assert ed.line_save()["ok"]
    assert compile_project(line).spec_hash == ed.compiled.spec_hash


def test_an_application_with_a_wrong_column_changes_nothing(work):
    line, spec_dir = work
    ed = Editor.open_line(line)
    before = ed.compiled.spec_hash
    r = ed.line_apply_data(_apply_form(ed, spec_dir, question_column="없는질문"))
    assert not r["ok"] and "없는질문" in r["detail"] or "{없는질문}" in r["detail"]
    assert ed.compiled.spec_hash == before and not ed.dirty


def test_a_blank_split_column_is_refused_in_the_dialog_path_too(work):
    line, spec_dir = work
    ed = Editor.open_line(line)
    src = os.path.join(spec_dir, "..", "..", "data", "open_parts", "new.csv")
    text = open(src, encoding="utf-8-sig").read().replace(",train,m3", ",,m3", 1)
    open(src, "w", encoding="utf-8-sig", newline="").write(text)
    r = ed.line_apply_data(_apply_form(ed, spec_dir))
    assert not r["ok"] and "비었거나 없다" in r["detail"]


# ── 정답 항목 표를 프로젝트 전용 파일로 ─────────────────────────────────
def test_the_answer_table_is_saved_as_a_project_file_and_the_original_is_left_alone(work):
    line, spec_dir = work
    schemas = os.path.join(spec_dir, "..", "..", "schemas")
    original = open(os.path.join(schemas, "answer.yaml"), "rb").read()
    ed = Editor.open_line(line)
    f = _apply_form(ed, spec_dir)
    sc = dict(f["schema"])
    sc["rows"] = sc["rows"] + [{"항목": "memo", "이름": "메모", "종류": "문장", "허용값": ""}]
    f["answers"] = {**f["answers"], "memo": "메모"}

    r = ed.line_apply_data(f, schema=sc)
    assert r["ok"], r
    assert ed.line_data["stations"][2]["params"]["schema_path"] == "../../schemas/01_open_answer.yaml"
    mine = yaml.safe_load(open(os.path.join(schemas, "01_open_answer.yaml"), encoding="utf-8"))
    assert [s["id"] for s in mine["steps"]] == ["part_type", "orientation", "memo"]
    assert mine["constraints"]["max_tokens"] == 48, "원래 제약이 승계되지 않았다"
    assert open(os.path.join(schemas, "answer.yaml"), "rb").read() == original, "공유될 수 있는 원본이 바뀌었다"
    assert "메모" in ed.compiled.nodes["n_cols"].params["columns"]

    # 같은 프로젝트가 다시 고치면 자기 파일을 덮는다
    sc2 = dict(sc, rows=sc["rows"][:2])
    f2 = _form(ed, spec_dir)
    f2.update(f, schema_path=ed.line_data["stations"][2]["params"]["schema_path"], answers=f["answers"])
    f2["answers"] = {k: v for k, v in f["answers"].items() if k != "memo"}
    assert ed.line_apply_data(f2, schema=sc2)["ok"]
    assert [s["id"] for s in yaml.safe_load(open(os.path.join(schemas, "01_open_answer.yaml"), encoding="utf-8"))["steps"]] \
        == ["part_type", "orientation"]


def test_a_rejected_application_does_not_leave_the_project_file_behind(work):
    line, spec_dir = work
    schemas = os.path.join(spec_dir, "..", "..", "schemas")
    ed = Editor.open_line(line)
    f = _apply_form(ed, spec_dir, answers={"part_type": "부품", "orientation": "방향", "memo": "없는열"})
    sc = dict(f["schema"])
    sc["rows"] = sc["rows"] + [{"항목": "memo", "이름": "메모", "종류": "문장", "허용값": ""}]
    r = ed.line_apply_data(f, schema=sc)
    assert not r["ok"] and not os.path.exists(os.path.join(schemas, "01_open_answer.yaml"))
    assert ed.line_data["stations"][2]["params"]["schema_path"] == "../../schemas/answer.yaml"


def test_a_bad_answer_row_is_refused_with_the_row_number_before_anything_is_written(work):
    line, spec_dir = work
    schemas = os.path.join(spec_dir, "..", "..", "schemas")
    ed = Editor.open_line(line)
    f = _apply_form(ed, spec_dir)
    sc = dict(f["schema"])
    sc["rows"] = [dict(sc["rows"][0]), dict(sc["rows"][0])]       # 항목이 겹친다
    r = ed.line_apply_data(f, schema=sc)
    assert not r["ok"] and "2번째 행" in r["detail"]
    assert not os.path.exists(os.path.join(schemas, "01_open_answer.yaml")) and not ed.dirty


# ── 대화상자 ────────────────────────────────────────────────────────────
pytest.importorskip("PySide6", reason="편집기는 선택 사항이다 — pip install -e .[app]")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_dialog_opens_filled_with_what_the_line_holds(app, work):
    from vlm_trainer.ui.app.data_dialog import DataDialog

    line, _ = work
    d = DataDialog(Editor.open_line(line))
    assert d.ed_path.text().endswith("index.jsonl") and "110건" in d.lbl_table.text()
    assert d.cb_key.currentData() == "sample_id" and d.cb_image.currentData() == "image_path"
    assert d.cb_question.currentData() == "question"
    assert d.tbl.rowCount() == 2 and d.tbl.cellWidget(0, 4).currentData() == "part_type"
    assert d.rb_column.isChecked() and d.cb_split.currentData() == "split_hint"
    assert "train 100건" in d.lbl_counts.text() and "val 10건" in d.lbl_counts.text()


def test_changing_the_table_leaves_unmatched_columns_blank_instead_of_guessing(app, work):
    from vlm_trainer.ui.app.data_dialog import BLANK, DataDialog

    line, _ = work
    d = DataDialog(Editor.open_line(line))
    d.ed_path.setText("../../data/open_parts/new.csv")
    d._reread_table()
    assert d.cb_key.currentText() == BLANK and d.cb_image.currentText() == BLANK
    assert d.tbl.cellWidget(0, 4).currentText() == BLANK, "이름이 다른 열을 짐작해 골랐다"
    d._apply()
    assert "표의 열이 비었다" in d.lbl_err.toPlainText() and not d.editor.dirty


def test_filling_the_dialog_and_pressing_apply_writes_the_line_and_the_project_schema(app, work):
    from PySide6 import QtWidgets
    from vlm_trainer.ui.app.data_dialog import DataDialog

    line, spec_dir = work
    ed = Editor.open_line(line)
    d = DataDialog(ed)
    d.ed_path.setText("../../data/open_parts/new.csv")
    d._reread_table()
    for cb, v in ((d.cb_key, "번호"), (d.cb_image, "사진"), (d.cb_question, "질문")):
        cb.setCurrentIndex(cb.findData(v))
    for r, v in ((0, "부품"), (1, "방향")):
        d.tbl.cellWidget(r, 4).setCurrentIndex(d.tbl.cellWidget(r, 4).findData(v))
    d.tbl.item(1, 3).setText("가로|세로|정사각|대각")                       # 허용값을 하나 늘린다
    d.rb_group.setChecked(True)
    d.cb_group.setCurrentIndex(d.cb_group.findData("묶음"))
    d.sp_val.setValue(20)
    d._apply()
    assert d.lbl_err.toPlainText() == "" and d.result() == QtWidgets.QDialog.Accepted
    assert ed.line_data["sample_space"]["splits"]["strategy"] == "group_holdout"
    assert ed.line_data["stations"][2]["params"]["schema_path"].endswith("01_open_answer.yaml")
    mine = yaml.safe_load(open(os.path.join(spec_dir, "..", "..", "schemas", "01_open_answer.yaml"), encoding="utf-8"))
    assert mine["steps"][1]["values"] == ["가로", "세로", "정사각", "대각"]


def test_a_rejected_application_stays_open_and_shows_why(app, work):
    from PySide6 import QtWidgets
    from vlm_trainer.ui.app.data_dialog import DataDialog

    line, _ = work
    ed = Editor.open_line(line)
    d = DataDialog(ed)
    d.tbl.item(0, 3).setText("")                                              # 목록인데 허용값이 없다
    d._apply()
    assert d.result() != QtWidgets.QDialog.Accepted and "1번째 행" in d.lbl_err.toPlainText() and not ed.dirty


def test_the_template_button_writes_the_columns_the_dialog_names(app, work, tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    from vlm_trainer.ui.app.data_dialog import DataDialog

    line, _ = work
    d = DataDialog(Editor.open_line(line))
    out = tmp_path / "양식.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    d._template()
    raw = out.read_bytes()
    assert raw.startswith("﻿".encode("utf-8"))
    assert raw.decode("utf-8-sig").strip() == "sample_id,image_path,question,part_type,orientation,split_hint"


def test_a_schema_the_table_cannot_hold_locks_the_item_table_but_not_the_column_choice(app, work):
    from vlm_trainer.ui.app.data_dialog import DataDialog

    line, spec_dir = work
    ed = Editor.open_line(line)
    ed.line_data["stations"][2]["params"]["schema_path"] = "../../../dummy_ecg/schemas/answer.yaml"
    d = DataDialog(ed)
    assert d._locked and not d.btns[0].isEnabled() and "읽기 전용" in d.lbl_schema.text()
    assert d.tbl.cellWidget(0, 4).isEnabled() and not d.tbl.cellWidget(0, 2).isEnabled()
