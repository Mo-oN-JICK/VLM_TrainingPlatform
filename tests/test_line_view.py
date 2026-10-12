"""5노드 화면 — 읽기 전용, 램프·카운터·NG 표, 그리고 화면에서 한 바퀴.

계산(`ui/line_model.py`)은 위젯 없이 확인하고, 창은 그 결과를 그리는지만 본다.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from vlm_trainer.core import registry
from vlm_trainer.core.compiler import compile_project
from vlm_trainer.engine.runner import RunReport
from vlm_trainer.spec.line import describe_file
from vlm_trainer.ui import line_model as M
from vlm_trainer.ui.api import Editor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJ = os.path.join(ROOT, "solutions", "vlm_open", "projects", "01_open")
LINE = os.path.join(PROJ, "line.yaml")


@pytest.fixture(autouse=True)
def _nodes():
    registry.load_builtin_nodes()


def _rep(cg, ok=10, ng_at=None, ng=0):
    rep = RunReport(order=list(cg.order))
    for nid in cg.order:
        rep.node_state[nid] = {"success": ok}
    if ng_at:
        rep.node_state[ng_at] = {"success": ok - ng, "failed": ng}
    return rep


# ── 계산 ────────────────────────────────────────────────────────────────
def test_the_description_lists_five_stations_with_their_inner_ids():
    d = describe_file(LINE)
    assert [s["number"] for s in d["stations"]] == [1, 2, 3, 4, 5]
    assert [s["label"] for s in d["stations"]] == ["이미지", "질문 형태", "정답 형태", "모델 학습", "검증"]
    assert {"n_guard", "p_answer", "n_schema"} <= set(d["stations"][2]["members"])
    sz = d["stations"][0]["settings"]["size"]
    assert (sz["label"], sz["value"], sz["level"]) == ("사진 크기", [448, 448], "basic")
    assert d["table"]["splits"]["strategy"] == "column"


def test_before_any_run_every_lamp_is_waiting():
    cg = compile_project(LINE)
    cs = M.cards(describe_file(LINE), cg, None)
    assert [c.lamp for c in cs] == [M.PENDING] * 5 and [c.lamp_text for c in cs] == ["대기"] * 5


def test_a_clean_run_turns_the_lamps_ok_and_counts_pass():
    cg = compile_project(LINE)
    cs = M.cards(describe_file(LINE), cg, _rep(cg))
    assert all(c.lamp == M.OK for c in cs[:3]) and cs[2].passed == 10 and cs[2].ng == 0
    assert cs[2].line2 == "통과 10 · NG 0"


def test_ejecting_some_samples_is_ng_not_error_and_the_card_that_ejected_them_says_so():
    cg = compile_project(LINE)
    cs = M.cards(describe_file(LINE), cg, _rep(cg, ng_at="n_guard", ng=2))
    assert [c.lamp for c in cs[:4]] == [M.OK, M.OK, M.NG, M.OK] or cs[2].lamp == M.NG
    assert cs[2].ng == 2 and cs[0].ng == 0


def test_a_station_where_every_sample_failed_is_an_error_not_ng():
    cg = compile_project(LINE)
    rep = _rep(cg)
    for nid in cg.order:
        if nid.startswith("p_prep"):
            rep.node_state[nid] = {"failed": 10}
    cs = M.cards(describe_file(LINE), cg, rep)
    assert cs[0].lamp == M.ERROR


def test_the_training_card_shows_stage_step_and_loss_not_sample_counts():
    cg = compile_project(LINE)
    run = {"pipeline": {"stages": [{"key": "train", "state": "running"}]},
           "train": {"stage": "lora_ft", "step": 12, "loss": 4.3871}}
    cs = M.cards(describe_file(LINE), cg, _rep(cg), run)
    assert cs[3].lamp == M.RUNNING and cs[3].line2 == "lora_ft · step 12 · loss 4.387"


def test_the_training_card_is_not_ok_just_because_the_samples_were_bundled():
    cg = compile_project(LINE)
    cs = M.cards(describe_file(LINE), cg, _rep(cg))        # 학습 단계 정보 없음
    assert cs[3].lamp == M.PENDING and cs[2].lamp == M.OK
    done = {"pipeline": {"stages": [{"key": "train", "state": "done"}]}}
    assert M.cards(describe_file(LINE), cg, _rep(cg), done)[3].lamp == M.OK
    bad = {"pipeline": {"stages": [{"key": "train", "state": "failed"}]}}
    assert M.cards(describe_file(LINE), cg, _rep(cg), bad)[3].lamp == M.ERROR


def test_ng_rows_name_the_station_that_ejected_the_sample():
    cg = compile_project(LINE)
    cs = M.cards(describe_file(LINE), cg, None)
    rows = M.ng_rows(cs, [{"sample_key": "o0007", "node_id": "n_guard", "cause": "정답 어휘가 프롬프트에 있다: ['기어']\n두 줄", "hint": "템플릿에서 뺀다"}])
    assert (rows[0].card_number, rows[0].card_label) == (3, "정답 형태")
    assert rows[0].one_line == "정답 어휘가 프롬프트에 있다: ['기어']"


# ── 읽기 전용 ───────────────────────────────────────────────────────────
def test_opening_a_line_changes_nothing_on_disk_and_refuses_every_edit():
    before = sorted(os.listdir(PROJ))
    e = Editor.open_line(LINE)
    assert sorted(os.listdir(PROJ)) == before, "열기만 했는데 폴더에 파일이 생겼다"
    assert e.readonly and e.compiled is not None and e.valid
    r = e.set_param("n_guard", "mode", "normalized")
    assert not r["ok"] and "상자와 배선" in r["reason"]
    assert not e.save()["ok"]
    with open(LINE, encoding="utf-8") as fh:
        assert "kind: Line" in fh.read()


def test_the_old_editor_still_refuses_a_line_file():
    with pytest.raises(ValueError, match="5노드"):
        Editor.open(LINE)


# ── 창 ──────────────────────────────────────────────────────────────────
pytest.importorskip("PySide6", reason="편집기는 선택 사항이다 — pip install -e .[app]")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_window_draws_five_cards_top_to_bottom(app):
    from vlm_trainer.ui.app.line_window import LineWindow

    w = LineWindow(Editor.open_line(LINE))
    ys = [w.widgets[c.id].mapTo(w, w.widgets[c.id].rect().topLeft()).y() for c in w.cards]
    assert len(w.cards) == 5 and ys == sorted(ys)


def _labels(w):
    return [w.tbl_settings.item(i, 0).text() for i in range(w.tbl_settings.rowCount())
            if w.tbl_settings.item(i, 0) is not None]


def test_basic_mode_shows_only_basic_settings_and_expert_mode_adds_the_rest_and_inner_boxes(app):
    from vlm_trainer.ui.app.line_window import LineWindow

    w = LineWindow(Editor.open_line(LINE))
    w._pick("s3")
    basic = _labels(w)
    assert basic == ["정답 형태 파일", "항목별 정답 열"], basic          # Q33: 정답 표기·노출 검사는 전문가
    w._pick("s4")
    assert _labels(w) == ["학습량"] and w.tbl_settings.cellWidget(0, 1).count() == 3

    w.act_expert.setChecked(True)
    ex = _labels(w)
    assert "학습량" in ex and "단계별 횟수 직접 입력  · 전문가" in ex
    assert any("n_train" in x for x in ex) and any("n_sample" in x for x in ex), "안쪽 상자가 안 보인다"
    w._pick("s3")
    assert any("n_guard" in x for x in _labels(w)) and any("정답 노출 검사 방식" in x for x in _labels(w))
    w.act_expert.setChecked(False)
    assert _labels(w)[:2] == ["정답 형태 파일", "항목별 정답 열"] and not any("n_guard" in x for x in _labels(w))


def test_an_accepted_edit_shows_up_in_the_table_and_marks_the_title_a_rejected_one_does_not(app):
    from vlm_trainer.ui.app.line_window import LineWindow

    w = LineWindow(Editor.open_line(LINE))
    w._pick("s1")
    assert "저장 안 됨" not in w.windowTitle()
    w._apply(lambda v: w.editor.line_set("s1", "size", v), [224, 224])
    assert "저장 안 됨" in w.windowTitle() and w.tbl_settings.cellWidget(1, 1).text() == "[224, 224]"
    assert w.lbl_refuse.text() == ""

    w._apply(lambda v: w.editor.line_set("s1", "image_column", v), "photo")       # 표에 없는 열
    assert w.lbl_refuse.text().startswith("거부") and "photo" in w.detail.toPlainText()
    assert w.tbl_settings.cellWidget(0, 1).text() == "image_path", "거부됐는데 화면에 틀린 값이 남았다"


def test_the_amount_combo_changes_the_training_epochs_and_normal_restores_the_original(app):
    from vlm_trainer.ui.app.line_window import LineWindow

    w = LineWindow(Editor.open_line(LINE))
    base = w.editor.compiled.spec_hash
    w._pick("s4")
    w._apply(lambda v: w.editor.line_set_amount("s4", v), "많게")
    assert w.editor.compiled.nodes["n_train"].params["epochs"] == {"projector_align": 8, "lora_ft": 8}
    assert w.tbl_settings.cellWidget(0, 1).currentText() == "많게"
    w._apply(lambda v: w.editor.line_set_amount("s4", v), "보통")
    assert w.editor.compiled.spec_hash == base


def test_widgets_report_only_when_the_input_is_finished_and_bad_json_is_not_sent(app):
    from PySide6 import QtWidgets
    from vlm_trainer.ui.app.line_window import _BAD, make_editor

    got = []
    cb = got.append
    box = make_editor("fail", ["fail", "mask"], cb)
    box.setCurrentIndex(1)
    box.activated.emit(1)
    assert got == ["mask"]

    got.clear()
    ed = make_editor([448, 448], [], cb)
    ed.setText("[1, 2")
    ed.editingFinished.emit()
    ed.setText("[224, 224]")
    ed.editingFinished.emit()
    assert got == [_BAD, [224, 224]]
    assert isinstance(make_editor(True, [], cb), QtWidgets.QCheckBox)
    assert isinstance(make_editor(3, [], cb), QtWidgets.QSpinBox)
    assert isinstance(make_editor(0.5, [], cb), QtWidgets.QDoubleSpinBox)


def test_the_inner_view_draws_the_expanded_graph_read_only_with_the_stations_boxes_selected(app):
    from vlm_trainer.ui.app.line_window import InnerView

    ed = Editor.open_line(LINE)
    members = describe_file(LINE)["stations"][2]["members"]
    v = InnerView(ed.compiled, members, "3 정답 형태")
    assert {"n_guard", "p_answer", "n_schema"} <= set(v.canvas.cards)
    assert all(v.canvas.cards[m].isSelected() for m in members)
    assert not v.canvas.cards["n_img"].isSelected()


def test_a_progress_file_with_ejected_samples_fills_the_ng_table_and_the_detail(app, tmp_path):
    from vlm_trainer.ui.app.line_window import LineWindow

    ed = Editor.open_line(LINE)
    cg = ed.compiled
    snap = {
        "run_id": "t", "phase": "running", "order": list(cg.order), "total": 10, "processed": 10,
        "node_state": {n: {"success": 8} for n in cg.order} | {"n_guard": {"success": 8, "failed": 2}},
        "node_ms": {}, "cache": {}, "quarantine_total": 2,
        "quarantine": [{"sample_key": f"o010{i}", "node_id": "n_guard",
                        "cause": "정답 어휘가 프롬프트에 있다: ['기어']", "hint": "템플릿에서 뺀다"} for i in (1, 2)],
    }
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(snap), encoding="utf-8")
    ed.progress_path = str(path)

    w = LineWindow(ed)
    w.editor.progress_path = str(path)
    w.refresh()
    assert w.cards[2].lamp == M.NG and w.cards[2].line2.endswith("NG 2")
    assert w.tbl_ng.rowCount() == 2 and w.lbl_ng.text().startswith("NG 2건")
    assert w.tbl_ng.item(0, 2).text() == "3 정답 형태"
    w.tbl_ng.selectRow(1)
    assert "o0102" in w.detail.toPlainText() and "해결: 템플릿에서 뺀다" in w.detail.toPlainText()


def test_pressing_run_goes_through_the_whole_line_from_the_window(app, tmp_path, monkeypatch):
    """화면의 `실행` 이 `vlmt pipeline line.yaml` 을 띄우고 끝까지 간다. 램프는 정상으로 끝난다."""
    if not os.path.exists(os.path.join(ROOT, "solutions", "vlm_open", "data", "open_parts", "index.jsonl")):
        pytest.skip("vlm_open 데이터가 없다")
    pytest.importorskip("torch")
    from PySide6 import QtWidgets
    from vlm_trainer.ui.app.line_window import LineWindow

    monkeypatch.chdir(tmp_path)
    w = LineWindow(Editor.open_line(LINE))
    w.editor.run_id = "t_view"
    w._run(False)
    assert w.editor.proc is not None and "pipeline" in (w.editor.phase or "")
    t0 = time.time()
    while w.editor.proc.poll() is None and time.time() - t0 < 300:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.2)
    assert w.editor.proc.returncode == 0, w.editor._console_tail(40)
    w.refresh()
    assert [c.lamp for c in w.cards][3:] == [M.OK, M.OK], [(c.label, c.lamp, c.line2) for c in w.cards]
    assert w.cards[4].passed == 10
    assert (tmp_path / "runs" / "t_view" / "expanded_project.yaml").exists()
