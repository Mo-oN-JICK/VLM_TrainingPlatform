"""Phase 7 — 편집기 API. UI 전용 실행 경로 없이 코어만 거친다."""

from __future__ import annotations

import os
import shutil

import pytest

from vlm_trainer.core.compiler import compile_project
from vlm_trainer.spec import recipe as recipe_mod
from vlm_trainer.ui import tokens as T
from vlm_trainer.ui.api import Editor, compat_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOLUTION = os.path.join(ROOT, "solutions", "dummy_ecg")
SPEC_DIR = os.path.join(SOLUTION, "projects", "01_dummy")
PROJECT = os.path.join(SPEC_DIR, "project.yaml")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "solutions", "dummy_ecg", "data", "dummy", "index.jsonl")),
    reason="더미 데이터가 없다. python tools/make_dummy_dataset.py 를 먼저 실행하라.",
)


@pytest.fixture
def ed(tmp_path):
    """진짜 Solution을 복사해 연다 — 원본은 편집하지 않는다.

    Procedure는 Solution 아래에 있으므로 프로젝트 디렉터리만 복사하면 찾지 못한다.
    데이터는 컴파일에 필요 없으므로 뺀다.
    """
    work = tmp_path / "dummy_ecg"
    # edit_history.jsonl은 append-only라 앞선 세션의 기록이 딸려오면 안 된다
    shutil.copytree(
        SOLUTION,
        work,
        ignore=shutil.ignore_patterns("data", "runs", "*.lock.yaml", "edit_history*"),
    )
    return Editor.open(str(work / "projects" / "01_dummy" / "project.yaml"))


@pytest.fixture
def ed_with_data(tmp_path):
    """데이터까지 복사한다. 실제로 그래프를 돌리는 테스트 하나만 쓴다."""
    work = tmp_path / "dummy_ecg"
    shutil.copytree(
        SOLUTION, work, ignore=shutil.ignore_patterns("runs", "*.lock.yaml", "edit_history*")
    )
    return Editor.open(str(work / "projects" / "01_dummy" / "project.yaml"))


# ── 호환성 표 ───────────────────────────────────────────────────────────


def test_compat_marks_type_mismatch_with_a_reason(ed):
    row = compat_matrix(ed.compiled)["n_ts:series"]
    # 시계열을 이미지 입력에 놓을 수 없다
    why = row["n_plot_rs:image"]
    assert why and ("base" in why or "TimeSeries" in why)


def test_compat_forbids_upward_wires(ed):
    """위로 향하는 배선은 만들 수 없다 — 이 규약이 순환을 구조적으로 막는다."""
    row = compat_matrix(ed.compiled)["n_stats:stats"]
    assert "위로 향하는 배선" in row["n_exp_ts:subject"]


def test_occupied_input_is_droppable_as_a_replacement(ed):
    """이미 배선이 있는 입력에도 놓을 수 있다. 갈아끼우기가 두 단계면 편집기가 쓸모없다."""
    from vlm_trainer.ui.api import occupied_inputs

    assert "n_ev:stats" in occupied_inputs(ed.compiled)
    assert compat_matrix(ed.compiled)["n_stats:stats"]["n_ev:stats"] == ""


# ── 변경은 게이트를 통과해야 받아들여진다 ───────────────────────────────


def test_incompatible_connect_is_refused_and_the_old_wire_survives(ed):
    """갈아끼우다 실패해도 원래 배선이 남아야 한다."""
    before = ed.compiled.spec_hash
    res = ed.connect("n_ts:series", "n_plot_rs:image")  # 시계열 -> 이미지 입력

    assert not res["ok"]
    assert "TypeError" in res["detail"] or "base" in res["detail"]
    assert any(e.dst == "n_plot_rs:image" and e.src == "n_plot:plot" for e in ed.graph.edges)
    assert ed.compiled.spec_hash == before


def test_connect_replaces_the_existing_wire(ed):
    """입력 포트에는 배선이 하나뿐이다. 새로 놓으면 이전 것이 사라진다."""
    # n_ev:regions 는 지금 시계열 구간(n_exp_ts)에서 온다. 이미지 영역 지목으로 갈아끼운다.
    assert [e.src for e in ed.graph.edges if e.dst == "n_ev:regions"] == ["n_exp_ts:regions"]

    res = ed.connect("p_crop/n_exp:regions", "n_ev:regions")
    assert res["ok"], res.get("detail")
    assert res.get("replaced") == "n_exp_ts:regions"
    assert [e.src for e in ed.graph.edges if e.dst == "n_ev:regions"] == ["p_crop/n_exp:regions"]


def test_disconnect_that_empties_a_required_input_marks_it_incomplete(ed):
    """배선을 끊는 것은 허용하되 저장은 막는다 — 다시 이을 수 있어야 편집이다."""
    res = ed.disconnect("n_ev:stats")
    assert res["ok"] and not ed.valid
    assert "필수 입력" in ed.error
    assert not ed.save()["ok"]

    assert ed.connect("n_stats:stats", "n_ev:stats")["ok"]
    assert ed.valid and ed.save()["ok"]


def test_optional_input_can_be_disconnected(ed):
    res = ed.disconnect("n_ev:regions")  # regions는 optional
    assert res["ok"], res.get("detail")
    assert all(e.dst != "n_ev:regions" for e in ed.graph.edges)


def test_param_change_recompiles_and_changes_the_hash(ed):
    before = ed.compiled.spec_hash
    assert ed.set_param("n_stats", "z_thresh", 4.5)["ok"]
    assert ed.compiled.nodes["n_stats"].params["z_thresh"] == 4.5
    assert ed.compiled.spec_hash != before


def test_bad_param_value_is_refused_not_crashed(ed):
    """노드가 일반 예외를 던져도 편집기는 살아 있고 변경은 거부된다."""
    before = ed.compiled.spec_hash
    res = ed.set_param("n_plot", "size", "이건 크기가 아니다")
    assert not res["ok"] and res["reason"]
    assert ed.compiled.spec_hash == before
    assert ed.compiled.nodes["n_plot"].params["size"] != "이건 크기가 아니다"


# ── 저장 ────────────────────────────────────────────────────────────────


def test_save_writes_the_spec_and_cli_gets_the_same_graph(ed):
    """UI로 만든 그래프를 CLI가 같은 결과로 실행한다 — Phase 7의 완료 조건."""
    assert ed.set_param("n_stats", "z_thresh", 4.5)["ok"]
    assert ed.dirty

    res = ed.save()
    assert res["ok"] and not ed.dirty

    again = compile_project(ed.path)  # CLI가 읽는 것과 같은 경로
    assert again.spec_hash == ed.compiled.spec_hash
    assert again.nodes["n_stats"].params["z_thresh"] == 4.5


def test_editing_does_not_touch_disk_until_save(ed):
    original = open(ed.path, encoding="utf-8").read()
    ed.set_param("n_stats", "z_thresh", 4.5)
    assert open(ed.path, encoding="utf-8").read() == original, "저장 전에 디스크가 바뀌었다"
    ed.save()
    assert open(ed.path, encoding="utf-8").read() != original


# ── HTTP 껍데기 ─────────────────────────────────────────────────────────


def test_api_answers_with_state_and_refusals(ed):
    """편집기가 웹에서 앱으로 옮겨가며 HTTP 라우팅이 사라졌다. 그 계층이 검사하던 것은
    라우팅이 아니라 **api 의 대답**이었으므로, 이제 직접 묻는다."""
    st = ed.state()
    assert st["ok"] and len(st["nodes"]) == 25

    res = ed.connect("n_ts:series", "n_plot_rs:image")
    assert not res["ok"] and res["reason"]

    assert len(ed.library()) >= 25


def test_successful_mutation_returns_fresh_state(ed):
    res = ed.set_param("n_stats", "z_thresh", 2.0)
    assert res["ok"]
    state = ed.state()
    assert state["dirty"] is True
    node = next(n for n in state["nodes"] if n["id"] == "n_stats")
    assert node["params"]["z_thresh"] == 2.0


# ── 파라미터 편집 패널 ──────────────────────────────────────────────────


def test_param_meta_picks_a_widget_and_marks_the_risky_ones(ed):
    from vlm_trainer.ui.api import param_meta

    meta = {m["name"]: m for m in param_meta("ts.plot@1.0.0", ed.compiled.nodes["n_plot"].params)}
    assert meta["size"]["kind"] == "json"          # 튜플은 JSON 입력으로
    assert meta["channel"]["kind"] == "number"
    assert meta["mark_regions"]["kind"] == "bool"

    # 바꾸면 배선 타입이 다시 검사되는 파라미터에는 표식이 붙는다
    assert meta["size"]["type_affecting"] and meta["size"]["overridable"]
    assert not meta["channel"]["type_affecting"]


def test_state_carries_param_meta(ed):
    st = ed.state()
    node = next(n for n in st["nodes"] if n["id"] == "n_stats")
    names = {m["name"] for m in node["param_meta"]}
    assert names == {"z_thresh", "channel"}


def test_type_affecting_change_is_rechecked_by_the_gates(ed):
    """size는 타입에 영향을 준다 — 하류가 안 맞으면 변경이 거부되어야 한다."""
    res = ed.set_param("n_plot", "size", [64, 64])
    assert res["ok"], res.get("detail")
    assert ed.compiled.nodes["n_plot"].output_types["plot"].shape == (64, 64, 3)

    # 그 아래 resize가 448로 고정하므로 여전히 컴파일된다. 반대로 resize를 깨면 거부된다.
    bad = ed.set_param("n_plot_rs", "size", "not a size")
    assert not bad["ok"]


# ── History 되감기 ──────────────────────────────────────────────────────


def test_every_edit_becomes_a_point_in_time(ed):
    assert len(ed.history) == 1 and ed.cursor == 0  # 열기

    ed.set_param("n_stats", "z_thresh", 4.5)
    ed.set_param("n_plot", "channel", 1)
    assert len(ed.history) == 3 and ed.cursor == 2

    view = ed.history_view()
    assert view[0]["label"] == "열기"
    assert "z_thresh" in view[1]["label"]
    assert view[-1]["current"] and not view[0]["current"]


def test_history_stores_a_diff_not_a_wall_of_text(ed):
    """스펙이 텍스트라 편집 하나가 두 줄로 남는다."""
    ed.set_param("n_stats", "z_thresh", 4.5)
    diff = ed.history[-1].diff
    assert diff == ["-    z_thresh: 3.0", "+    z_thresh: 4.5"], diff


def test_undo_and_redo_are_cursor_moves(ed):
    before = ed.compiled.spec_hash
    ed.set_param("n_stats", "z_thresh", 4.5)
    after = ed.compiled.spec_hash
    assert after != before

    assert ed.undo()["ok"]
    assert ed.compiled.spec_hash == before
    assert ed.compiled.nodes["n_stats"].params["z_thresh"] == 3.0

    assert ed.redo()["ok"]
    assert ed.compiled.spec_hash == after

    assert not ed.redo()["ok"]  # 끝에서 한 번 더
    ed.undo()
    assert not ed.undo()["ok"]  # 처음에서 한 번 더


def test_rewind_jumps_to_any_point(ed):
    hashes = [ed.compiled.spec_hash]
    for value in (4.0, 5.0, 6.0):
        ed.set_param("n_stats", "z_thresh", value)
        hashes.append(ed.compiled.spec_hash)

    assert ed.rewind(1)["ok"]
    assert ed.compiled.spec_hash == hashes[1]
    assert ed.compiled.nodes["n_stats"].params["z_thresh"] == 4.0


def test_editing_after_undo_drops_the_redo_branch(ed):
    ed.set_param("n_stats", "z_thresh", 4.0)
    ed.set_param("n_stats", "z_thresh", 5.0)
    assert len(ed.history) == 3

    ed.undo()
    ed.set_param("n_stats", "channel", 1)  # 다른 가지로 갈라진다
    assert len(ed.history) == 3 and ed.cursor == 2
    assert not ed.redo()["ok"]
    assert ed.compiled.nodes["n_stats"].params["z_thresh"] == 4.0


def test_refused_edit_leaves_no_trace_in_history(ed):
    n = len(ed.history)
    assert not ed.set_param("n_plot", "size", "이건 크기가 아니다")["ok"]
    assert len(ed.history) == n


def test_journal_is_written_next_to_the_spec(ed):
    import json as _json

    ed.set_param("n_stats", "z_thresh", 4.5)
    lines = [_json.loads(ln) for ln in open(ed.history_path, encoding="utf-8") if ln.strip()]
    assert len(lines) == 2
    assert lines[-1]["spec_hash"] == ed.compiled.spec_hash
    assert any("z_thresh" in d for d in lines[-1]["diff"])


def test_history_records_and_rewinds(ed):
    assert ed.set_param("n_stats", "z_thresh", 4.5)["ok"]
    assert len(ed.state()["history"]) == 2

    assert ed.undo()["ok"] and ed.state()["cursor"] == 0

    again = ed.undo()
    assert not again["ok"] and "되돌릴 편집이 없다" in again["reason"]

    assert ed.rewind(1)["ok"] and ed.state()["cursor"] == 1


# ── 노드 추가·삭제와 미완성 상태 ────────────────────────────────────────


def test_adding_a_node_is_allowed_but_marks_the_graph_incomplete(ed):
    """노드를 놓고 배선을 잇는 사이의 상태를 허용해야 편집기로 그래프를 만들 수 있다."""
    assert ed.valid

    res = ed.add_node("ts.stats@1.0.0")
    assert res["ok"] and res["id"] == "n_stats_2"
    assert not ed.valid, "미완성인데 valid로 남았다"
    assert "필수 입력" in ed.error
    assert "n_stats_2" in ed.compiled.nodes  # 그려는 볼 수 있어야 한다


def test_incomplete_graph_is_not_saved(ed):
    ed.add_node("ts.stats@1.0.0")
    res = ed.save()
    assert not res["ok"] and "필수 입력" in res["detail"]

    original = open(ed.path, encoding="utf-8").read()
    assert "n_stats_2" not in original


def test_wiring_the_new_node_makes_it_valid_again(ed):
    ed.add_node("ts.stats@1.0.0")
    assert ed.connect("n_ts:series", "n_stats_2:series")["ok"]
    assert not ed.valid  # 아직 아무 Output에도 기여하지 않는다

    assert ed.remove_node("n_stats_2")["ok"]
    assert ed.valid and ed.save()["ok"]


def test_type_errors_are_still_refused_while_incomplete(ed):
    """구조는 미뤄도 타입은 미루지 않는다."""
    ed.add_node("ts.stats@1.0.0")
    res = ed.connect("n_img:image", "n_stats_2:series")  # 이미지를 시계열 입력에
    assert not res["ok"]
    assert "TypeError" in res["detail"] or "base" in res["detail"]


def test_removing_a_node_leaves_the_rest_wired(ed):
    res = ed.remove_node("n_stats")
    assert res["ok"], "삭제는 되어야 한다 — 다시 이을 수 있게"
    assert not ed.valid
    assert all(e.src_node != "n_stats" and e.dst_node != "n_stats" for e in ed.graph.edges)

    assert ed.undo()["ok"]
    assert ed.valid and "n_stats" in ed.compiled.nodes


def test_add_and_remove(ed):
    res = ed.add_node("ts.stats@1.0.0")
    assert res["ok"] and res["id"] == "n_stats_2"
    assert ed.state()["valid"] is False      # 아직 아무것도 안 물린 노드가 있다

    assert ed.remove_node("n_stats_2")["ok"]
    assert ed.state()["valid"] is True


def test_unwired_external_call_node_waits_for_wiring(ed):
    """배선 전에는 물질화 경계의 앞뒤가 정해지지 않는다 — 미루되 저장은 막는다."""
    res = ed.add_node("expert.propose@1.0.0")
    assert res["ok"], res.get("reason")
    assert not ed.valid and not ed.save()["ok"]


# ── Parameter Recipe ────────────────────────────────────────────────────


def test_recipe_overlays_values_without_touching_the_spec(ed):
    """레시피는 값만 덮는다. 화면의 값과 저장될 값이 다르다는 것이 요점이다."""
    assert ed.recipe_select(2)["ok"]

    assert ed.compiled.nodes["n_stats"].params["z_thresh"] == 4.5   # 화면
    assert ed.base.nodes["n_stats"].params["z_thresh"] == 3.0       # 스펙
    assert "n_stats:z_thresh" in ed.state()["overlaid"]

    assert ed.save()["ok"]
    assert "4.5" not in open(ed.path, encoding="utf-8").read(), "레시피 값이 스펙에 스몄다"


def test_procedure_exposed_override_reaches_the_inner_node(ed):
    """`p_crop.max_n`은 Procedure 파일 안의 노드를 가리킨다 — 스펙에는 쓸 수 없는 경로다."""
    assert ed.recipe_select(3)["ok"]
    assert ed.compiled.nodes["p_crop/n_crop"].params["max_n"] == 1
    assert "p_crop/n_crop:max_n" in ed.state()["overlaid"]


def test_editing_an_overlaid_param_changes_the_overlay(ed):
    ed.recipe_select(2)
    assert ed.set_param("n_stats", "z_thresh", 5.5)["ok"]

    assert ed.overlay["n_stats.z_thresh"] == 5.5
    assert ed.base.nodes["n_stats"].params["z_thresh"] == 3.0
    assert not ed.dirty, "오버레이 편집은 프로젝트를 더럽히지 않는다"


def test_editing_a_param_the_recipe_does_not_cover_changes_the_spec(ed):
    ed.recipe_select(2)
    assert ed.set_param("n_plot", "line_width", 2)["ok"]
    assert ed.base.nodes["n_plot"].params["line_width"] == 2
    assert ed.dirty


def test_a_path_outside_the_whitelist_is_refused(ed):
    res = ed.recipe_add_path("n_img.color_space")
    assert not res["ok"]
    assert "덮어쓸 수 없다" in res["detail"]
    assert not ed.overlay, "거부된 축이 남으면 안 된다"


def test_a_captured_recipe_is_written_only_on_save(ed):
    ed.recipe_select(2)
    ed.set_param("n_stats", "z_thresh", 5.5)

    res = ed.recipe_store(None, "captured")
    assert res["ok"] and res["id"] == 5
    before = open(ed.book.path, encoding="utf-8").read()
    assert "captured" not in before, "편집기에서 디스크가 바뀌는 순간은 Save 하나뿐이다"

    written = ed.save()["written"]
    assert any(p.endswith("recipes.yaml") for p in written)
    after = open(ed.book.path, encoding="utf-8").read()
    assert "captured" in after and "5.5" in after


def test_status_is_customized_when_the_project_drifts(ed):
    """활성 레시피가 프로젝트를 더 이상 설명하지 못하면 Customized다 (Mech-Vision 규약)."""
    assert ed.recipe_view()["status"] == "1"

    assert ed.set_param("n_stats", "z_thresh", 9.0)["ok"]
    assert ed.recipe_view()["status"] == recipe_mod.CUSTOMIZED


def test_deleting_the_applied_recipe_takes_the_overlay_off(ed):
    ed.recipe_select(2)
    assert ed.recipe_delete(2)["ok"]
    assert ed.recipe_id is None and not ed.overlay
    assert ed.compiled.nodes["n_stats"].params["z_thresh"] == 3.0


def test_recipe_roundtrip(ed):
    assert ed.recipe_select(3)["ok"] and ed.state()["recipe"]["applied"] == 3

    assert ed.recipe_drop_path("p_crop.max_n")["ok"]

    # 화이트리스트 밖의 파라미터는 레시피 축이 될 수 없다
    assert not ed.recipe_add_path("n_img.color_space")["ok"]

    assert ed.recipe_select(None)["ok"] and ed.state()["recipe"]["applied"] is None


# ── 실행 ────────────────────────────────────────────────────────────────


def test_run_uses_the_same_cli_command(ed):
    """편집기에 전용 실행 경로는 없다. 사람이 터미널에 그대로 쳐도 같아야 한다."""
    ed.extra_modules = ("fixture_nodes",)
    ed.recipe_select(2)
    cmd = ed.run_command(limit=4, debug_output=True)

    assert cmd[1:4] == ["-m", "vlm_trainer.cli.main", "run"]
    assert cmd[4] == ed.path
    assert "--trigger" in cmd and cmd[cmd.index("--trigger") + 1] == "ui"
    assert "--debug-output" in cmd
    assert cmd[cmd.index("--nodes") + 1] == "fixture_nodes"
    # 화면에 보이는 값 그대로 돈다 — 오버레이는 CLI의 --set 으로 넘어간다
    assert "n_stats.z_thresh=4.5" in cmd


def test_run_is_refused_while_the_spec_on_disk_differs(ed):
    ed.set_param("n_stats", "z_thresh", 4.5)
    res = ed.run_start()
    assert not res["ok"] and "저장하지 않은 변경" in res["reason"]
    assert "화면과 다른 그래프가 돈다" in res["detail"]

    assert ed.save()["ok"]
    assert ed.run_command()  # 저장한 뒤에는 막을 이유가 없다


def test_run_is_refused_while_the_graph_is_incomplete(ed):
    ed.add_node("ts.stats@1.0.0")
    res = ed.run_start()
    assert not res["ok"] and "필수 입력" in res["detail"]


def test_progress_snapshot_round_trips(ed):
    from vlm_trainer.engine import runner as runner_mod

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    rep.count("n_stats", runner_mod.SUCCESS)
    rep.node_ms["n_stats"] = 12.5
    rep.processed = 3
    rep.quarantine.append(runner_mod.Quarantined("s1", "n_ev", "이유", "힌트"))

    data = runner_mod.snapshot(rep, run_id="r1", total=4, phase="running")
    back = runner_mod.report_from_snapshot(data)

    assert back.states_of("n_stats") == {"success": 1}
    assert back.node_ms["n_stats"] == 12.5
    assert back.quarantine[0].node_id == "n_ev"
    from vlm_trainer.ui.layout import state_of

    state, extra = state_of(back, "n_stats")
    assert state == "success" and "1건" in extra


def test_run_state_reads_the_snapshot_a_run_leaves(ed, tmp_path):
    import json as _json

    from vlm_trainer.engine import runner as runner_mod

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    for nid in ed.compiled.order:
        rep.count(nid, runner_mod.SUCCESS)
    rep.processed = 2

    ed.progress_path = str(tmp_path / "progress.json")
    with open(ed.progress_path, "w", encoding="utf-8") as fh:
        _json.dump(runner_mod.snapshot(rep, run_id="r1", total=2, phase="done"), fh)

    st = ed.run_state()
    assert st["running"] is False and st["phase"] == "done"
    assert st["processed"] == 2
    assert st["states"]["n_stats"]["state"] == "success"


def test_a_half_written_snapshot_is_ignored(ed, tmp_path):
    """진행 파일은 원자 교체로 쓰이지만, 읽는 쪽도 깨진 내용에 죽지 않아야 한다."""
    ed.progress_path = str(tmp_path / "progress.json")
    with open(ed.progress_path, "w", encoding="utf-8") as fh:
        fh.write('{"phase": "run')

    st = ed.run_state()
    assert st["states"] == {} and st["running"] is False


def test_the_editor_actually_runs_the_graph(ed_with_data, tmp_path, monkeypatch):
    """끝에서 끝까지 — 편집기가 띄운 프로세스가 진행 파일을 남기고 상태가 칠해진다.

    저장소 밖에서 돌린다. 하위 프로세스가 CWD에 기대 임포트하면 여기서 걸린다.
    """
    import time as _time

    ed = ed_with_data
    monkeypatch.chdir(tmp_path)
    assert ed.run_start(limit=2)["ok"]

    for _ in range(1200):  # 최대 120초. 전체 실행 중에는 앞선 테스트의 프로세스와 겹친다
        st = ed.run_state()
        if not st["running"]:
            break
        _time.sleep(0.1)
    else:
        raise AssertionError("끝나지 않았다")

    assert st["phase"] == "done", st.get("console", "")
    assert st["exit"] == 0, st.get("console", "")
    assert st["states"]["n_answer"]["state"] in ("success", "cached")


def test_a_stopped_run_is_not_reported_as_a_failure(ed_with_data, tmp_path, monkeypatch):
    """사람이 멈춘 것과 죽은 것은 다르다. 빨간 글씨로 같이 묶으면 신호가 죽는다."""
    import time as _time

    ed = ed_with_data
    monkeypatch.chdir(tmp_path)
    assert ed.run_start(limit=24)["ok"]
    assert ed.run_stop()["ok"]

    for _ in range(300):
        st = ed.run_state()
        if not st["running"]:
            break
        _time.sleep(0.1)

    assert st["stopped"] is True
    assert "console" not in st, "중지는 실패가 아니므로 콘솔 꼬리를 들이밀지 않는다"


# ── 세션을 넘는 History ─────────────────────────────────────────────────


def test_history_survives_reopening(ed):
    """편집기를 닫았다 열어도 지난 시점으로 되감을 수 있어야 한다."""
    ed.set_param("n_stats", "z_thresh", 4.5)
    ed.set_param("n_plot", "line_width", 3)
    assert ed.save()["ok"]
    labels_before = [h["label"] for h in ed.history_view()]

    again = Editor.open(ed.path)
    labels = [h["label"] for h in again.history_view()]
    assert labels == labels_before, "지난 세션의 시점이 그대로 되살아나야 한다"
    assert all(h["past"] for h in again.history_view()), "되살린 시점은 표식이 붙는다"

    # 디스크의 스펙이 마지막 시점과 같으면 '열기'를 겹쳐 적지 않는다
    assert labels.count("열기") == 1
    assert again.compiled.nodes["n_stats"].params["z_thresh"] == 4.5

    assert again.rewind(0)["ok"], "첫 시점으로 되감을 수 있어야 한다"
    assert again.compiled.nodes["n_stats"].params["z_thresh"] == 3.0


def test_snapshots_are_content_addressed(ed):
    """같은 상태로 돌아오면 파일이 늘지 않는다."""
    ed.set_param("n_stats", "z_thresh", 4.5)
    ed.set_param("n_stats", "z_thresh", 3.0)  # 원래 값으로
    files = sorted(os.listdir(ed.history_store))
    assert len(files) == 2, files  # 열기 시점과 4.5 시점 둘뿐이다


def test_a_snapshot_that_is_gone_is_not_offered(ed):
    """되감을 수 없는 항목을 목록에 두면 눌러도 되는 것과 아닌 것이 섞인다."""
    ed.set_param("n_stats", "z_thresh", 4.5)
    ed.save()
    for name in os.listdir(ed.history_store):
        os.remove(os.path.join(ed.history_store, name))

    again = Editor.open(ed.path)
    assert [h["label"] for h in again.history_view()] == ["열기"]
    assert not again.history_view()[0]["past"]


def test_a_hand_edited_spec_gets_its_own_point(ed):
    """저널의 마지막 시점과 디스크가 다르면 지금 상태를 새 시점으로 적는다."""
    ed.set_param("n_stats", "z_thresh", 4.5)
    ed.save()

    text = open(ed.path, encoding="utf-8").read().replace("z_thresh: 4.5", "z_thresh: 6.0")
    open(ed.path, "w", encoding="utf-8").write(text)

    again = Editor.open(ed.path)
    view = again.history_view()
    assert view[-1]["label"] == "열기" and not view[-1]["past"]
    assert again.compiled.nodes["n_stats"].params["z_thresh"] == 6.0
    assert again.rewind(0)["ok"]


def test_the_journal_stays_readable(ed):
    """저널은 사람이 읽는 diff다. 본문은 옆 저장소에 따로 둔다."""
    import json as _json

    ed.set_param("n_stats", "z_thresh", 4.5)
    lines = [_json.loads(ln) for ln in open(ed.history_path, encoding="utf-8") if ln.strip()]
    assert set(lines[-1]) == {"at", "label", "spec_hash", "diff"}
    assert lines[-1]["diff"] == ["-    z_thresh: 3.0", "+    z_thresh: 4.5"]
    assert lines[-1]["at"].startswith("20"), "세션을 넘으려면 날짜가 있어야 한다"


# ── Debug Output 이미지 ─────────────────────────────────────────────────


def test_previews_are_served_only_from_this_runs_folder(ed, tmp_path, monkeypatch):
    """경로를 그대로 실어 보내면 서버가 아무 파일이나 내주는 문이 된다."""
    monkeypatch.chdir(tmp_path)
    ed.run_id = "ui_test"
    os.makedirs(ed.preview_dir, exist_ok=True)
    with open(os.path.join(ed.preview_dir, "n_img.png"), "wb") as fh:
        fh.write(b"PNG-ish")
    with open(tmp_path / "secret.txt", "w", encoding="utf-8") as fh:
        fh.write("남의 파일")

    assert ed.preview_file("n_img.png") == b"PNG-ish"
    for probe in ("../../secret.txt", r"..\..\secret.txt", "n_img.txt", "", "secret.txt"):
        assert ed.preview_file(probe) is None, probe


def test_preview_paths_become_urls_not_disk_paths(ed):
    urls = ed._preview_urls(
        {"n_img": {"kind": "image", "text": "t", "image_path": r"C:\runs\x\preview\n_img.png"},
         "n_kb": {"kind": "text", "text": "t", "image_path": ""}}
    )
    assert urls["n_img"]["image"] == "/preview/n_img.png"
    assert "image_path" not in urls["n_img"], "디스크 경로를 페이지로 내보내지 않는다"
    assert urls["n_kb"]["image"] == ""


def test_the_run_writes_preview_images_when_the_toggle_is_on(ed_with_data, tmp_path, monkeypatch):
    """Debug Output이 그림을 보여주지 못하면 크롭이 어긋났는지 알 수 없다."""
    import time as _time

    ed = ed_with_data
    monkeypatch.chdir(tmp_path)
    assert ed.run_start(limit=1, debug_output=True)["ok"]

    for _ in range(600):
        st = ed.run_state()
        if not st["running"]:
            break
        _time.sleep(0.1)

    assert st["phase"] == "done", st.get("console", "")
    imgs = {k: v["image"] for k, v in st["previews"].items() if v.get("image")}
    assert imgs, "이미지 미리보기가 하나도 없다"
    assert ed.preview_file(os.path.basename(imgs["n_img"])), "서버가 그 파일을 못 찾는다"


def test_no_preview_images_when_the_toggle_is_off(ed_with_data, tmp_path, monkeypatch):
    """토글이 꺼져 있으면 **생성조차 하지 않는다** (Mech-Vision 규약)."""
    import time as _time

    ed = ed_with_data
    monkeypatch.chdir(tmp_path)
    assert ed.run_start(limit=1, debug_output=False)["ok"]

    for _ in range(600):
        st = ed.run_state()
        if not st["running"]:
            break
        _time.sleep(0.1)

    assert st["phase"] == "done"
    assert not st["previews"]
    assert not os.path.isdir(ed.preview_dir) or not os.listdir(ed.preview_dir)


# ── Sample Space ────────────────────────────────────────────────────────


def test_the_panel_reads_the_index_rather_than_echoing_it(ed_with_data):
    """값을 보여주는 데서 그치면 key 하나가 어긋난 것을 실행 때까지 모른다."""
    ed = ed_with_data
    v = ed.sample_space_view()
    assert v["error"] == ""
    assert v["rows"] == 22 and v["split_counts"] == {"train": 11, "val": 11}
    assert v["columns"][:2] == ["sample_id", "patient_id"]
    assert v["key"] == "sample_id" and "quality_flag" in v["filter"]


def test_changing_the_filter_changes_what_the_probe_reports(ed_with_data):
    ed = ed_with_data
    assert ed.set_sample_space("filter", "")["ok"]
    assert ed.sample_space_view()["rows"] == 24  # 필터가 걸러내던 2건이 돌아온다
    assert ed.dirty


def test_a_key_that_is_not_in_the_index_is_refused(ed_with_data):
    ed = ed_with_data
    res = ed.set_sample_space("key", "없는열")
    assert not res["ok"] and "키 컬럼" in res["detail"]
    assert ed.graph.sample_space.key == "sample_id"
    assert ed.sample_space_view()["rows"] == 22


def test_a_missing_index_file_is_refused(ed_with_data):
    ed = ed_with_data
    res = ed.set_sample_space("index", "../../data/dummy/nope.jsonl")
    assert not res["ok"] and "인덱스 파일이 없다" in res["detail"]
    assert ed.sample_space_view()["rows"] == 22


def test_a_refused_sample_space_edit_leaves_no_trace(ed_with_data):
    """기록한 뒤에 되돌리면 저널에는 이미 줄이 들어간 뒤다 — 거부된 편집이 History에 남는다."""
    ed = ed_with_data
    import json as _json

    before = len(ed.history)
    ed.set_sample_space("key", "없는열")
    ed.set_sample_space("index", "../../data/dummy/nope.jsonl")

    assert len(ed.history) == before
    lines = [_json.loads(ln) for ln in open(ed.history_path, encoding="utf-8") if ln.strip()]
    assert not any("없는열" in ln["label"] or "nope" in ln["label"] for ln in lines)


def test_unknown_sample_space_fields_are_refused(ed_with_data):
    ed = ed_with_data
    res = ed.set_sample_space("nodes", [])
    assert not res["ok"] and "항목은 없다" in res["reason"]
    res = ed.set_sample_space("splits", "문자열이 아니라 객체여야 한다")
    assert not res["ok"] and "객체" in res["reason"]


def test_the_sample_space_edit_survives_a_save(ed_with_data):
    ed = ed_with_data
    assert ed.set_sample_space("filter", "")["ok"]
    assert ed.save()["ok"]

    again = Editor.open(ed.path)
    assert again.sample_space_view()["rows"] == 24
    assert again.graph.sample_space.filter == ""


# ── 물질화와 학습 ───────────────────────────────────────────────────────


def test_one_button_is_one_cli_command(ed):
    """편집기가 물질화와 학습을 엮어 돌리면 CLI에 없는 경로가 하나 생긴다."""
    ed.extra_modules = ("fixture_nodes",)
    ed.run_id = "ui_x"
    for sub in ("materialize", "train"):
        cmd = ed._base_command(sub)
        assert cmd[1:5] == ["-m", "vlm_trainer.cli.main", sub, ed.path]
        assert cmd[cmd.index("--run-id") + 1] == "ui_x"
        assert cmd[cmd.index("--nodes") + 1] == "fixture_nodes"


def test_training_before_materializing_says_so(ed):
    res = ed.train_start()
    assert not res["ok"] and "Materialize" in res["reason"]
    assert "vlmt materialize" in res["detail"], "터미널로도 할 수 있다는 것을 말한다"


def test_materialize_and_train_refuse_unsaved_changes(ed):
    ed.set_param("n_stats", "z_thresh", 4.5)
    for res in (ed.materialize_start(), ed.train_start()):
        assert not res["ok"] and "저장하지 않은 변경" in res["reason"]


def test_materialize_refuses_an_incomplete_graph(ed):
    ed.add_node("ts.stats@1.0.0")
    assert not ed.materialize_start()["ok"]


def test_only_one_thing_runs_at_a_time(ed_with_data, tmp_path, monkeypatch):
    ed = ed_with_data
    monkeypatch.chdir(tmp_path)
    assert ed.materialize_start()["ok"]
    second = ed.materialize_start()
    assert not second["ok"] and "돌고 있다" in second["reason"]
    ed.run_stop()


def test_the_editor_materializes_and_then_trains(ed_with_data, tmp_path, monkeypatch):
    """끝에서 끝까지 — 두 버튼이 두 CLI 명령을 띄우고 학습이 완주한다."""
    import time as _time

    ed = ed_with_data
    monkeypatch.chdir(tmp_path)

    def wait():
        for _ in range(1200):  # 최대 120초
            st = ed.run_state()
            if not st["running"]:
                return st
            _time.sleep(0.1)
        raise AssertionError("끝나지 않았다")

    assert ed.materialize_start()["ok"]
    st = wait()
    assert st["exit"] == 0, st.get("console", "")
    assert os.path.isdir(os.path.join(tmp_path, "runs", ed.run_id, "materialized"))

    assert ed.train_start()["ok"]
    st = wait()
    assert st["exit"] == 0, st.get("console", "")
    assert st["kind"] == "train"
    assert st["train"]["step"] > 0 and st["train"]["stage"]
    assert os.path.exists(
        os.path.join(tmp_path, "runs", ed.run_id, "train", "inference_contract.json")
    )


# ── 물질화 경계와 실행 프로파일 ─────────────────────────────────────────


def test_an_empty_boundary_is_not_a_way_to_skip_the_check(ed):
    """검사를 끄는 방법이 '경계를 안 적는 것'이어서는 안 된다."""
    assert ed.set_boundary("n_sample", False)["ok"]
    assert not ed.valid
    assert "물질화 경계가 비어 있다" in ed.error
    assert not ed.save()["ok"]

    assert ed.set_boundary("n_sample", True)["ok"]
    assert ed.valid


def test_the_check_sees_inside_procedures(ed):
    """바깥 그래프의 배선만 보면 Procedure 안에서 외부 모델을 부르는 노드가 빠진다."""
    ed.set_boundary("n_sample", False)
    assert "p_crop/n_exp" in ed.error and "n_exp_ts" in ed.error


def test_a_graph_that_does_not_train_is_not_asked_about_a_boundary():
    """루프가 없으면 '루프 안에서 돈다'는 위험 자체가 없다."""
    from vlm_trainer.core.compiler import compile_project

    demo = os.path.join(ROOT, "tests", "data", "solution", "projects", "01_demo", "project.yaml")
    if not os.path.exists(demo):
        pytest.skip("데모 스펙이 없다")
    cg = compile_project(demo)  # 경계가 있든 없든 학습 노드가 없으면 통과한다
    assert cg.order


def test_toggling_the_boundary_is_recorded_like_any_edit(ed):
    before = len(ed.history)
    ed.set_boundary("n_sample", False)
    assert len(ed.history) == before + 1
    assert "물질화 경계에서 제거" in ed.history_view()[-1]["label"]
    assert ed.undo()["ok"] and ed.valid


def test_the_same_node_is_not_added_twice(ed):
    res = ed.set_boundary("n_sample", True)
    assert not res["ok"] and "이미 경계에 있다" in res["reason"]
    res = ed.set_boundary("없는노드", True)
    assert not res["ok"] and "그런 노드가 없다" in res["reason"]


def test_the_profile_can_be_switched_from_the_editor(ed):
    assert ed.graph.runtime_profile == "windows_single_gpu"
    assert "linux_multi_gpu" in ed.profiles()

    assert ed.set_profile("linux_multi_gpu")["ok"]
    assert ed.graph.runtime_profile == "linux_multi_gpu"
    assert ed.save()["ok"]
    assert Editor.open(ed.path).graph.runtime_profile == "linux_multi_gpu"


def test_expanding_a_box_does_not_touch_the_spec(ed):
    """접기는 보는 방식일 뿐이다. 스펙도 History도 건드리지 않는다."""
    before_hash = ed.compiled.spec_hash
    before_hist = len(ed.history)

    assert ed.toggle_expand("p_crop")["ok"]
    assert "p_crop" in ed.expanded
    assert ed.compiled.spec_hash == before_hash
    assert len(ed.history) == before_hist
    assert not ed.dirty

    assert ed.toggle_expand("p_crop")["ok"]
    assert "p_crop" not in ed.expanded


def test_an_unknown_procedure_says_what_exists(ed):
    res = ed.toggle_expand("p_없음")
    assert not res["ok"] and "p_crop" in res["reason"]


def test_the_gates_see_the_expanded_graph_either_way(ed):
    """접혀 있든 펼쳐져 있든 컴파일 결과는 같다 — 접기가 게이트를 건드리면 안 된다."""
    folded = ed.compiled.spec_hash
    ed.toggle_expand("p_crop")
    assert ed.compiled.spec_hash == folded
    assert len(ed.compiled.nodes) == 25, "게이트는 언제나 펼쳐진 그래프를 본다"


# ── 캔버스 자리 ─────────────────────────────────────────────────────────


def test_moving_a_box_does_not_touch_the_spec(ed):
    """자리는 무엇이 실행되는지와 무관하다. spec_hash 에 섞이면 상자를 옮긴 것만으로
    캐시가 통째로 무효가 된다."""
    before = ed.compiled.spec_hash
    before_hist = len(ed.history)

    assert ed.move_node("n_stats", 320, 480)["ok"]
    assert ed.layout["n_stats"] == (320, 480)
    assert ed.compiled.spec_hash == before
    assert len(ed.history) == before_hist
    assert not ed.dirty


def test_the_position_survives_reopening(ed):
    ed.move_node("n_stats", 160, 240)
    assert os.path.exists(ed.layout_path), "layout.yaml 이 스펙 옆에 있어야 한다"

    again = Editor.open(ed.path)
    assert again.layout["n_stats"] == (160, 240)

    # 스펙 파일에는 좌표가 없다
    assert "160" not in open(ed.path, encoding="utf-8").read().split("nodes:")[0]


def test_resetting_drops_the_file(ed):
    ed.move_node("n_stats", 64, 64)
    assert os.path.exists(ed.layout_path)

    assert ed.reset_layout()["ok"]
    assert ed.layout == {} and not os.path.exists(ed.layout_path)


def test_an_unknown_box_is_refused(ed):
    res = ed.move_node("없는상자", 10, 10)
    assert not res["ok"] and "그런 상자가 없다" in res["reason"]


def test_a_moved_box_keeps_its_place_and_the_rest_flow_around_it(ed):
    from vlm_trainer.ui.layout import fold
    from vlm_trainer.ui import layout as layout_mod
    from vlm_trainer.ui import render as render_mod

    ed.move_node("n_stats", 640, 720)
    shown, edges = fold(ed.compiled)
    placed = layout_mod._layout(shown, edges, {k: tuple(v) for k, v in ed.layout.items()})

    assert (placed["n_stats"].x, placed["n_stats"].y) == (640, 720)
    assert (placed["n_img"].x, placed["n_img"].y) != (640, 720), "나머지는 자동 배치 그대로다"


# ── 실행 중인 노드를 화면이 따라간다 ──────────────────────────────────────


def test_snapshot_carries_the_node_being_run(ed):
    """누계(node_ms)만으로는 "어디까지 왔나"에 답이 안 된다.

    느린 노드 앞에서 멈춘 것인지 그 노드가 도는 중인지 구별되지 않기 때문이다.
    """
    import time as _t

    from vlm_trainer.engine import runner as runner_mod
    from vlm_trainer.ui.layout import state_of

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    rep.active, rep.active_since = "n_stats", _t.perf_counter() - 3.0

    data = runner_mod.snapshot(rep, run_id="r1", total=4, phase="running")
    assert data["active"] == "n_stats"
    assert data["active_ms"] >= 2900

    back = runner_mod.report_from_snapshot(data)
    state, extra = state_of(back, "n_stats")
    # 카운트가 아직 0이어도(첫 샘플의 첫 통과) 현재 위치로 보여야 한다
    assert state == "running" and extra.endswith("s")


def test_a_finished_run_highlights_nothing(ed):
    """다 끝난 그래프에 노드 하나가 계속 빛나고 있으면 그것이 마지막으로 돈 노드인지
    지금 도는 노드인지 화면만 보고는 알 수 없다."""
    from vlm_trainer.engine import runner as runner_mod

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    rep.active, rep.active_since = "n_stats", 0.0

    for phase in ("done", "aborted"):
        data = runner_mod.snapshot(rep, run_id="r1", total=4, phase=phase)
        assert data["active"] == "", phase
        assert data["active_ms"] == 0.0, phase


def test_elapsed_time_shows_on_finished_nodes(ed):
    """실패한 노드도 5ms 만에 터진 것과 40초를 쓰고 터진 것은 원인이 다르다."""
    from vlm_trainer.engine import runner as runner_mod
    from vlm_trainer.ui.layout import state_of

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    rep.count("n_stats", runner_mod.SUCCESS)
    rep.node_ms["n_stats"] = 1234.0
    rep.count("n_ev", runner_mod.FAILED)
    rep.node_ms["n_ev"] = 42_000.0

    assert state_of(rep, "n_stats") == ("success", "1건 · 1.2s")
    assert state_of(rep, "n_ev") == ("failed", "1건 · 42.0s")


def test_run_state_answers_with_the_ids_the_canvas_draws(ed, tmp_path):
    """접힌 Procedure는 안쪽 노드 id로 그려져 있지 않다.

    compiled.order를 그대로 내보내면 상자가 실행 내내 아무 색도 바뀌지 않는다.
    """
    import json as _json

    from vlm_trainer.engine import runner as runner_mod
    from vlm_trainer.ui.layout import fold

    shown, _ = fold(ed.compiled, ed.expanded)

    rep = runner_mod.RunReport(order=list(ed.compiled.order))
    rep.active, rep.active_since = ed.compiled.order[0], 0.0

    ed.progress_path = str(tmp_path / "progress.json")
    with open(ed.progress_path, "w", encoding="utf-8") as fh:
        _json.dump(runner_mod.snapshot(rep, run_id="r1", total=2, phase="running"), fh)

    st = ed.run_state()
    assert set(st["states"]) == set(shown), "캔버스에 없는 id로 답하면 칠할 곳이 없다"
    assert st["active"] in shown
    assert st["states"][st["active"]]["state"] == "running"


# ── Procedure 노출 파라미터 편집 ─────────────────────────────────────────


def test_a_procedures_exposed_param_is_editable(ed):
    """상자로 접힌 Procedure의 노출 파라미터를 우측 패널에서 고칠 수 있어야 한다.

    `GraphModel.node()`가 NodeInstance만 뒤져서, 프로시저 상자는 이름조차 찾지 못하고
    KeyError로 거부됐다. 화면은 입력칸을 내주는데 서버는 매번 되돌리는 상태였다.
    """
    before = dict(ed.graph.procedures[0].params)
    assert ed.graph.procedures[0].id == "p_crop"

    res = ed.set_param("p_crop", "topk", 3)
    assert res["ok"], res.get("detail")
    assert ed.graph.procedures[0].params["topk"] == 3
    assert before["topk"] != 3, "픽스처가 이미 3이면 이 테스트는 아무것도 증명하지 못한다"


def test_the_panel_addresses_exposed_params_by_the_box(ed):
    """패널이 내보내는 주소가 스펙이 받는 주소와 같아야 한다.

    안쪽 노드(`p_crop/n_expert`)는 Procedure 파일에 있고 프로젝트가 건드릴 수 있는 것이
    아니다. 프로젝트는 노출 이름을 인스턴스에 적는다(`p_crop.topk`).
    """
    import re

    from vlm_trainer.ui.render import render

    html_out = render(ed.compiled, editable=True, editor=ed)
    sent = set(re.findall(r"vlmtParam\('([^']+)','([^']+)'", html_out))

    boxes = set(ed.graph.ids)
    for nid, param in sent:
        assert nid in boxes, f"패널이 스펙에 없는 주소로 보낸다: {nid}.{param}"
        # 실제로 받아들여지는지까지 확인한다 — 주소만 맞고 거부되면 소용없다
        cur = ed.set_param(nid, param, _same_value(ed, nid, param))
        assert cur["ok"], f"{nid}.{param} 거부됨: {cur.get('detail')}"


def _same_value(ed, nid, param):
    """지금 값 그대로. 값을 바꾸지 않고 경로만 시험하려는 것이다."""
    inst = ed.graph.instance(nid)
    return inst.params.get(param)


def test_a_rejected_procedure_edit_leaves_the_graph_alone(ed):
    """거부된 편집이 그래프에 남으면, 화면은 '거부됨'인데 값은 바뀌어 있게 된다."""
    before = dict(ed.graph.procedures[0].params)
    res = ed.set_param("p_crop", "topk", "셋")  # 숫자 자리에 문자열
    if res["ok"]:
        pytest.skip("이 파라미터는 문자열도 받는다 — 거부 경로를 시험할 수 없다")
    assert ed.graph.procedures[0].params == before


def test_an_unknown_box_says_which_boxes_exist(ed):
    """에러 메시지는 KeyError 한 줄이 아니라 무엇을 해야 하는지 말해야 한다."""
    res = ed.set_param("없는상자", "x", 1)
    assert not res["ok"]
    assert "없는상자" in res["detail"]
    assert "있는 상자" in res["detail"]
    assert "p_crop" in res["detail"]


def test_a_misspelled_param_name_is_refused_not_drafted(ed):
    """draft가 미루는 것은 완결성이지 이름이 틀린 파라미터가 아니다.

    입력이 없는 노드(Input 등)는 draft 관용 분기에 언제나 걸려서, 오타 하나가
    조용히 통과하고 그래프만 valid=False로 남았다. 편집기는 ok를 돌려주는데
    저장과 실행은 거부되는, 설명이 안 되는 상태다.
    """
    before = dict(ed.graph.node("n_schema").params)

    res = ed.set_param("n_schema", "no_such_param", 1)

    assert not res["ok"], "오타가 통과했다"
    assert "no_such_param" in res["detail"]
    assert ed.graph.node("n_schema").params == before, "거부됐는데 값이 남았다"
    assert ed.valid, "거부된 편집이 그래프를 망가진 채로 두었다"


# ── 작업에 들어선 뒤 흐른 시간 ───────────────────────────────────────────


def _running(ed, tmp_path, *, processed: int, total: int, ago: float):
    """진행 중인 작업 하나를 흉내낸다. 시계를 믿지 않도록 값을 직접 놓는다."""
    import json as _json
    import time as _t

    from vlm_trainer.engine import runner as runner_mod

    class _Alive:
        def poll(self):
            return None

    ed.proc = _Alive()
    ed.launched_at = _t.time() - ago
    ed.finished_at = 0.0
    ed.progress_path = str(tmp_path / "progress.json")

    first = ed.compiled.order[0]
    rep = runner_mod.RunReport(order=list(ed.compiled.order), started=_t.time() - ago)
    rep.processed = processed
    rep.active, rep.active_since = first, _t.perf_counter() - 4.0
    rep.node_ms[first] = 8200.0
    with open(ed.progress_path, "w", encoding="utf-8") as fh:
        _json.dump(runner_mod.snapshot(rep, run_id="r", total=total, phase="running"), fh)
    return first


def test_elapsed_time_counts_from_when_the_job_started(ed, tmp_path):
    """노드별 누계를 다 더해도 이 값이 나오지 않는다.

    캐시 적중, 샘플 적재, shard 쓰기처럼 어느 노드에도 속하지 않는 시간이 있다.
    """
    _running(ed, tmp_path, processed=20, total=110, ago=30.0)
    st = ed.run_state()
    assert 29_000 <= st["elapsed_ms"] <= 31_000


def test_elapsed_time_stops_when_the_job_does(ed, tmp_path):
    """다 끝난 작업의 숫자가 계속 올라가면 그것은 경과 시간이 아니라 시계다."""
    import time as _t

    class _Dead:
        returncode = 0

        def poll(self):
            return 0

    _running(ed, tmp_path, processed=110, total=110, ago=12.0)
    ed.proc = _Dead()
    first = ed.run_state()["elapsed_ms"]
    _t.sleep(0.05)
    again = ed.run_state()["elapsed_ms"]
    assert first == again, "끝난 뒤에도 경과 시간이 흐른다"


def test_remaining_time_is_withheld_until_the_rate_means_something(ed, tmp_path):
    """처음 한두 건으로 "남은 40분"을 띄우면 그 수를 믿고 자리를 뜨게 된다."""
    _running(ed, tmp_path, processed=3, total=110, ago=30.0)
    assert ed.run_state()["eta_ms"] == 0.0

    _running(ed, tmp_path, processed=20, total=110, ago=30.0)
    eta = ed.run_state()["eta_ms"]
    # 20건에 30초 -> 남은 90건은 135초
    assert 130_000 <= eta <= 140_000


def test_a_running_node_shows_this_pass_and_the_running_total(ed, tmp_path):
    """앞만 있으면 이 노드가 전체에서 얼마나 무거운지 모르고,
    뒤만 있으면 지금 한 건이 유난히 오래 걸리는 중인지 알 수 없다."""
    first = _running(ed, tmp_path, processed=20, total=110, ago=30.0)
    st = ed.run_state()
    extra = st["states"][st["active"]]["extra"]
    assert extra.startswith("4."), extra          # 이번에 들어가서 4초
    assert "누계 8.2s" in extra, extra            # 지금까지 8.2초
