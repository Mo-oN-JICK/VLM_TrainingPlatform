"""Phase 7 — 편집기 API. UI 전용 실행 경로 없이 코어만 거친다."""

from __future__ import annotations

import os
import shutil

import pytest

from vlm_trainer.core.compiler import compile_project
from vlm_trainer.spec import recipe as recipe_mod
from vlm_trainer.ui import server as server_mod
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
    shutil.copytree(SOLUTION, work, ignore=shutil.ignore_patterns("data", "runs", "*.lock.yaml"))
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


def test_router_returns_state_and_errors(ed):
    code, payload = server_mod.handle(ed, "/api/state", {})
    assert code == 200 and payload["ok"] and len(payload["nodes"]) == 25

    code, payload = server_mod.handle(ed, "/api/connect", {"from": "n_ts:series", "to": "n_plot_rs:image"})
    assert code == 409 and not payload["ok"] and payload["reason"]

    code, payload = server_mod.handle(ed, "/api/nope", {})
    assert code == 404

    code, payload = server_mod.handle(ed, "/api/library", {})
    assert code == 200 and len(payload["nodes"]) >= 25


def test_successful_mutation_returns_fresh_state(ed):
    code, payload = server_mod.handle(ed, "/api/param", {"node": "n_stats", "param": "z_thresh", "value": 2.0})
    assert code == 200 and payload["ok"]
    assert payload["state"]["dirty"] is True
    node = next(n for n in payload["state"]["nodes"] if n["id"] == "n_stats")
    assert node["params"]["z_thresh"] == 2.0


def test_editor_page_carries_the_compat_table(ed):
    from vlm_trainer.ui import render as render_mod

    page = render_mod.render_editor(ed)
    assert "window.COMPAT" in page and "window.EDITABLE" in page
    assert 'data-ref="n_ts:series"' in page
    assert "okdrop" in page and "nodrop" in page
    assert "vlmtSave" in page


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


def test_editor_page_has_a_params_panel_per_node(ed):
    from vlm_trainer.ui import render as render_mod

    page = render_mod.render_editor(ed)
    assert page.count('class="params"') == len(ed.compiled.order)
    assert 'data-node="n_stats"' in page
    assert "vlmtParam(" in page and "vlmtSelect(" in page
    # 선택된 카드는 실측한 선택 색을 쓴다
    assert T.NODE["bg_selected"] in page and T.NODE["border_selected"] in page


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


def test_router_exposes_history(ed):
    code, payload = server_mod.handle(ed, "/api/param", {"node": "n_stats", "param": "z_thresh", "value": 4.5})
    assert code == 200 and len(payload["state"]["history"]) == 2

    code, payload = server_mod.handle(ed, "/api/undo", {})
    assert code == 200 and payload["state"]["cursor"] == 0

    code, payload = server_mod.handle(ed, "/api/undo", {})
    assert code == 409 and "되돌릴 편집이 없다" in payload["reason"]

    code, payload = server_mod.handle(ed, "/api/rewind", {"index": 1})
    assert code == 200 and payload["state"]["cursor"] == 1


def test_history_panel_renders_points_and_diffs(ed):
    from vlm_trainer.ui import render as render_mod

    ed.set_param("n_stats", "z_thresh", 4.5)
    page = render_mod.render_editor(ed)
    assert page.count('class="hrow') == 2
    assert "vlmtRewind(" in page and "vlmtUndo()" in page
    assert "z_thresh: 4.5" in page


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


def test_library_panel_offers_click_and_drag(ed):
    from vlm_trainer.ui import render as render_mod

    page = render_mod.render_editor(ed)
    assert page.count('class="libnode"') >= 25
    assert "vlmtAdd(" in page and "vlmtRemove(" in page
    # 끌어다 놓기는 배선 드래그와 같은 마우스 이벤트를 쓴다
    assert "libdrag" in page and "candrop" in page
    assert "dataTransfer" not in page, "손잡이는 한 벌만 둔다"

    viewer = render_mod.render(ed.compiled)
    assert "vlmtAdd(" not in viewer, "뷰어에는 편집 손잡이가 없어야 한다"


def test_router_add_and_remove(ed):
    code, payload = server_mod.handle(ed, "/api/add", {"type": "ts.stats@1.0.0"})
    assert code == 200 and payload["id"] == "n_stats_2"
    assert payload["state"]["valid"] is False

    code, payload = server_mod.handle(ed, "/api/remove", {"node": "n_stats_2"})
    assert code == 200 and payload["state"]["valid"] is True


def test_unwired_external_call_node_waits_for_wiring(ed):
    """배선 전에는 물질화 경계의 앞뒤가 정해지지 않는다 — 미루되 저장은 막는다."""
    res = ed.add_node("expert.propose@1.0.0")
    assert res["ok"], res.get("reason")
    assert not ed.valid and not ed.save()["ok"]


def test_every_handler_the_page_calls_is_defined(ed):
    """인라인 onclick이 부르는 함수가 실제로 선언되어 있어야 한다."""
    import re

    from vlm_trainer.ui import render as render_mod

    page = render_mod.render_editor(ed)
    assert "async async" not in page, "패치가 키워드를 겹쳐 스크립트 전체가 죽는다"

    called = set(re.findall(r'onclick="(vlmt\w+)\(', page))
    declared = set(re.findall(r"(?:async )?function (vlmt\w+)\(", page))
    assert called, "편집기인데 손잡이가 하나도 없다"
    assert called <= declared, f"선언되지 않은 핸들러: {sorted(called - declared)}"


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


def test_recipe_panel_says_the_values_are_not_saved(ed):
    from vlm_trainer.ui import render as render_mod

    ed.recipe_select(2)
    page = render_mod.render_editor(ed)
    assert "적용 중" in page and "프로젝트에는 저장되지 않는다" in page
    assert page.count('class="rrow') >= 4
    assert "레시피</span>" in page, "덮인 파라미터에 표식이 없다"


def test_router_recipe_roundtrip(ed):
    code, payload = server_mod.handle(ed, "/api/recipe/select", {"id": 3})
    assert code == 200 and payload["state"]["recipe"]["applied"] == 3

    code, payload = server_mod.handle(ed, "/api/recipe/drop-path", {"path": "p_crop.max_n"})
    assert code == 200

    code, payload = server_mod.handle(ed, "/api/recipe/add-path", {"path": "n_img.color_space"})
    assert code == 409 and not payload["ok"]

    code, payload = server_mod.handle(ed, "/api/recipe/select", {"id": None})
    assert code == 200 and payload["state"]["recipe"]["applied"] is None


def test_click_targets_are_big_enough_to_hit(ed):
    """8px짜리 손잡이는 사람도 못 누른다. 그리드 칸과 줄 높이로 영역을 확보한다."""
    from vlm_trainer.ui import render as render_mod

    ed.recipe_select(2)
    css = render_mod.render_editor(ed)
    for handle in (".odrop{", ".rrow .rdel{", ".padd{"):
        block = css.split(handle, 1)[1].split("}", 1)[0]
        assert "line-height:20px" in block or "line-height:16px" in block, handle
        assert "min-width:20px" in block or "min-width:16px" in block, handle
