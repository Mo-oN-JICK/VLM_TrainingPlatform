"""Phase 7 — 편집기 API. UI 전용 실행 경로 없이 코어만 거친다."""

from __future__ import annotations

import os
import shutil

import pytest

from vlm_trainer.core.compiler import compile_project
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


def test_disconnect_that_empties_a_required_input_is_refused(ed):
    """필수 입력을 비우는 것은 거부된다. 그래프는 언제나 컴파일되는 상태로 남는다."""
    res = ed.disconnect("n_ev:stats")
    assert not res["ok"] and "필수 입력" in res["detail"]
    assert any(e.dst == "n_ev:stats" for e in ed.graph.edges)


def test_optional_input_can_be_disconnected(ed):
    res = ed.disconnect("n_ev:regions")  # regions는 optional
    assert res["ok"], res.get("detail")
    assert all(e.dst != "n_ev:regions" for e in ed.graph.edges)


def test_removing_a_node_that_others_need_is_refused(ed):
    res = ed.remove_node("n_stats")
    assert not res["ok"], "상류가 사라지면 하류의 필수 입력이 비므로 거부되어야 한다"
    assert any(n.id == "n_stats" for n in ed.graph.nodes)


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


def test_adding_a_node_gives_a_free_id(ed):
    res = ed.add_node("ts.stats@1.0.0")
    # 새 노드는 아직 아무것도 먹이지 않으므로 필수 입력이 비어 거부된다
    assert not res["ok"] and "필수 입력" in res["detail"]


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
