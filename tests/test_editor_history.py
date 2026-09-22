"""이력·되감기·저장 — `test_editor.py` 에서 갈라 나왔다.

1234줄 한 파일이라 무엇이 어디 있는지 찾기 어려웠다. 테스트 이름과 내용은 그대로다.
"""

from __future__ import annotations

import os
import shutil

import pytest

from vlm_trainer.core.compiler import compile_project
from vlm_trainer.spec import recipe as recipe_mod
from vlm_trainer.ui import tokens as T
from vlm_trainer.ui.api import Editor

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


def test_incomplete_graph_is_not_saved(ed):
    ed.add_node("ts.stats@1.0.0")
    res = ed.save()
    assert not res["ok"] and "필수 입력" in res["detail"]

    original = open(ed.path, encoding="utf-8").read()
    assert "n_stats_2" not in original


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


def test_the_journal_stays_readable(ed):
    """저널은 사람이 읽는 diff다. 본문은 옆 저장소에 따로 둔다."""
    import json as _json

    ed.set_param("n_stats", "z_thresh", 4.5)
    lines = [_json.loads(ln) for ln in open(ed.history_path, encoding="utf-8") if ln.strip()]
    assert set(lines[-1]) == {"at", "label", "spec_hash", "diff"}
    assert lines[-1]["diff"] == ["-    z_thresh: 3.0", "+    z_thresh: 4.5"]
    assert lines[-1]["at"].startswith("20"), "세션을 넘으려면 날짜가 있어야 한다"


# ── Debug Output 이미지 ─────────────────────────────────────────────────


def test_the_sample_space_edit_survives_a_save(ed_with_data):
    ed = ed_with_data
    assert ed.set_sample_space("filter", "")["ok"]
    assert ed.save()["ok"]

    again = Editor.open(ed.path)
    assert again.sample_space_view()["rows"] == 24
    assert again.graph.sample_space.filter == ""


# ── 물질화와 학습 ───────────────────────────────────────────────────────


def test_materialize_and_train_refuse_unsaved_changes(ed):
    ed.set_param("n_stats", "z_thresh", 4.5)
    for res in (ed.materialize_start(), ed.train_start()):
        assert not res["ok"] and "저장하지 않은 변경" in res["reason"]


