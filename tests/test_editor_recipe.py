"""Parameter Recipe — `test_editor.py` 에서 갈라 나왔다.

1234줄 한 파일이라 무엇이 어디 있는지 찾기 어려웠다. 테스트 이름과 내용은 그대로다.
"""

from __future__ import annotations

import os
import shutil

import pytest

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



def test_recipe_overlays_values_without_touching_the_spec(ed):
    """레시피는 값만 덮는다. 화면의 값과 저장될 값이 다르다는 것이 요점이다."""
    assert ed.recipe_select(2)["ok"]

    assert ed.compiled.nodes["n_stats"].params["z_thresh"] == 4.5   # 화면
    assert ed.base.nodes["n_stats"].params["z_thresh"] == 3.0       # 스펙
    assert "n_stats:z_thresh" in ed.state()["overlaid"]

    assert ed.save()["ok"]
    assert "4.5" not in open(ed.path, encoding="utf-8").read(), "레시피 값이 스펙에 스몄다"


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


