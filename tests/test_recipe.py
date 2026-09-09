"""Phase 6 완료 조건 — Parameter Recipe, 스윕 전개, 예산 사전 검사, 물질화 공유."""

from __future__ import annotations

import os

import pytest
import yaml

from vlm_trainer.core.compiler import compile_project, current_value
from vlm_trainer.core.errors import PolicyError
from vlm_trainer.engine import sweep as sweep_mod
from vlm_trainer.spec import recipe as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPEC_DIR = os.path.join(ROOT, "solutions", "dummy_ecg", "projects", "01_dummy")
PROJECT = os.path.join(SPEC_DIR, "project.yaml")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "solutions", "dummy_ecg", "data", "dummy", "index.jsonl")),
    reason="더미 데이터가 없다. python tools/make_dummy_dataset.py 를 먼저 실행하라.",
)


def _book() -> R.RecipeBook:
    return R.load(PROJECT)


# ── 로드와 검증 ─────────────────────────────────────────────────────────


def test_book_loads_with_ids_names_and_sweeps():
    b = _book()
    assert sorted(b.recipes) == [1, 2, 3, 4]
    assert b.get(1).name == "baseline"
    assert b.active == 1
    assert "spike_x_crop" in b.sweeps
    assert b.display_name("n_stats.z_thresh") == "돌출 임계 z"
    assert b.display_name("없는.경로") == "없는.경로"


def test_recipe_may_not_change_structure(tmp_path):
    """레시피는 값만 덮는다. 구조가 다르면 같은 실험의 변주가 아니다."""
    d = tmp_path / "p"
    d.mkdir()
    (d / "recipes.yaml").write_text(
        yaml.safe_dump(
            {"kind": "ParameterRecipes", "recipes": [{"id": 1, "overrides": {"sample_space.index": "x"}}]},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(PolicyError, match="바꿀 수 없다"):
        R.load(str(d / "project.yaml"))


def test_duplicate_and_out_of_range_ids_are_rejected(tmp_path):
    d = tmp_path / "p"
    d.mkdir()

    def write(recipes):
        (d / "recipes.yaml").write_text(
            yaml.safe_dump({"kind": "ParameterRecipes", "recipes": recipes}, allow_unicode=True),
            encoding="utf-8",
        )

    write([{"id": 1, "overrides": {}}, {"id": 1, "overrides": {}}])
    with pytest.raises(Exception, match="중복"):
        R.load(str(d / "project.yaml"))

    write([{"id": 100, "overrides": {}}])
    with pytest.raises(Exception, match="1~99"):
        R.load(str(d / "project.yaml"))


# ── 컴파일에 적용 ───────────────────────────────────────────────────────


def test_overrides_reach_the_nodes_including_procedure_params():
    b = _book()
    base = compile_project(PROJECT)
    tight = compile_project(PROJECT, recipe_overrides=b.overrides_for(2))
    single = compile_project(PROJECT, recipe_overrides=b.overrides_for(3))

    assert base.nodes["n_stats"].params["z_thresh"] == 3.0
    assert tight.nodes["n_stats"].params["z_thresh"] == 4.5
    # Procedure의 노출 파라미터가 내부 노드로 옮겨진다
    assert single.nodes["p_crop/n_crop"].params["max_n"] == 1
    assert base.spec_hash != tight.spec_hash != single.spec_hash


def test_whitelist_still_guards_the_overlay():
    with pytest.raises(PolicyError, match="덮어쓸 수 없다"):
        compile_project(PROJECT, recipe_overrides={"n_stats.channel_does_not_exist": 1})
    with pytest.raises(PolicyError, match="덮어쓸 수 없다"):
        compile_project(PROJECT, recipe_overrides={"n_crop_missing.x": 1} if False else {"n_sample.placeholder": "!"})


def test_unexposed_procedure_param_is_rejected():
    with pytest.raises(PolicyError, match="노출하지 않은 파라미터"):
        compile_project(PROJECT, recipe_overrides={"p_crop.min_side_px": 4})


def test_status_is_active_until_the_project_is_edited():
    b = _book()
    cg = compile_project(PROJECT)
    paths = {p for r in b.recipes.values() for p in r.overrides}
    current = {p: current_value(cg, p) for p in paths}
    assert R.status(b, current) == "1"

    current["n_stats.z_thresh"] = 9.9  # 프로젝트를 직접 수정한 셈
    assert R.status(b, current) == R.CUSTOMIZED


# ── 스윕 전개 ───────────────────────────────────────────────────────────


def test_grid_expansion_is_deterministic_and_named():
    b = _book()
    a1 = R.expand(b, "spike_x_crop")
    a2 = R.expand(_book(), "spike_x_crop")
    assert [(r.id, r.name, r.overrides) for r in a1] == [(r.id, r.name, r.overrides) for r in a2]
    assert len(a1) == 6  # 3 x 2
    assert [r.id for r in a1] == [10, 11, 12, 13, 14, 15]  # 쓰이지 않은 번호부터
    assert a1[0].name == "z2.5_n1"
    # base 레시피의 값이 깔리고 축이 그 위를 덮는다
    assert a1[0].overrides["n_stats.z_thresh"] == 2.5
    assert a1[0].origin == "sweep:spike_x_crop"


def test_list_and_random_strategies(tmp_path):
    b = _book()
    b.sweeps["s_list"] = R.Sweep(
        name="s_list", axes={"n_stats.z_thresh": [2.0, 3.0], "p_crop.max_n": [1, 2]},
        strategy="list", id_range=(20, 30),
    )
    got = R.expand(b, "s_list")
    assert len(got) == 2  # zip
    assert got[0].overrides["n_stats.z_thresh"] == 2.0 and got[0].overrides["p_crop.max_n"] == 1

    b.sweeps["s_rand"] = R.Sweep(
        name="s_rand", axes={"n_stats.z_thresh": [2.0, 3.0, 4.0], "p_crop.max_n": [1, 2]},
        strategy="random", samples=3, seed=7, id_range=(40, 60),
    )
    first = [r.overrides for r in R.expand(b, "s_rand")]
    second = [r.overrides for r in R.expand(b, "s_rand")]
    assert len(first) == 3 and first == second, "seed가 같으면 같은 조합이어야 한다"


def test_lock_fixes_the_expansion(tmp_path):
    b = _book()
    expanded = R.expand(b, "spike_x_crop")
    R.save_lock(b, expanded, str(tmp_path))
    back = R.load_lock(str(tmp_path))
    assert [(r.id, r.name, r.overrides) for r in back] == [
        (r.id, r.name, r.overrides) for r in expanded
    ]


def test_diff_reports_only_what_changed():
    b = _book()
    rows = R.diff(b.get(1), b.get(3))
    assert rows == [("p_crop.max_n", 2, 1)]


# ── 물질화 공유와 예산 사전 검사 ─────────────────────────────────────────


def test_materialize_key_ignores_trainer_only_differences():
    """전처리가 같으면 같은 bake를 공유한다 — 스윕의 실질 비용을 결정한다."""
    b = _book()
    k1 = sweep_mod.materialize_key(compile_project(PROJECT, recipe_overrides=b.overrides_for(1)))
    k4 = sweep_mod.materialize_key(compile_project(PROJECT, recipe_overrides=b.overrides_for(4)))
    k3 = sweep_mod.materialize_key(compile_project(PROJECT, recipe_overrides=b.overrides_for(3)))

    assert k1 == k4, "Trainer 설정만 다른데 물질화가 갈렸다"
    assert k1 != k3, "전처리가 다른데 물질화를 공유한다"


def test_plan_rejects_over_budget_recipes_before_the_queue():
    b = _book()
    rep, compiled = sweep_mod.plan(
        PROJECT, [b.get(1), b.get(3)], trainer_config="trainer.yaml", device="rtx3060_12gb"
    )
    assert len(rep.entries) == 2 and not rep.rejected  # dummy-2b는 12GB에 들어간다

    # 7B를 양자화 없이 전체 미세조정하는 설정이면 큐에 들어가지 못한다
    over = os.path.join(SPEC_DIR, "_over.yaml")
    text = open(os.path.join(SPEC_DIR, "trainer.yaml"), encoding="utf-8").read()
    text = text.replace("backbone: dummy-2b", "backbone: dummy-7b").replace("  mode: nf4", "  mode: none")
    try:
        with open(over, "w", encoding="utf-8") as fh:
            fh.write(text)
        rep2, compiled2 = sweep_mod.plan(PROJECT, [b.get(1), b.get(3)], trainer_config="_over.yaml")
        assert not compiled2, "예산을 넘는 레시피가 큐에 들어갔다"
        assert all(e.status == sweep_mod.OVER_BUDGET for e in rep2.entries)
        assert all("예산 초과" in e.reason for e in rep2.entries)
    finally:
        if os.path.exists(over):
            os.remove(over)


def test_sweep_runs_three_experiments_sharing_one_bake(tmp_path):
    """완료 조건 — 그래프 1개 + 레시피 3개가 코드 수정 없이 순차 실행된다."""
    pytest.importorskip("torch")
    b = _book()
    rep = sweep_mod.run(
        PROJECT,
        [b.get(1), b.get(3), b.get(4)],
        run_root=str(tmp_path / "sweeps"),
        limit=2,
        shard_size=2,
        max_steps=1,
        torch_device="cpu",
        cache_dir=str(tmp_path / "cache"),
    )

    assert len(rep.queued) == 3 and not rep.rejected
    assert len(rep.bakes) == 2, "전처리 지문이 2종이어야 한다(1과 4는 공유)"
    assert len(rep.baked_now) == 2
    for e in rep.entries:
        assert e.steps > 0 and e.train_dir and os.path.exists(e.train_dir)
    # 1번과 4번은 같은 bake를 가리킨다
    by_id = {e.recipe.id: e for e in rep.entries}
    assert by_id[1].bake == by_id[4].bake != by_id[3].bake
