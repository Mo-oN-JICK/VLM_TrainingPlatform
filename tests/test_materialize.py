"""Phase 4 완료 조건 — 물질화 경계, 원자 커밋, 재개, 워커 종료."""

from __future__ import annotations

import json
import os

import pytest

from vlm_trainer.core.compiler import compile_project
from vlm_trainer.core.errors import StructureError
from vlm_trainer.engine import materialize as mat
from vlm_trainer.engine import samples as samples_mod
from vlm_trainer.engine import worker
from vlm_trainer.engine.journal import Journal
from vlm_trainer.engine.runner import RunOptions
from vlm_trainer.train import shards as shards_mod

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJECT = os.path.join(ROOT, "solutions", "dummy_ecg", "projects", "01_dummy", "project.yaml")
SPEC_DIR = os.path.dirname(PROJECT)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "solutions", "dummy_ecg", "data", "dummy", "index.jsonl")),
    reason="더미 데이터가 없다. python tools/make_dummy_dataset.py 를 먼저 실행하라.",
)


def _load():
    cg = compile_project(PROJECT)
    return cg, samples_mod.load(cg.sample_space, SPEC_DIR)


def _run(tmp_path, cg, space, **kw):
    ro = RunOptions(run_id="t", cache_dir=str(tmp_path / "cache"), spec_dir=SPEC_DIR)
    mo = mat.MaterializeOptions(out_dir=str(tmp_path / "materialized"), **kw)
    return mat.materialize(cg, space, mo, ro)


def test_materialize_writes_shards_and_manifest(tmp_path):
    cg, space = _load()
    rep = _run(tmp_path, cg, space, limit=6, shard_size=2)

    assert rep.ok, rep.aborted
    assert rep.written == 6 and rep.reused == 0
    assert len(rep.shards) == 3

    entries = shards_mod.manifest(rep.out_dir)
    assert [e["n"] for e in entries] == [2, 2, 2]
    assert sum(len(e["keys"]) for e in entries) == 6

    st = shards_mod.stats(rep.out_dir)
    assert st.samples == 6 and st.shards == 3
    assert st.max_images_per_sample == 3  # 파형 1장 + crop 2장
    assert st.prompt_chars > 0 and st.answer_chars > 0


def test_samples_can_be_read_back_with_images(tmp_path):
    cg, space = _load()
    rep = _run(tmp_path, cg, space, limit=2, shard_size=2)
    recs = list(shards_mod.iter_samples(rep.out_dir, with_images=True))
    assert len(recs) == 2
    r = recs[0]
    assert r["prompt"] and r["answer"].startswith("<trend>")
    assert len(r["_images"]) == 3
    assert r["_images"][0].shape == (448, 448, 3)
    # 파일이 shard 디렉터리 안에 실제로 있다
    for rel in r["images"]:
        assert os.path.exists(os.path.join(rep.out_dir, r["_shard"], rel))


def test_resume_does_not_rebuild_committed_shards(tmp_path):
    cg, space = _load()
    first = _run(tmp_path, cg, space, limit=4, shard_size=2)
    assert first.written == 4

    second = _run(tmp_path, cg, space, limit=6, shard_size=2, resume=True)
    assert second.reused == 4, "커밋된 샘플을 다시 구웠다"
    assert second.written == 2
    assert shards_mod.stats(second.out_dir).samples == 6

    # 같은 키가 두 번 들어가지 않는다
    keys = [k for e in shards_mod.manifest(second.out_dir) for k in e["keys"]]
    assert len(keys) == len(set(keys))


def test_uncommitted_shard_is_treated_as_absent_and_removed(tmp_path):
    """매니페스트에 줄이 붙는 순간이 커밋이다. 그 전에 죽은 shard는 없는 것으로 본다."""
    cg, space = _load()
    rep = _run(tmp_path, cg, space, limit=2, shard_size=2)

    orphan_dir = os.path.join(rep.out_dir, "shard-00001")
    os.makedirs(os.path.join(orphan_dir, "images"), exist_ok=True)
    with open(os.path.join(rep.out_dir, "shard-00001.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "junk", "prompt": "x", "answer": "y", "images": []}) + "\n")

    # 커밋되지 않았으므로 읽히지 않는다
    assert [r["id"] for r in shards_mod.iter_samples(rep.out_dir)] != ["junk"]
    assert all(r["id"] != "junk" for r in shards_mod.iter_samples(rep.out_dir))

    again = _run(tmp_path, cg, space, limit=4, shard_size=2, resume=True)
    assert "shard-00001.jsonl" in again.orphans_removed or "shard-00001" in again.orphans_removed
    assert all(r["id"] != "junk" for r in shards_mod.iter_samples(again.out_dir))


def test_resume_is_refused_when_the_spec_changed(tmp_path):
    cg, space = _load()
    _run(tmp_path, cg, space, limit=2, shard_size=2)

    changed = compile_project(PROJECT, recipe_overrides={"n_plot.channel": 1})
    assert changed.spec_hash != cg.spec_hash
    rep = _run(tmp_path, changed, space, limit=4, shard_size=2, resume=True)

    assert not rep.ok
    assert "재개 거부" in rep.aborted and "spec_hash" in rep.aborted
    assert rep.written == 0


def test_expert_worker_is_torn_down_after_materialize(tmp_path):
    """전문가 모델이 잡고 있던 GPU를 완전히 반납해야 학습이 시작될 수 있다."""
    cg, space = _load()
    _run(tmp_path, cg, space, limit=2, shard_size=2)
    assert worker._POOL is None, "물질화가 끝났는데 워커 풀이 남아 있다"


def test_journal_records_only_committed_facts(tmp_path):
    cg, space = _load()
    rep = _run(tmp_path, cg, space, limit=4, shard_size=2)

    j = Journal(os.path.join(tmp_path, "journal.jsonl"))
    replay = j.replay()
    assert replay.spec_hash == cg.spec_hash
    assert len(replay.shards) == 2
    assert replay.committed_samples == 4
    assert "materialize" in replay.phases_done


def test_output_nodes_are_not_run_during_materialize(tmp_path):
    cg, space = _load()
    rep = _run(tmp_path, cg, space, limit=2, shard_size=2)
    from vlm_trainer.core.node import NodeKind

    for nid in cg.order:
        if cg.nodes[nid].kind is NodeKind.OUTPUT:
            assert not rep.run.states_of(nid), f"{nid}는 Output인데 물질화 중에 실행되었다"


def test_boundary_is_required(tmp_path):
    cg, space = _load()
    cg.materialize = dict(cg.materialize)
    cg.materialize["boundary"] = []
    with pytest.raises(StructureError, match="materialize.boundary"):
        _run(tmp_path, cg, space, limit=1)


def test_only_the_boundary_upstream_is_computed(tmp_path):
    """경계 상류만 돈다. 경계와 무관한 가지는 물질화 단계에서 계산되지 않는다."""
    cg, space = _load()
    targets = mat.boundary_targets(cg)
    assert "n_sample" in targets and "n_train" not in targets and "n_export" not in targets

    rep = _run(tmp_path, cg, space, limit=1, shard_size=1)
    ran = {i for i in rep.run.order if rep.run.states_of(i)}
    assert ran <= targets
