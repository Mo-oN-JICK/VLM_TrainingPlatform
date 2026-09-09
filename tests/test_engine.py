"""Phase 2 완료 조건 — 엔진, 캐시, 미리보기, 격리, 결정성 감사, 위반율 게이트."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

from vlm_trainer.core.compiler import compile_project
from vlm_trainer.core.graph import Edge, GraphModel, NodeInstance
from vlm_trainer.core.node import NodeKind
from vlm_trainer.engine import samples as samples_mod
from vlm_trainer.engine.dryrun import dryrun
from vlm_trainer.engine.runner import RunOptions, ancestors, execute

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJECT = os.path.join(ROOT, "solutions", "dummy_ecg", "projects", "01_dummy", "project.yaml")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "solutions", "dummy_ecg", "data", "dummy", "index.jsonl")),
    reason="더미 데이터가 없다. python tools/make_dummy_dataset.py 를 먼저 실행하라.",
)


def _load():
    cg = compile_project(PROJECT)
    space = samples_mod.load(cg.sample_space, os.path.dirname(PROJECT))
    return cg, space


def test_sample_space_splits_do_not_mix_groups():
    _, space = _load()
    assert len(space) > 0
    by_patient: Dict[str, set] = {}
    for r in space.rows:
        by_patient.setdefault(r["patient_id"], set()).add(r["_split"])
    assert all(len(v) == 1 for v in by_patient.values()), "같은 환자가 두 split에 들어갔다"
    assert all(r["quality_flag"] == "ok" for r in space.rows), "filter가 적용되지 않았다"


def test_dryrun_flows_to_sample_assemble(tmp_path):
    cg, space = _load()
    opts = RunOptions(run_id="test", cache_dir=str(tmp_path / "cache"))
    res = dryrun(cg, space, n=3, opts=opts)
    assert res.ok, res.type_mismatches + res.determinism_failures + [res.aborted]
    assert res.processed == 3
    assert "n_sample:sample" in res.measured
    assert not res.type_mismatches


def test_second_run_is_fully_cached(tmp_path):
    cg, space = _load()
    rows = space.rows[:2]
    opts = RunOptions(run_id="test", cache_dir=str(tmp_path / "cache"))

    first = execute(cg, space, rows, opts)
    assert first.processed == 2
    assert first.cache["writes"] > 0

    second = execute(cg, space, rows, opts)
    assert second.processed == 2
    processing = [i for i in cg.order if cg.nodes[i].kind is not NodeKind.OUTPUT]
    for nid in processing:
        st = second.states_of(nid)
        assert st.get("cached") == 2, f"{nid}가 캐시되지 않았다: {st}"
        assert "success" not in st


def test_preview_computes_only_the_upstream(tmp_path):
    cg, space = _load()
    target = "n_stats"
    up = ancestors(cg, target)
    assert "n_img" not in up and "n_ts" in up

    opts = RunOptions(run_id="test", cache_dir=str(tmp_path / "cache"))
    rep = execute(cg, space, space.pick(1), opts, targets=up)
    ran = {i for i in rep.order if rep.states_of(i)}
    assert ran <= up, f"상류 밖 노드가 실행되었다: {ran - up}"
    assert f"{target}:stats" in rep.last_values

    # 두 번째 호출은 전부 캐시에서 온다
    again = execute(cg, space, space.pick(1), opts, targets=up)
    assert all(again.states_of(i).get("cached") for i in ran)


def test_output_nodes_are_not_run_unless_asked(tmp_path):
    cg, space = _load()
    opts = RunOptions(run_id="test", cache_dir=str(tmp_path / "cache"))
    rep = execute(cg, space, space.pick(1), opts)
    for nid in cg.order:
        if cg.nodes[nid].kind is NodeKind.OUTPUT:
            assert not rep.states_of(nid), f"{nid}는 Output인데 실행되었다"


# ── 격리 · 결정성 (fixture 노드 사용) ────────────────────────────────────


def _tiny(node_type: str) -> GraphModel:
    g = GraphModel(id="t")
    g.nodes = [
        NodeInstance("n_src", "test.text_asset@1.0.0", {}),
        NodeInstance("n_mid", node_type, {}),
        NodeInstance("n_out", "test.sink@1.0.0", {}),
    ]
    g.edges = [Edge.parse("n_src:text", "n_mid:text"), Edge.parse("n_mid:text", "n_out:text")]
    return g


def test_worker_crash_is_isolated_to_one_node(tmp_path):
    from vlm_trainer.core.compiler import compile_graph

    cg = compile_graph(_tiny("test.crasher@1.0.0"))
    space = samples_mod.SampleSpace(root=str(tmp_path), key="sample_id", rows=[{"sample_id": "a"}])
    opts = RunOptions(
        run_id="test",
        cache_dir=str(tmp_path / "cache"),
        extra_modules=("fixture_nodes",),
        quarantine_ratio_threshold=1.0,
    )
    rep = execute(cg, space, space.rows, opts)

    assert rep.states_of("n_mid").get("failed") == 1, "크래시 노드가 failed로 표시되지 않았다"
    assert rep.states_of("n_src").get("success") == 1, "상류는 정상 실행되었어야 한다"
    assert rep.quarantine and rep.quarantine[0].node_id == "n_mid"
    assert "워커 프로세스가 죽었다" in rep.quarantine[0].cause


def test_determinism_audit_catches_impure_processing(tmp_path):
    from vlm_trainer.core.compiler import compile_graph

    cg = compile_graph(_tiny("test.nondet@1.0.0"))
    space = samples_mod.SampleSpace(root=str(tmp_path), key="sample_id", rows=[{"sample_id": "a"}])
    opts = RunOptions(run_id="test", cache_dir=str(tmp_path / "cache"), determinism_audit=True, use_cache=False)
    rep = execute(cg, space, space.rows, opts)
    assert "n_mid" in rep.determinism_failures


# ── 정답 스키마 위반율 게이트 ────────────────────────────────────────────


def test_schema_validate_catches_violations():
    from vlm_trainer.answer.schema import AnswerSchema

    schema = AnswerSchema.load(
        os.path.join(ROOT, "solutions", "dummy_ecg", "schemas", "answer.yaml")
    )
    good = schema.render_answer(
        {
            "trend": "rising",
            "periodicity": {"present": True, "period_n": 10},
            "spike": {"found": True, "count": 2, "max_z": 6.0},
            "verdict": "abnormal",
        },
        {k: "충분히 긴 근거 문장입니다." for k in ("trend", "periodicity", "spike", "verdict")},
    )
    assert schema.validate(good) == []

    # enum 밖의 값
    bad_enum = good.replace("<trend>rising", "<trend>약간 상승")
    assert any(v.startswith("enum_domain") for v in schema.validate(bad_enum))

    # 근거 문장 누락
    bad_ev = schema.render_answer(
        {
            "trend": "flat",
            "periodicity": {"present": False, "period_n": 0},
            "spike": {"found": False, "count": 0, "max_z": 1.0},
            "verdict": "normal",
        },
        {},
    )
    assert any(v.startswith("evidence_missing") for v in schema.validate(bad_ev))

    # 단계 간 규칙: 큰 돌출이 있는데 normal
    bad_rule = schema.render_answer(
        {
            "trend": "flat",
            "periodicity": {"present": True, "period_n": 10},
            "spike": {"found": True, "count": 3, "max_z": 7.0},
            "verdict": "normal",
        },
        {k: "충분히 긴 근거 문장입니다." for k in ("trend", "periodicity", "spike", "verdict")},
    )
    assert any(v.startswith("rule:") for v in schema.validate(bad_rule))

    # 단계 누락
    assert any(v.startswith("step_missing") for v in schema.validate("<trend>flat — 근거 문장입니다.</trend>"))


def test_dryrun_stops_when_violation_ratio_is_too_high(tmp_path):
    """근거 문장을 만들지 않는 규칙 파일로 바꾸면 학습 전에 멈춰야 한다."""
    cg = compile_project(
        PROJECT, recipe_overrides={"n_ev.rules": "../../schemas/evidence_empty.yaml"}
    )
    space = samples_mod.load(cg.sample_space, os.path.dirname(PROJECT))
    opts = RunOptions(run_id="test", cache_dir=str(tmp_path / "cache"))
    res = dryrun(cg, space, n=3, opts=opts)

    assert not res.ok
    assert res.aborted
    assert res.quarantine and all("위반" in q for q in res.quarantine)
