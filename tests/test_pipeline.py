"""파이프라인 — 준비 → 굽기 → 학습 → 추론을 한 번에.

두 가지를 지킨다. **단계는 그래프가 정한다**(사람이 고르지 않는다), 그리고 **파이프라인은
새 실행 경로를 만들지 않는다**(기존 CLI 명령을 그 순서로 부를 뿐이다). 둘 중 하나라도
무너지면 터미널에서 세 명령을 따로 친 것과 `vlmt pipeline` 이 다르게 도는 날이 온다.
"""

from __future__ import annotations

import json
import os

import pytest

from vlm_trainer.core import registry
from vlm_trainer.core.compiler import compile_project
from vlm_trainer.engine import pipeline as pipe

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OPEN_SPEC = os.path.join(ROOT, "solutions", "vlm_open", "projects", "01_open", "project.yaml")
ECG_SPEC = os.path.join(ROOT, "solutions", "dummy_ecg", "projects", "01_dummy", "project.yaml")


@pytest.fixture(autouse=True)
def _nodes():
    registry.load_builtin_nodes()


# ── 단계는 그래프가 정한다 ──────────────────────────────────────────────
def test_the_graph_decides_which_stages_exist():
    """Trainer 가 없는 그래프에 학습 단계를 띄워 놓고 "돌지 않음" 이라 적는 것보다
    애초에 없는 편이 읽기 쉽다."""
    full = pipe.plan(compile_project(OPEN_SPEC))
    assert [s.key for s in full.stages] == ["prepare", "bake", "train", "infer"]

    plain = pipe.plan(compile_project(ECG_SPEC))
    assert "prepare" in [s.key for s in plain.stages]
    assert "infer" not in [s.key for s in plain.stages], "추론 노드가 없는데 추론 단계가 생겼다"


def test_training_is_found_by_what_it_does_not_by_its_name():
    """`per_sample=False` 가 "데이터셋 전체를 본다" 는 뜻이고 학습이 정확히 그것이다.
    타입 이름(`train.vlm_trainer`)으로 찾으면 남이 만든 학습 노드가 안 보인다."""
    cg = compile_project(OPEN_SPEC)
    assert pipe.trainers(cg) == {"n_train"}


def test_the_stages_after_training_are_the_same_ones_the_gate_exempts():
    """컴파일러의 `external_call` 게이트와 같은 계산이어야 한다. 두 곳이 다르게 답하면
    게이트를 통과한 그래프가 파이프라인에서 빠지는 날이 온다."""
    cg = compile_project(OPEN_SPEC)
    assert pipe.after_training(cg) == {"n_infer", "n_answers"}


def test_the_inference_stage_runs_the_split_the_node_declared():
    """노드가 `only_split: val` 이라 적어 두고도 110건을 다 돌면, 100건은 전처리를
    끝까지 거친 뒤 추론에서 버려진다. 그 시간이 그대로 낭비다."""
    assert pipe.infer_split(compile_project(OPEN_SPEC)) == "val"


def test_a_split_disagreement_picks_nothing():
    """추론 노드 둘이 다른 split 을 말하면 한쪽만 듣고 나머지를 굶기는 것보다
    전부 도는 편이 낫다."""
    cg = compile_project(OPEN_SPEC)
    cg.nodes["n_answers"].params["only_split"] = "train"   # n_infer 는 val 이다
    assert pipe.infer_split(cg) == ""


# ── 실행 경로는 하나뿐이다 ──────────────────────────────────────────────
def test_every_stage_names_a_command_that_exists_on_the_cli():
    """파이프라인이 제 실행 코드를 갖기 시작하면, 터미널에서 친 것과 다르게 도는 날이
    온다. 그때 어느 쪽이 맞는지 아무도 모른다."""
    from vlm_trainer.cli.cmd_execute import _STAGE_FUNCS
    from vlm_trainer.cli.main import build_parser

    sub = build_parser()._subparsers._group_actions[0].choices
    for stage in pipe.plan(compile_project(OPEN_SPEC)).stages:
        assert stage.command in _STAGE_FUNCS
        assert stage.command in sub, f"{stage.command} 는 CLI 에 없는 명령이다"


def test_the_pipeline_subcommand_accepts_what_the_stages_need():
    """단계마다 같은 Namespace 를 넘긴다. 하위 명령이 읽는 인자가 하나라도 빠져 있으면
    그 단계에서 AttributeError 로 죽는다 — 굽기가 끝난 뒤에."""
    from vlm_trainer.cli.cmd_execute import STAGE_SET_ARGS
    from vlm_trainer.cli.main import build_parser

    sub = build_parser()._subparsers._group_actions[0].choices
    # `split` 처럼 단계마다 값이 다른 것은 파이프라인이 직접 꽂는다.
    have = {a.dest for a in sub["pipeline"]._actions} | set(STAGE_SET_ARGS)
    for name in ("budget", "materialize", "train", "run"):
        need = {a.dest for a in sub[name]._actions} - {"help", "func"}
        assert need <= have, f"{name} 가 읽는 인자가 pipeline 에 없다: {sorted(need - have)}"


# ── 굽다 만 것 ──────────────────────────────────────────────────────────
def test_leftover_reports_but_does_not_decide(tmp_path):
    """말없이 이어받으면 예전 파라미터로 구운 샘플이 섞이고, 말없이 지우면 구운 시간이
    날아간다. 양쪽 다 나쁘므로 세어서 알려만 준다."""
    root = tmp_path / "runs"
    out = root / "r1" / "materialized"
    out.mkdir(parents=True)
    (out / "manifest.jsonl").write_text(
        json.dumps({"shard": "shard-00000", "n": 64, "keys": []}) + "\n"
        + json.dumps({"shard": "shard-00001", "n": 36, "keys": []}) + "\n",
        encoding="utf-8")

    left = pipe.leftover("r1", root=str(root))
    assert left and left.shards == 2 and left.samples == 100
    assert not pipe.leftover("r2", root=str(root)), "없는 실행에 남은 것이 있다고 한다"


# ── 남기기 ──────────────────────────────────────────────────────────────
def test_the_snapshot_survives_a_round_trip(tmp_path):
    """편집기는 이 파일만 읽는다. 여기서 형태가 어긋나면 앱의 진행 표시가 조용히 빈다."""
    p = pipe.plan(compile_project(OPEN_SPEC), "r9")
    p.stages[0].state, p.stages[0].ms = pipe.DONE, 1234.0
    p.stages[1].state = pipe.RUNNING

    path = str(tmp_path / "pipeline.json")
    pipe.write(p, path)
    back = pipe.read(path)

    assert back is not None and back.run_id == "r9"
    assert [s.state for s in back.stages] == [s.state for s in p.stages]
    assert back.active == "bake"
    assert "굽기" in pipe.headline(back)


def test_a_failed_stage_shows_in_the_headline():
    p = pipe.plan(compile_project(OPEN_SPEC), "r9")
    p.get("bake").state = pipe.FAILED
    assert p.failed and "굽기 x" in pipe.headline(p)
    assert pipe.as_dict(p)["failed"] is True
