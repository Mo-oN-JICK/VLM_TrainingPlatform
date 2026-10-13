"""Line(5노드) — 펼친 결과가 11노드와 같아야 한다.

같은 데이터·같은 설정이면 5노드와 11노드의 결과가 같다는 것이 이 작업의 약속이다.
그래프가 같은지(spec_hash)와 실제로 돌려 답이 같은지(answers.jsonl) 둘 다 본다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import yaml

from vlm_trainer.core import registry
from vlm_trainer.core.compiler import compile_project
from vlm_trainer.core.errors import SpecError
from vlm_trainer.spec.line import expand_data, expand_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJ = os.path.join(ROOT, "solutions", "vlm_open", "projects", "01_open")
OLD, NEW = os.path.join(PROJ, "project.yaml"), os.path.join(PROJ, "line.yaml")


@pytest.fixture(autouse=True)
def _nodes():
    registry.load_builtin_nodes()


def _line():
    with open(NEW, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_the_expanded_line_is_the_same_graph_as_the_eleven_box_project():
    """펼친 그래프에 새 노드가 끼면 팔레트가 실행 비용이 된 것이다. 해시가 곧 그 증거다."""
    a, b = compile_project(OLD), compile_project(NEW)
    assert a.spec_hash == b.spec_hash
    assert sorted(a.nodes) == sorted(b.nodes)


def test_an_empty_cell_is_refused_where_it_is_read_and_says_which_one():
    d = _line()
    d["stations"] = [s for s in d["stations"] if s["id"] != "s2"]  # 질문 형태를 뺀다
    with pytest.raises(SpecError) as e:
        expand_data(d, PROJ)
    msg = str(e.value)
    assert "prompt" in msg and "비어 있습니다" in msg and "정답 형태" in msg


def test_a_setting_the_station_does_not_expose_is_refused():
    d = _line()
    d["stations"][2]["params"]["n_guard.mode"] = "x"
    with pytest.raises(SpecError, match="노출하지 않은"):
        expand_data(d, PROJ)


def test_the_answer_station_keeps_the_leak_guard_and_the_format_check():
    """정답 노출 검사와 형식 검사는 화면에서 숨겨도 꺼지면 안 된다."""
    nodes = {n["id"]: n["type"] for n in expand_file(NEW)["nodes"]}
    assert nodes["n_guard"].startswith("answer.leakage_guard")
    procs = {p["id"]: p["ref"] for p in expand_file(NEW)["procedures"]}
    assert procs["p_answer"].startswith("answer_build")
    with open(os.path.join(ROOT, "solutions", "stations", "answer.yaml"), encoding="utf-8") as fh:
        exposed = yaml.safe_load(fh)["exposed_params"]
    assert not any("enable" in k or "skip" in k or "off" in k for k in exposed)


def _run(spec: str, cwd: str, run_id: str) -> str:
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [sys.executable, "-m", "vlm_trainer.cli.main", "pipeline", spec,
         "--run-id", run_id, "--no-cache"],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600,
    )
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return os.path.join(cwd, "runs", run_id)


def _shards(run_dir: str) -> dict:
    root = os.path.join(run_dir, "materialized")
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            with open(os.path.join(dp, f), "rb") as fh:
                out[os.path.relpath(os.path.join(dp, f), root)] = fh.read()
    return out


def _rows(run_dir: str, *, drop=()) -> list:
    with open(os.path.join(run_dir, "infer", "answers.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    return [{k: v for k, v in r.items() if k not in drop} for r in rows]


def test_both_forms_bake_the_same_data_and_ask_the_same_questions(tmp_path):
    """사전 준비 묶음은 바이트까지, 검증 건 목록·모델이 낸 답 문장은 통째로 같다.

    학습이 시드를 쥐고 있으므로 같은 데이터·같은 설정이면 답 문장도 같아야 한다.
    """
    pytest.importorskip("torch")
    if not os.path.exists(os.path.join(ROOT, "solutions", "vlm_open", "data", "open_parts", "index.jsonl")):
        pytest.skip("vlm_open 데이터가 없다")
    old, new = _run(OLD, str(tmp_path), "old"), _run(NEW, str(tmp_path), "new")

    assert _shards(old) and _shards(old) == _shards(new)
    assert _rows(old) == _rows(new)
    assert len(_rows(new)) == 10
    # 실제로 돈 11노드 스펙과 해시가 실행 기록에 남는다
    rec = os.path.join(new, "expanded_project.yaml")
    with open(rec, encoding="utf-8") as fh:
        assert compile_project(NEW).spec_hash in fh.readline()
    assert not os.path.exists(os.path.join(old, "expanded_project.yaml"))


# ── 정답 노출 검사는 모르는 값으로 꺼지지 않는다 ───────────────────────────────
class _Vocab:
    def vocabulary_terms(self):
        return ["기어"]


def _guard(on_leak, mode="exact"):
    from types import SimpleNamespace

    from vlm_trainer.nodes.answer import GuardParams, LeakageGuard

    return LeakageGuard().run(
        SimpleNamespace(node_id="n_guard", sample_key="k"),
        GuardParams(mode=mode, on_leak=on_leak),
        prompt="이 기어 사진을 보고 판정하시오", answer="x", schema=_Vocab())


def test_an_unknown_on_leak_value_is_refused_not_silently_passed():
    """`quarantine` 이 이 검사를 통째로 끈 채 통과했었다."""
    from vlm_trainer.core.node import NodeError

    with pytest.raises(NodeError, match="없는 값"):
        _guard("quarantine")
    with pytest.raises(NodeError, match="없는 값"):
        _guard("fail", mode="regex")


def test_a_leaking_prompt_is_stopped_with_fail_and_masked_with_mask():
    from vlm_trainer.core.node import NodeError

    with pytest.raises(NodeError, match="기어"):
        _guard("fail")
    assert "[MASKED]" in _guard("mask")["prompt"]


def test_no_shipped_spec_uses_a_value_the_guard_does_not_know():
    for rel in (("vlm_open", "projects", "01_open", "project.yaml"),
                ("vlm_open", "projects", "01_open", "line.yaml"),
                ("vlm_parts", "projects", "01_parts", "project.yaml")):
        with open(os.path.join(ROOT, "solutions", *rel), encoding="utf-8") as fh:
            assert "on_leak: quarantine" not in fh.read(), rel
    with open(os.path.join(ROOT, "solutions", "stations", "answer.yaml"), encoding="utf-8") as fh:
        assert "on_leak: quarantine" not in fh.read()
