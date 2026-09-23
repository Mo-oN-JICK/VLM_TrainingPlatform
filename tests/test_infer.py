"""학습된 모델로 답을 만드는 경로.

`모델 학습` 이 산출물을 내보내고 `모델 추론` 이 그것을 받는다. Output 이 그래프의
끝이어야 한다는 규칙을 푼 이유가 이 연결이다 — 경로를 파라미터에 적으면 동작은 하지만
"학습한 모델로 추론한다" 가 화면에서 사라진다.
"""

from __future__ import annotations

import os

import pytest

from vlm_trainer.core import registry
from vlm_trainer.core.compiler import compile_project
from vlm_trainer.core.node import NodeError, NodeKind
from vlm_trainer.core.registry import resolve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


@pytest.fixture(autouse=True)
def _nodes():
    registry.load_builtin_nodes()


def test_training_hands_the_model_to_whoever_asks():
    """Output 이면서 출력을 낸다. 부작용의 자리이지 반드시 끝은 아니다."""
    d = resolve("train.vlm_trainer@0.1.0")
    assert d.kind is NodeKind.OUTPUT
    assert "model" in d.outputs, "학습이 산출물을 안 내놓으면 추론이 받을 것이 없다"


def test_inference_declares_that_it_calls_a_model():
    """모델 파일을 디스크에서 읽으므로 순수 함수가 아니다. 그 예외를 선언으로 드러낸다 —
    `expert.propose` 가 같은 문제를 그렇게 풀었다."""
    d = resolve("infer.vlm@1.0.0")
    assert d.kind is NodeKind.PROCESSING
    assert d.external_call, "외부 모델 호출이 선언되지 않으면 게이트가 볼 수 없다"
    assert not d.deterministic, "체크포인트가 바뀌면 답이 바뀐다"
    assert set(d.inputs) >= {"model", "prompt"} and "answer" in d.outputs


def test_inference_after_training_is_not_asked_for_a_boundary():
    """`external_call` 은 물질화 경계 앞에 있어야 한다. 그 규칙이 막는 위험은
    **학습 루프 안에서** VRAM 을 뺏는 것인데, 학습 뒤에 있는 노드는 루프 안에 있을 수 없다.
    없는 위험을 게이트가 말하기 시작하면 사람이 게이트를 믿지 않게 된다.

    `vlm_open` 이 실물 증거다 — 추론이 경계 뒤에 배선된 채로 컴파일을 통과한다.
    """
    spec = os.path.join(ROOT, "solutions", "vlm_open", "projects", "01_open", "project.yaml")
    cg = compile_project(spec)     # 추론이 경계 뒤에 있는데도 통과해야 한다
    assert cg.lanes["n_train"] < cg.lanes["n_infer"], "학습이 추론보다 아래에 놓였다"
    assert cg.lanes["n_answers"] == max(cg.lanes.values()), "종결 Output 이 맨 아래가 아니다"


def test_the_inference_node_takes_the_same_pictures_training_saw():
    """전처리를 거친 이미지를 받아야 한다. 원본을 바로 물리면 모델이 한 번도 본 적 없는
    크기가 들어가고, 답이 나빠진 이유를 모델 탓으로 돌리게 된다."""
    d = resolve("infer.vlm@1.0.0")
    assert "images" in d.inputs, "포트가 없으면 노드가 이미지를 읽어도 배선할 길이 없다"

    spec = os.path.join(ROOT, "solutions", "vlm_open", "projects", "01_open", "project.yaml")
    cg = compile_project(spec)
    src = cg.nodes["n_infer"].inputs.get("images", "")
    assert src.startswith("p_prep"), f"전처리를 거치지 않은 이미지가 들어온다: {src!r}"


def test_a_terminal_output_still_sits_at_the_bottom():
    """뒤로 이어지지 않는 Output 은 여전히 맨 아래다. 규칙을 푼 것이지 버린 것이 아니다."""
    spec = os.path.join(ROOT, "solutions", "vlm_parts", "projects", "01_parts", "project.yaml")
    cg = compile_project(spec)
    tail = [i for i in cg.order if cg.nodes[i].kind is NodeKind.OUTPUT]
    bottom = max(cg.lanes.values())
    assert tail and all(cg.lanes[i] == bottom for i in tail)


def test_the_report_does_not_score():
    """자동 채점 숫자가 나오면 그 숫자를 믿게 되는데, 라벨에 잡음이 있으면 믿을 값이 아니다.
    답과 정답을 나란히 적을 뿐이다."""
    d = resolve("io.answer_report@1.0.0")
    assert d.kind is NodeKind.OUTPUT
    assert set(d.inputs) == {"answer", "expected"}
    assert not d.outputs, "채점 결과를 내놓기 시작하면 그 숫자를 믿게 된다"


# ── 학습한 것을 정말로 다시 올리는가 ────────────────────────────────────
def test_the_checkpoint_records_what_lora_did_to_the_module_tree():
    """LoRA 를 끼우면 `q_proj` 가 `q_proj.base` 가 된다. 나중에 이 체크포인트를 읽는 쪽이
    같은 트리를 먼저 만들지 못하면 `strict=False` 가 그 가중치를 **통째로 버리고**,
    학습 안 된 모델이 답을 내놓는데 아무도 모른다."""
    import inspect

    from vlm_trainer.train import loop as loop_mod

    src = inspect.getsource(loop_mod.train)
    assert '"lora": (' in src, "체크포인트에 LoRA 설정이 남지 않는다"


def test_loading_a_checkpoint_that_does_not_fit_is_loud(tmp_path, monkeypatch):
    """조용히 절반만 올라가는 것이 이 경로에서 제일 나쁜 결말이다 — 답은 나오고,
    그 답을 보고 "파인튜닝이 소용없다" 고 판단하게 된다."""
    torch = pytest.importorskip("torch")
    from torch import nn

    from vlm_trainer.nodes import infer as infer_mod
    from vlm_trainer.plugins.base import BackboneAdapter

    stage = tmp_path / "s1"
    stage.mkdir()
    torch.save({"model": {"없는이름.weight": torch.zeros(2)}, "lora": {}}, stage / "c.pt")

    class _Adapter(BackboneAdapter):
        @classmethod
        def build(cls, cfg, st):
            return nn.Linear(2, 2)

        @classmethod
        def generate(cls, *a, **k):
            return ""

    monkeypatch.setattr("vlm_trainer.plugins.base.resolve_backbone", lambda ref: _Adapter)
    infer_mod._LOADED.clear()
    with pytest.raises(NodeError, match="들어갈 자리가 없다"):
        infer_mod._load("n_infer", str(tmp_path), "아무백본")


def test_an_empty_answer_is_still_written(tmp_path):
    """빈 답도 결과다. 버리면 `n_infer 10건 성공` 인데 파일에는 8줄이 남고,
    모델이 아무 말도 안 한 두 건이 세상에서 사라진다(실측)."""
    import json

    from vlm_trainer.core.node import RunCtx

    d = resolve("io.answer_report@1.0.0")
    out = str(tmp_path / "infer")
    ctx = RunCtx(run_id="r1", node_id="n_answers", sample_key="o0100",
                 sample={"_split": "val"}, spec_dir=str(tmp_path))
    d.impl().run(ctx, d.build_params({"out_dir": out}), answer="", expected="기어")

    rows = [json.loads(l) for l in open(os.path.join(out, "answers.jsonl"), encoding="utf-8")]
    assert len(rows) == 1, "모델이 아무 말도 안 한 건이 사라졌다"
    assert rows[0]["answer"] == "" and rows[0]["split"] == "val"
