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
from vlm_trainer.core.node import NodeKind
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
    """
    from vlm_trainer.core.compiler import compile_graph
    from vlm_trainer.core.graph import Edge, NodeInstance
    from vlm_trainer.spec.loader import load_project

    spec = os.path.join(ROOT, "solutions", "vlm_open", "projects", "01_open", "project.yaml")
    g = load_project(spec)
    g.nodes.append(NodeInstance("n_infer", "infer.vlm@1.0.0", {}))
    g.nodes.append(NodeInstance("n_rep", "io.answer_report@1.0.0", {}))
    g.edges += [
        Edge.parse("n_train:model", "n_infer:model"),
        Edge.parse("n_guard:prompt", "n_infer:prompt"),
        Edge.parse("n_infer:answer", "n_rep:answer"),
        Edge.parse("p_answer:answer", "n_rep:expected"),
    ]
    cg = compile_graph(g)     # 추론이 경계 뒤에 있는데도 통과해야 한다
    assert cg.lanes["n_train"] < cg.lanes["n_infer"], "학습이 추론보다 아래에 놓였다"
    assert cg.lanes["n_rep"] == max(cg.lanes.values()), "종결 Output 이 맨 아래가 아니다"


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
