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


# ── 이미 학습해 둔 모델 들여오기 ────────────────────────────────────────
#
# 모델을 내보내는 노드가 `train.vlm_trainer` 뿐이면, 쓸 수 있는 것은 **같은 그래프 안에서
# 방금 학습한 것** 하나뿐이다. 어려운 일을 단계로 쪼개 단계마다 모델을 학습하는 방식이
# 그래프에서 표현되지 않고, 단계마다 처음부터 다시 학습해야 한다.

def test_a_second_node_can_hand_a_model_to_inference():
    d = resolve("source.model@1.0.0")
    assert d.kind is NodeKind.INPUT
    assert not d.inputs and "model" in d.outputs


def test_it_declares_a_fingerprint_because_it_reads_the_outside(tmp_path):
    """Input 은 `fingerprint()` 를 구현해야 한다(C4). 폴더를 갈아 끼우면 캐시가
    갈려야 하고, 그것을 아는 것은 이 노드뿐이다."""
    from vlm_trainer.core.node import Node, RunCtx

    d = resolve("source.model@1.0.0")
    assert d.impl.fingerprint is not Node.fingerprint

    impl, ctx = d.impl(), RunCtx(run_id="r", node_id="n_model")
    p = d.build_params({"dir": str(tmp_path), "backbone": ""})
    before = impl.fingerprint(ctx, p)
    (tmp_path / "inference_contract.json").write_text('{"backbone": {"id": "x"}}',
                                                      encoding="utf-8")
    assert impl.fingerprint(ctx, p) != before, "폴더가 바뀌었는데 지문이 그대로다"


def test_the_backbone_is_read_from_the_contract_not_guessed(tmp_path):
    """bf16 으로 학습한 가중치를 다른 자료형으로 열면 답이 조용히 나빠지고,
    그것을 모델 탓으로 돌리게 된다."""
    import json

    from vlm_trainer.core.node import RunCtx

    (tmp_path / "inference_contract.json").write_text(
        json.dumps({"backbone": {"id": "tiny-vlm", "dtype": "bf16"}}), encoding="utf-8")
    d = resolve("source.model@1.0.0")
    got = d.impl().run(RunCtx(run_id="r", node_id="n_model"),
                       d.build_params({"dir": str(tmp_path), "backbone": ""}))
    assert got["model"]["backbone"] == "tiny-vlm"


def test_a_missing_folder_says_so_instead_of_answering_from_nothing(tmp_path):
    from vlm_trainer.core.node import RunCtx

    d = resolve("source.model@1.0.0")
    with pytest.raises(NodeError, match="모델 폴더가 없다"):
        d.impl().run(RunCtx(run_id="r", node_id="n_model"),
                     d.build_params({"dir": str(tmp_path / "없음"), "backbone": ""}))


def test_an_unknown_backbone_is_refused_rather_than_guessed(tmp_path):
    from vlm_trainer.core.node import RunCtx

    d = resolve("source.model@1.0.0")
    with pytest.raises(NodeError, match="백본을 알 수 없다"):
        d.impl().run(RunCtx(run_id="r", node_id="n_model"),
                     d.build_params({"dir": str(tmp_path), "backbone": ""}))


def test_a_graph_with_no_trainer_at_all_compiles(tmp_path):
    """이것이 요점이다. 학습 노드를 빼고 이미 만들어 둔 모델로 추론만 도는 그래프가
    게이트를 지나야 한다.

    `external_call` 배치 검사가 여기서 걸리면 안 된다 — 모델을 받아 도는 노드는
    굽기 단계에 있을 수 없으므로 "경계 앞에 두라" 는 요구가 **만족 불가능**하다.
    못 지킬 것을 요구하는 게이트는 사람이 믿지 않게 된다.
    """
    import shutil

    import yaml

    from vlm_trainer.core.compiler import compile_project

    shutil.copytree(os.path.join(ROOT, "solutions"), tmp_path / "solutions")
    spec = tmp_path / "solutions" / "vlm_open" / "projects" / "01_open" / "project.yaml"
    d = yaml.safe_load(spec.read_text(encoding="utf-8"))

    drop = {"n_train", "n_sample"}
    d["nodes"] = [n for n in d["nodes"] if n["id"] not in drop]
    d["nodes"].append({"id": "n_model", "type": "source.model@1.0.0",
                       "params": {"dir": "some/where"}})
    d["edges"] = [e for e in d["edges"]
                  if e["from"].split(":")[0] not in drop and e["to"].split(":")[0] not in drop]
    d["edges"].append({"from": "n_model:model", "to": "n_infer:model"})
    d["materialize"] = {"boundary": ["n_guard"]}
    spec.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")

    cg = compile_project(str(spec))
    assert "n_train" not in cg.nodes and cg.nodes["n_infer"].inputs["model"] == "n_model:model"


def test_a_node_error_survives_the_isolated_worker():
    """`external_call` 노드는 별도 프로세스에서 돈다. `NodeError` 가 그 경계를 못 넘으면
    실패가 전부 "워커 프로세스가 죽었다" 로 바뀌고, 네 줄짜리 진단이 통째로 사라진다 —
    사람은 네이티브 크래시를 의심하며 엉뚱한 곳을 판다."""
    import pickle

    e = NodeError("n_infer", cause="체크포인트가 없다", port="model",
                  sample_key="o0100", hint="  먼저 학습을 돌려라.")
    back = pickle.loads(pickle.dumps(e))
    assert back.node_id == "n_infer" and back.port == "model"
    assert back.sample_key == "o0100" and "먼저 학습" in back.hint
    assert back.cause == "체크포인트가 없다"


def test_an_exported_folder_is_opened_by_its_own_layout(tmp_path):
    """학습이 남긴 폴더는 단계별 하위 폴더에 `.pt` 가 있고, Export 한 폴더는 합쳐진
    모델이 평평하게 놓인다. 같은 코드로 열려고 하면 방금 내보낸 그 모델을 두고
    "체크포인트가 없다" 가 난다."""
    import json

    from vlm_trainer.train import checkpoint as ckpt

    assert ckpt.exported_format(str(tmp_path)) == ""
    (tmp_path / "inference_contract.json").write_text(
        json.dumps({"exported": {"format": "huggingface"}}), encoding="utf-8")
    assert ckpt.exported_format(str(tmp_path)) == "huggingface"
