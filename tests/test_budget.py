"""Phase 3 완료 조건 — 자원 예산 게이트(G4)."""

from __future__ import annotations

import argparse
import os

import pytest

from vlm_trainer.core.compiler import compile_project
from vlm_trainer.engine import budget as budget_mod
from vlm_trainer.plugins.base import resolve_backbone
from vlm_trainer.train.config import DEVICES, TrainerConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJECT = os.path.join(ROOT, "solutions", "dummy_ecg", "projects", "01_dummy", "project.yaml")
TRAINER = os.path.join(ROOT, "solutions", "dummy_ecg", "projects", "01_dummy", "trainer.yaml")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "solutions", "dummy_ecg", "data", "dummy", "index.jsonl")),
    reason="더미 데이터가 없다. python tools/make_dummy_dataset.py 를 먼저 실행하라.",
)


def _cfg() -> TrainerConfig:
    return TrainerConfig.load(TRAINER)


def _graph():
    cg = compile_project(PROJECT)
    return cg, budget_mod.train_nodes(cg)[0]


def test_vision_info_comes_from_the_port_types():
    """이미지 개수와 해상도를 학습을 돌려보지 않고 그래프에서 읽는다."""
    cg, tn = _graph()
    images, hw = budget_mod.vision_from_graph(cg, tn)
    assert images == 3  # list.concat.max_n = 파형 1장 + crop 2장
    assert hw == (448, 448)


def test_default_config_fits_on_the_12gb_profile():
    cg, tn = _graph()
    res = budget_mod.for_graph(cg, _cfg(), tn)
    assert res.device == "rtx3060_12gb"
    assert res.limit == pytest.approx(11.0)
    assert res.ok, res.errors
    assert [s.name for s in res.stages] == ["projector_align", "lora_ft"]
    assert res.s_vision == 3 * 1 * 256
    # 단계마다 합계가 항목의 합과 일치한다
    for s in res.stages:
        assert s.total == pytest.approx(
            s.weights + s.grads + s.optimizer + s.activations + s.logits + s.context
        )


def test_full_finetune_of_a_7b_is_rejected_before_training():
    cg, tn = _graph()
    cfg = _cfg()
    cfg.backbone = "dummy-7b"
    cfg.quantization.mode = "none"
    cfg.stages[1].trainable["llm"] = "true"
    res = budget_mod.for_graph(cg, cfg, tn)

    assert not res.ok
    text = "\n".join(res.errors)
    assert "예산 초과" in text
    assert "초과 기여" in text
    assert "이 검사가 없었다면" in text
    # 가장 큰 기여가 가중치이고 대안이 함께 나온다
    top = res.stages[1].contributions[0]
    assert top.name in ("가중치", "옵티마이저", "그래디언트")
    assert any(c.fix for c in res.stages[1].contributions[:3])


def test_quantization_and_lora_bring_it_back_under_budget():
    cg, tn = _graph()
    cfg = _cfg()
    cfg.backbone = "dummy-7b"
    cfg.quantization.mode = "none"
    cfg.stages[1].trainable["llm"] = "true"
    assert not budget_mod.for_graph(cg, cfg, tn).ok

    cfg.quantization.mode = "nf4"
    cfg.stages[1].trainable["llm"] = "lora"
    cfg.stages[0].per_device = 1
    cfg.stages[0].grad_checkpointing = True
    assert budget_mod.for_graph(cg, cfg, tn).ok


def test_device_profiles_are_separate():
    cg, tn = _graph()
    cfg = _cfg()
    cfg.backbone = "dummy-7b"
    cfg.stages[1].trainable["llm"] = "true"
    cfg.stages[1].optimizer = "adamw_bnb_8bit"

    cfg.budget.device = "rtx3060_12gb"
    small = budget_mod.for_graph(cg, cfg, tn)
    cfg.budget.device = "rtx4090_24gb"
    big = budget_mod.for_graph(cg, cfg, tn)

    assert small.limit == pytest.approx(DEVICES["rtx3060_12gb"]["vram_gb"] - 1.0)
    assert big.limit == pytest.approx(DEVICES["rtx4090_24gb"]["vram_gb"] - 1.5)
    assert not small.ok and small.limit < big.limit


def test_context_overflow_is_rejected_not_truncated():
    cg, tn = _graph()
    cfg = _cfg()
    cfg.sequence.max_len = 512  # 비전 768 + 텍스트만으로도 넘는다
    res = budget_mod.for_graph(cg, cfg, tn)
    assert not res.ok
    text = "\n".join(res.errors)
    assert "시퀀스 길이 초과" in text
    assert "잘라내지 않고 거부한다" in text


def test_backbone_context_limit_is_checked():
    cg, tn = _graph()
    cfg = _cfg()
    cfg.vision.max_images_per_sample = 64  # 64 x 256 = 16384 > dummy-2b 컨텍스트 8192
    cfg.sequence.max_len = 65536
    res = budget_mod.for_graph(cg, cfg, tn)
    assert any("백본 컨텍스트 초과" in e for e in res.errors)


def test_declared_text_tokens_smaller_than_measured_is_rejected():
    """추정이 실측보다 작으면 예산 계산 전체가 낙관적으로 기운다."""
    cg, tn = _graph()
    cfg = _cfg()
    cfg.sequence.text_tokens = 100
    res = budget_mod.for_graph(cg, cfg, tn, measured_text_tokens=400)
    assert any("실측" in e and "작다" in e for e in res.errors)


def test_sensitivity_reports_what_actually_helps():
    cg, tn = _graph()
    cfg = _cfg()
    cfg.backbone = "dummy-7b"
    cfg.quantization.mode = "none"
    cfg.stages[1].trainable["llm"] = "true"
    res = budget_mod.for_graph(cg, cfg, tn)

    labels = dict(res.sensitivity)
    assert any("nf4" in k for k in labels), labels
    assert min(labels.values()) < 0  # 줄여 주는 선택지가 있다


def test_lora_params_scale_with_rank_and_targets():
    spec = resolve_backbone("dummy-2b").spec()
    small = spec.lora_params(16, ("q_proj", "k_proj"))
    large = spec.lora_params(32, ("q_proj", "k_proj"))
    more = spec.lora_params(16, ("q_proj", "k_proj", "gate_proj"))
    assert large == pytest.approx(small * 2)
    assert more > small
    # 전체 파라미터에 비하면 아주 작아야 한다
    assert spec.lora_params(32, ("q_proj", "k_proj", "v_proj", "o_proj")) < spec.params_total * 0.02


def test_run_refuses_to_start_when_over_budget(tmp_path, monkeypatch, capsys):
    """G4에 걸리면 Output 노드는 실행되지 않는다."""
    from vlm_trainer.cli import main as cli

    over = tmp_path / "trainer_over.yaml"
    cfg_text = open(TRAINER, encoding="utf-8").read()
    cfg_text = cfg_text.replace("backbone: dummy-2b", "backbone: dummy-7b").replace(
        "  mode: nf4", "  mode: none"
    )
    over.write_text(cfg_text, encoding="utf-8")

    real_load = TrainerConfig.load
    monkeypatch.setattr(TrainerConfig, "load", staticmethod(lambda p: real_load(str(over))))

    args = argparse.Namespace(
        spec=PROJECT,
        nodes=[],
        set=[],
        limit=1,
        split="",
        cache_dir=str(tmp_path / "cache"),
        no_cache=True,
        run_id="test",
        device="",
        skip_budget=False,
        what_if=[],
    )
    rc = cli.cmd_run(args)
    err = capsys.readouterr().err
    assert rc == 4
    assert "예산 초과" in err
    assert "학습을 시작하지 않았다" in err
    assert not (tmp_path / "dataset").exists()


# ── 예산이 실측보다 낮으면 게이트가 무의미하다 ──────────────────────────
def test_the_vision_tower_holds_activations_too():
    """비전 타워가 예산 모형에서 통째로 빠져 있었다. Qwen2-VL 에서 그 타워는
    32층 x 1280 이고 이미지당 패치 1024개를 붙드는데, LLM 쪽(28층 x 1536, 304토큰)
    활성화보다 크다. 빠뜨리면 예산이 조용히 낙관적으로 기운다."""
    from vlm_trainer.engine import budget as B

    big = B._act(b=2, s_len=1024, h=1280, layers=32, checkpointing=False)
    llm = B._act(b=2, s_len=304, h=1536, layers=28, checkpointing=False)
    assert big > llm, "비전 타워가 LLM 보다 작게 잡혔다 — 숫자를 다시 보라"


def test_a_conv_only_tower_adds_nothing():
    """층이 없으면 0이다. `tiny-vlm` 의 비전 타워는 conv 한 겹이라 모델링할
    트랜스포머가 없고, 그 경우 이 항이 예산에 끼어들면 안 된다."""
    from vlm_trainer.engine import budget as B

    assert B._act(b=4, s_len=1024, h=0, layers=0, checkpointing=False) == 0.0


def test_the_tower_sequence_is_longer_than_what_reaches_the_llm():
    """spatial merge 가 패치를 합치기 **전**의 길이다 — Qwen2-VL 에서 256 토큰은
    타워 안에서 1024 패치였다. 제곱을 빼먹으면 비전 활성화를 네 배 낮게 잡는다."""
    from vlm_trainer.plugins.base import BackboneSpec

    s = BackboneSpec(id="x", params_total=0, params_by_group={}, n_layers=1, hidden=1,
                     intermediate=1, vocab=1, tokens_per_tile=256, max_context=1,
                     spatial_merge=2)
    assert s.vision_seq(256) == 1024

    flat = BackboneSpec(id="y", params_total=0, params_by_group={}, n_layers=1, hidden=1,
                        intermediate=1, vocab=1, tokens_per_tile=256, max_context=1)
    assert flat.vision_seq(256) == 256, "merge 가 없으면 합쳐지지 않는다"


def test_checkpointing_is_a_saving_only_if_someone_actually_turns_it_on():
    """예산이 이 스위치를 켠 것으로 보고 메모리를 깎는다. 선언만 받아 놓고 모델에
    적용하지 않으면 예산은 **일어나지 않는 절약**을 빼고, 학습은 약속보다 많이 쓴다.
    실측으로 겪었다 — Qwen2-VL `lora_ft` 예산 6.2 GB / 실측 11.58 GB."""
    import inspect

    from vlm_trainer.train import loop as loop_mod

    src = inspect.getsource(loop_mod.train)
    assert "set_grad_checkpointing" in src, "예산이 깎는 것을 루프가 켜지 않는다"


def test_the_switch_reports_whether_it_is_on_not_whether_it_was_called():
    """끄기에 성공한 것을 "켜짐" 으로 적으면 보고서가 거짓이 된다."""
    from vlm_trainer.plugins.base import BackboneAdapter

    class _M:
        def gradient_checkpointing_enable(self):
            pass

        def gradient_checkpointing_disable(self):
            pass

    assert BackboneAdapter.set_grad_checkpointing(_M(), True) is True
    assert BackboneAdapter.set_grad_checkpointing(_M(), False) is False
    # 켤 줄 모르는 백본은 False. 루프가 그것을 보고 경고한다
    assert BackboneAdapter.set_grad_checkpointing(object(), True) is False


def test_a_stage_handoff_does_not_copy_the_checkpoint_onto_the_gpu():
    """`map_location=dev` 로 읽으면 2B 모델에서 체크포인트 사본 4.25 GB 가 GPU 에
    따로 생기고, `state` 를 놓지 않는 한 그 단계가 끝날 때까지 붙잡혀 있다.
    예산이 6.4 GB 인 자리에서 실측 10.11 GB 가 나온 이유가 이것이었다."""
    import ast
    import inspect
    import textwrap

    from vlm_trainer.train import loop as loop_mod

    # 주석에 그 글자가 적혀 있을 수 있으므로 **호출을 파싱해서** 본다.
    tree = ast.parse(textwrap.dedent(inspect.getsource(loop_mod.train)))
    loads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.load"]
    assert loads, "체크포인트를 읽는 자리를 못 찾았다 — 이 테스트가 낡았다"
    for call in loads:
        where = next((k.value for k in call.keywords if k.arg == "map_location"), None)
        assert isinstance(where, ast.Constant) and where.value == "cpu", (
            f"체크포인트를 GPU 로 읽는다: {ast.unparse(call)}")

    # 읽은 것을 놓아야 한다. 놓지 않으면 그 단계가 끝날 때까지 붙잡혀 있다.
    freed = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Delete)
                and any(ast.unparse(t) == "state" for t in n.targets))
    assert freed >= len(loads), f"읽은 체크포인트 {len(loads)}개 중 {freed}개만 놓는다"
