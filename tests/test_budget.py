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
