"""Phase 5 완료 조건 — freeze 정책, 다단계 학습, 재개, 추론 계약, 프로파일 검사."""

from __future__ import annotations

import json
import os

import pytest

from vlm_trainer.core.compiler import compile_project
from vlm_trainer.engine import materialize as mat
from vlm_trainer.engine import samples as samples_mod
from vlm_trainer.engine.runner import RunOptions, execute
from vlm_trainer.train.config import TrainerConfig

torch = pytest.importorskip("torch", reason="Phase 5는 torch가 필요하다")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPEC_DIR = os.path.join(ROOT, "solutions", "dummy_ecg", "projects", "01_dummy")
PROJECT = os.path.join(SPEC_DIR, "project.yaml")
TINY = os.path.join(SPEC_DIR, "trainer_tiny.yaml")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "solutions", "dummy_ecg", "data", "dummy", "index.jsonl")),
    reason="더미 데이터가 없다. python tools/make_dummy_dataset.py 를 먼저 실행하라.",
)


def _cfg(**over) -> TrainerConfig:
    c = TrainerConfig.load(TINY)
    for s in c.stages:
        s.epochs = over.get("epochs", 1)
    return c


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    """작은 물질화 산출물 하나를 만들어 모듈 전체가 함께 쓴다."""
    d = tmp_path_factory.mktemp("phase5")
    cg = compile_project(PROJECT)
    space = samples_mod.load(cg.sample_space, SPEC_DIR)
    rep = mat.materialize(
        cg,
        space,
        mat.MaterializeOptions(out_dir=str(d / "materialized"), shard_size=2, limit=4),
        RunOptions(run_id="p5t", cache_dir=str(d / "cache"), spec_dir=SPEC_DIR),
    )
    assert rep.ok and rep.written == 4
    return {"dir": str(d), "materialized": rep.out_dir, "cg": cg}


# ── freeze 정책 ─────────────────────────────────────────────────────────


def test_freeze_matches_the_declaration():
    from vlm_trainer.plugins.base import resolve_backbone
    from vlm_trainer.train import freeze as fz

    adapter = resolve_backbone("tiny-vlm")
    cfg = _cfg()
    model = adapter.build(cfg, cfg.stages[0])

    rep = fz.apply(model, cfg.stages[0], adapter)  # projector만 true
    assert fz.verify(model, cfg.stages[0], adapter) == []
    assert rep.trainable["projector"] > 0
    assert rep.trainable["vision_tower"] == 0 and rep.trainable["llm"] == 0
    assert 0 < rep.ratio < 0.2


def test_lora_trains_only_the_adapters():
    from vlm_trainer.plugins.base import resolve_backbone
    from vlm_trainer.train import freeze as fz

    adapter = resolve_backbone("tiny-vlm")
    cfg = _cfg()
    model = adapter.build(cfg, cfg.stages[1])
    rep = fz.apply(model, cfg.stages[1], adapter)  # llm: lora

    assert len(rep.lora_modules) == 16, rep.lora_modules  # 4 레이어 x 4 타깃
    assert fz.verify(model, cfg.stages[1], adapter) == []

    lora_params = [p for n, p in model.llm.named_parameters() if ".a." in n or ".b." in n]
    base_params = [p for n, p in model.llm.named_parameters() if ".a." not in n and ".b." not in n]
    assert all(p.requires_grad for p in lora_params)
    assert not any(p.requires_grad for p in base_params), "베이스 가중치가 열려 있다"


def test_declaring_nothing_trainable_is_an_error(baked):
    from vlm_trainer.train import loop as loop_mod

    cfg = _cfg()
    cfg.stages = cfg.stages[:1]
    cfg.stages[0].trainable = {"vision_tower": "false", "projector": "false", "llm": "false"}
    with pytest.raises(ValueError, match="학습할 파라미터가 없다"):
        loop_mod.train(cfg, baked["materialized"], os.path.join(baked["dir"], "t_none"), device="cpu")


# ── 학습 루프 ───────────────────────────────────────────────────────────


def test_two_stage_training_completes(baked):
    from vlm_trainer.train import loop as loop_mod

    out = os.path.join(baked["dir"], "train")
    rep = loop_mod.train(_cfg(epochs=2), baked["materialized"], out, device="cpu")

    assert rep.ok and rep.samples == 4
    assert [s.name for s in rep.stages] == ["projector_align", "lora_ft"]
    for s in rep.stages:
        assert s.steps > 0 and s.last_loss > 0
        assert os.path.exists(s.ckpt)
    # 2단계는 LoRA 어댑터를 실제로 주입했다
    assert rep.stages[1].lora_modules == 16
    # 체크포인트에 설정 지문이 함께 남는다
    state = torch.load(rep.stages[0].ckpt, map_location="cpu", weights_only=False)
    assert state["backbone"] == "tiny-vlm" and state["config"]["stages"][0]["name"] == "projector_align"


def test_resume_continues_from_the_checkpoint(baked):
    from vlm_trainer.train import loop as loop_mod

    out = os.path.join(baked["dir"], "resume")
    cfg = _cfg(epochs=2)
    first = loop_mod.train(cfg, baked["materialized"], out, device="cpu", max_steps=1)
    assert first.stages[0].steps == 1

    second = loop_mod.train(cfg, baked["materialized"], out, device="cpu", resume=True)
    assert second.stages[0].resumed_from == 1, "체크포인트에서 이어받지 않았다"
    assert second.stages[0].steps > first.stages[0].steps


def test_budget_estimate_is_not_optimistic_against_the_real_peak(baked):
    """정적 추정이 실측보다 작으면 예산 게이트가 무의미해진다."""
    from vlm_trainer.engine import budget as budget_mod
    from vlm_trainer.train import loop as loop_mod

    if not torch.cuda.is_available():
        pytest.skip("GPU가 없으면 peak를 잴 수 없다")

    cfg = _cfg(epochs=1)
    cg = baked["cg"]
    tn = budget_mod.train_nodes(cg)[0]
    est = budget_mod.for_graph(cg, cfg, tn)
    rep = loop_mod.train(cfg, baked["materialized"], os.path.join(baked["dir"], "peak"), device="cuda")

    peak = max(s.peak_vram_gb for s in rep.stages)
    assert peak > 0
    assert max(s.total for s in est.stages) >= peak, "추정이 실측 peak보다 작다"


# ── 추론 계약 ───────────────────────────────────────────────────────────


def test_inference_graph_drops_the_answer_path(baked):
    from vlm_trainer.train import contract as contract_mod

    cg = baked["cg"]
    # 누설 차단 노드는 정답을 받으므로 추론에는 없다. 한 단계 위가 프롬프트 종단이다.
    assert contract_mod.prompt_terminus(cg) == "n_prompt_img"

    keep = contract_mod.inference_nodes(cg)
    assert "n_answer" not in keep and "n_validate" not in keep and "n_guard" not in keep
    assert "n_prompt_img" in keep and "n_images" in keep and "p_crop/n_crop" in keep

    spec = contract_mod.extract_inference_graph(cg)
    types = {n["type"] for n in spec["nodes"]}
    assert not any(t.startswith(("answer.", "train.")) for t in types)
    assert any(t.startswith("io.prompt_export") for t in types)


def test_inference_prompt_is_byte_identical_to_training(baked, tmp_path):
    """학습 때 쓴 프롬프트와 추론 그래프가 만든 프롬프트가 바이트 단위로 같아야 한다."""
    import yaml

    from vlm_trainer.train import contract as contract_mod
    from vlm_trainer.train import shards as shards_mod

    spec = contract_mod.extract_inference_graph(baked["cg"])
    path = os.path.join(SPEC_DIR, "_test_infer.yaml")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(spec, fh, allow_unicode=True, sort_keys=False)
        cg = compile_project(path)
        space = samples_mod.load(cg.sample_space, SPEC_DIR)

        baked_prompts = {r["id"]: r["prompt"] for r in shards_mod.iter_samples(baked["materialized"])}
        rows = [r for r in space.rows if str(r[space.key]) in baked_prompts]
        assert rows

        out_dir = str(tmp_path / "infer")
        for n in cg.nodes.values():
            if n.ref.startswith("io.prompt_export"):
                n.params["out_dir"] = out_dir
        execute(
            cg, space, rows,
            RunOptions(run_id="t", run_outputs=True, cache_dir=str(tmp_path / "c"), spec_dir=SPEC_DIR),
        )

        produced = {}
        with open(os.path.join(out_dir, "prompts.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                produced[rec["id"]] = rec["prompt"]

        assert set(produced) == set(baked_prompts)
        for key, text in baked_prompts.items():
            assert produced[key].encode("utf-8") == text.encode("utf-8"), f"{key} 프롬프트가 다르다"
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_contract_records_what_inference_needs(baked, tmp_path):
    from vlm_trainer.train import contract as contract_mod

    files = contract_mod.write(
        str(tmp_path / "ck"), baked["cg"], _cfg(), SPEC_DIR, vision_tokens=48, schema_id="dummy_stepwise"
    )
    with open(files["contract"], encoding="utf-8") as fh:
        c = json.load(fh)

    assert c["spec_hash"] == baked["cg"].spec_hash
    assert c["backbone"]["id"] == "tiny-vlm"
    assert c["image_slots"]["placeholder"] == "<image>"
    assert c["prompt_template"] and "{trend_slope}" in c["prompt_template"]
    assert c["knowledge_block"]["sha"] not in ("", "missing")
    assert c["answer_schema"]["id"] == "dummy_stepwise"
    assert c["expected_vision_tokens_per_sample"] == 48
    assert any(step["type"].startswith("adapt.image_resize") for step in c["image_pipeline"])


# ── 실행 프로파일 ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "field,value,needle",
    [
        ("attn_impl", "flash_attn2", "flash-attn은 Windows 휠이 없다"),
        ("offload", "disk", "offload를 none 또는 cpu"),
    ],
)
def test_windows_profile_rejects_unsupported_options(field, value, needle):
    cfg = _cfg()
    setattr(cfg, field, value)
    errs = cfg.profile_errors("windows_single_gpu")
    assert errs and needle in "\n".join(errs)
    assert not cfg.profile_errors("linux_multi_gpu")  # 다른 프로파일에는 적용되지 않는다


def test_budget_carries_the_profile_check(baked):
    from vlm_trainer.engine import budget as budget_mod

    cfg = _cfg()
    cfg.stages[0].optimizer = "deepspeed_adam"
    tn = budget_mod.train_nodes(baked["cg"])[0]
    res = budget_mod.for_graph(baked["cg"], cfg, tn)
    assert not res.ok
    assert any("DeepSpeed" in e for e in res.errors)
