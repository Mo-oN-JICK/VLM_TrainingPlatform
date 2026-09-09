"""실물 HF 백본 어댑터 — 가중치를 열지 않고 config만으로 예산을 답한다."""

from __future__ import annotations

import json
import os

import pytest

from vlm_trainer.core.errors import RegistrationError
from vlm_trainer.plugins import hf_backbone as hfb
from vlm_trainer.plugins.base import resolve_backbone

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "data", "hf_fake_2b")


def test_spec_comes_from_config_json_alone():
    """G4가 학습 전에 호출한다. 이 시점에 5GB를 올리면 게이트의 의미가 사라진다."""
    s = hfb.spec_from_config(os.path.join(FAKE, "config.json"), "hf:fake")

    assert s.hidden == 1536 and s.n_layers == 28
    assert s.intermediate == 8960 and s.vocab == 151936
    assert s.max_context == 32768
    # (448/14)^2 = 1024 패치, spatial_merge 2 -> 1024/4
    assert s.tokens_per_tile == 256
    # 파라미터는 근사지만 보수적이어야 한다(2B급이면 2~3B 사이)
    assert 2.0e9 < s.params_total < 3.5e9
    assert set(s.params_by_group) == {"llm", "vision_tower", "projector"}
    assert s.params_by_group["llm"] > s.params_by_group["vision_tower"]


def test_hf_prefix_registers_lazily():
    cls = resolve_backbone(f"hf:{FAKE}")
    assert cls.spec().id.startswith("hf:")
    assert resolve_backbone(f"hf:{FAKE}") is cls  # 두 번째는 등록된 것을 돌려준다


def test_missing_model_says_what_to_do():
    with pytest.raises(RegistrationError) as e:
        hfb.find_config("hf:org/does-not-exist-xyz")
    msg = str(e.value)
    assert "config.json을 찾을 수 없다" in msg
    assert "huggingface-cli download" in msg
    assert "로컬 경로를 직접 지정" in msg


def test_config_key_variants_are_tolerated(tmp_path):
    """모델 계열마다 키 이름이 다르다. 흔한 것들을 훑는다."""
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"n_embd": 2048, "n_layer": 16, "vocab_size": 50000, "n_positions": 8192}),
        encoding="utf-8",
    )
    s = hfb.spec_from_config(str(d / "config.json"), "hf:variant")
    assert s.hidden == 2048 and s.n_layers == 16 and s.max_context == 8192
    assert s.intermediate == 2048 * 4  # 없으면 4배로 가정


def test_budget_uses_the_real_config(tmp_path):
    """같은 그래프·같은 스펙에서 backbone 한 줄만 바꾸면 예산이 다시 계산된다."""
    from vlm_trainer.engine import budget as budget_mod
    from vlm_trainer.train.config import TrainerConfig

    cfg = TrainerConfig.from_dict(
        {
            "backbone": f"hf:{FAKE}",
            "quantization": {"mode": "nf4"},
            "sequence": {"max_len": 4096, "text_tokens": 700},
            "stages": [
                {
                    "name": "lora_ft",
                    "trainable": {"vision_tower": "false", "projector": "true", "llm": "lora"},
                    "optimizer": "adamw_bnb_8bit",
                    "per_device": 1,
                    "grad_checkpointing": True,
                }
            ],
            "budget": {"device": "rtx3060_12gb"},
        }
    )
    fits = budget_mod.estimate(cfg, images=3)
    assert fits.tokens_per_tile == 256
    assert fits.s_vision == 3 * 256
    assert fits.ok, fits.errors

    # 양자화를 끄고 전체 미세조정하면 12GB에 들어가지 않는다
    cfg.quantization.mode = "none"
    cfg.stages[0].trainable["llm"] = "true"
    over = budget_mod.estimate(cfg, images=3)
    assert not over.ok and "예산 초과" in "\n".join(over.errors)


def test_training_entry_points_require_transformers():
    """transformers가 없으면 무엇을 설치해야 하는지 말한다 — 조용히 실패하지 않는다."""
    pytest.importorskip("torch")
    cls = resolve_backbone(f"hf:{FAKE}")
    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RegistrationError, match="transformers"):
            cls.build(object(), object())
    else:
        pytest.skip("transformers가 설치되어 있어 이 경로는 실물 모델로 검증해야 한다")


def test_module_groups_sorts_children_by_common_names():
    class _Fake:
        def named_children(self):
            return [("visual", "V"), ("multi_modal_projector", "P"), ("language_model", "L")]

    g = hfb.HFBackbone.module_groups(_Fake())
    assert g["vision_tower"] == ["V"]
    assert g["projector"] == ["P"]
    assert g["llm"] == ["L"]
