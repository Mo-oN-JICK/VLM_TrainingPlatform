"""이미지 자리표시자는 백본이 번역한다.

그래프는 `<image>` 하나만 안다. 그것을 모델의 실제 토큰으로 옮기는 것은 어댑터의 일이고,
그래야 `backbone:` 한 줄로 모델이 바뀐다.

**이 파일이 막는 것은 조용한 실패다.** Qwen2-VL 에 `<image>` 를 주면 프로세서가 거부하지
않는다. vocab 에 없으니 평범한 바이트로 쪼개지고, 이미지 토큰이 하나도 안 생긴 채
학습이 돈다. 손실은 내려가는데 모델은 그림을 보지 않는다. 예외도 경고도 없다.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

from vlm_trainer.plugins import hf_backbone as hfb
from vlm_trainer.plugins.base import GRAPH_PLACEHOLDER, BackboneAdapter

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "data", "hf_fake_2b")

IMG_ID = 900


class _Tok:
    """토크나이저 흉내. **자리표시자를 부풀리지 않는다** — 진짜도 그렇다."""

    padding_side = "left"
    pad_token_id = 0

    def __init__(self, known: List[str], lengths: Dict[str, int]) -> None:
        self._known, self._lengths = known, lengths

    def convert_tokens_to_ids(self, toks: Any) -> Any:
        if isinstance(toks, str):
            return IMG_ID if toks.startswith("<|IMG") else (1 if toks in self._known else -1)
        return [self.convert_tokens_to_ids(t) for t in toks]

    def __call__(self, text: str) -> Dict[str, List[int]]:
        return {"input_ids": [7] * self._lengths[text]}


class _Proc:
    def __init__(self, tok: _Tok, image_token: str) -> None:
        self.tokenizer, self.image_token = tok, image_token


def _adapter(proc: Any) -> type:
    """프로세서를 끼워 넣은 어댑터 한 벌. 실물 모델 없이 번역만 본다."""
    return type("Probe", (hfb.HFBackbone,), {"_proc": proc, "config_path": FAKE})


# ── 선언 ────────────────────────────────────────────────────────────────
def test_the_contract_asks_every_backbone_how_it_writes_an_image():
    """기본값은 "번역 없음"이다. 물어볼 자리가 있다는 것 자체가 요점이다 —
    없으면 어댑터마다 제 방식으로 몰래 처리하게 된다."""
    assert BackboneAdapter.image_placeholder() == GRAPH_PLACEHOLDER


def test_the_tiny_backbone_declares_the_same_spelling_it_encodes():
    """번역이 없다는 뜻이지 선언이 필요 없다는 뜻이 아니다."""
    torch = pytest.importorskip("torch")  # noqa: F841
    from vlm_trainer.plugins.tiny_backbone import TinyVlmAdapter, encode

    ph = TinyVlmAdapter.image_placeholder()
    assert ph == GRAPH_PLACEHOLDER
    assert encode(f"앞{ph}뒤").count(256) == 1, "선언한 글자를 제 토크나이저가 못 알아본다"


# ── 번역 ────────────────────────────────────────────────────────────────
def test_the_placeholder_is_read_off_the_processor_not_written_by_hand():
    """모델마다 다르다. 어댑터가 글자를 적어 두면 그 글자가 틀린 날 아무도 모른다."""
    cls = _adapter(_Proc(_Tok([], {}), "<|IMG_PAD|>"))
    assert cls.image_placeholder() == "<|IMG_PAD|>"


def test_vision_markers_are_used_only_when_the_vocabulary_has_them():
    """Qwen2-VL 은 자리표시자를 `<|vision_start|>…<|vision_end|>` 로 감싼다.
    그 토큰이 없는 모델에 감싸서 주면 다시 조용히 쪼개진다."""
    known = ["<|vision_start|>", "<|vision_end|>"]
    wrapped = _adapter(_Proc(_Tok(known, {}), "<|IMG_PAD|>")).image_placeholder()
    assert wrapped == "<|vision_start|><|IMG_PAD|><|vision_end|>"


def test_the_graph_keeps_writing_the_same_thing_for_every_backbone():
    """이것이 "`backbone:` 한 줄" 주장의 실물이다. 그래프의 `placeholder` 파라미터는
    백본이 뭐든 그대로다."""
    cls = _adapter(_Proc(_Tok([], {}), "<|IMG_PAD|>"))
    assert cls._retarget("부품은? <image>") == "부품은? <|IMG_PAD|>"
    assert BackboneAdapter.image_placeholder() == GRAPH_PLACEHOLDER


# ── 손실 ────────────────────────────────────────────────────────────────
def test_loss_lands_on_the_answer_even_though_the_image_token_expands():
    """프로세서가 자리표시자 하나를 타일 토큰 수백 개로 부풀린다(실측 31 -> 286).
    맨 토크나이저로 프롬프트 길이를 세면 그 차이만큼 **이미지 토큰에 손실이 걸린다** —
    모델에게 자기가 본 그림을 받아쓰라고 시키는 꼴이고, 그래도 손실은 내려간다.

    그리고 이 토크나이저는 **왼쪽으로 패딩한다.** 앞에서부터 가리면 프롬프트가 아니라
    패딩이 가려지고 정답은 멀쩡히 남아, 두 실수가 겹치면 아무 데도 티가 안 난다.
    """
    torch = pytest.importorskip("torch")

    prompt, answer = "부품은? <|IMG_PAD|>", "기어"
    tok = _Tok([], {prompt: 3})          # 부풀리기 전 3토큰: 부품은? / ? / 자리표시자
    cls = _adapter(_Proc(tok, "<|IMG_PAD|>"))

    # 왼쪽 패딩 2 + 프롬프트(2 + 이미지 4) + 정답 3 = 11
    ids = torch.tensor([[0, 0, 11, 12, IMG_ID, IMG_ID, IMG_ID, IMG_ID, 21, 22, 23]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    labels = cls._answer_only_labels(
        [{"prompt": prompt, "answer": answer}], {"input_ids": ids, "attention_mask": mask})

    kept = labels[0][labels[0] != -100].tolist()
    assert kept == [21, 22, 23], f"정답 세 토큰에만 걸려야 한다 — 실제 {kept}"
    assert IMG_ID not in kept, "이미지 토큰에 손실이 걸렸다"


def test_a_backbone_without_an_image_token_still_masks_the_prompt():
    """텍스트 전용 백본. 부풀릴 것이 없으니 맨 길이가 곧 프롬프트 길이다."""
    torch = pytest.importorskip("torch")

    prompt = "부품은?"
    cls = _adapter(_Proc(_Tok([], {prompt: 2}), ""))
    ids = torch.tensor([[11, 12, 21, 22]])
    mask = torch.tensor([[1, 1, 1, 1]])
    labels = cls._answer_only_labels(
        [{"prompt": prompt, "answer": "기어"}], {"input_ids": ids, "attention_mask": mask})
    assert labels[0][labels[0] != -100].tolist() == [21, 22]


# ── 실물 모델이 드러낸 것들 ─────────────────────────────────────────────
class _Lin:
    """`named_children` 만 흉내 내는 자리표시. 덩이 가르기는 이름만 본다."""

    def __init__(self, kids=()):
        self._kids = list(kids)

    def named_children(self):
        return list(self._kids)


def test_module_groups_reach_into_the_wrapper_that_holds_the_tower():
    """Qwen2-VL 은 최상위가 `model` 과 `lm_head` 뿐이고 비전 타워는 그 안에 있다.
    최상위만 훑으면 `projector` 가 **빈 목록**이 되고, `projector: true` 단계가
    "학습할 파라미터가 없다" 로 죽는다 — 가중치를 다 올린 뒤에."""
    merger, blocks, lm, head = _Lin(), _Lin(), _Lin(), _Lin()
    visual = _Lin([("blocks", blocks), ("merger", merger)])
    model = _Lin([("model", _Lin([("visual", visual), ("language_model", lm)])),
                  ("lm_head", head)])

    g = hfb.HFBackbone.module_groups(model)
    assert g["projector"] == [merger], "비전 타워 안의 프로젝터를 못 찾았다"
    assert blocks in g["vision_tower"] and merger not in g["vision_tower"]
    assert g["llm"] == [lm, head]


def test_the_projector_is_not_left_inside_the_frozen_tower():
    """`vision_tower: false` 와 `projector: true` 가 서로를 부정하면 freeze 검증이
    "false 로 선언했는데 학습 대상이 있다" 를 낸다."""
    merger = _Lin()
    visual = _Lin([("merger", merger)])
    g = hfb.HFBackbone.module_groups(_Lin([("visual", visual)]))
    assert merger in g["projector"] and merger not in g["vision_tower"]


def test_a_flat_model_is_left_flat():
    """껍데기가 없는 모델까지 파고들면 LLM 레이어 스물여덟 개가 최상위 덩이로 쏟아진다."""
    v, p, l = _Lin(), _Lin(), _Lin()
    g = hfb.HFBackbone.module_groups(
        _Lin([("visual", v), ("multi_modal_projector", p), ("language_model", l)]))
    assert g["vision_tower"] == [v] and g["projector"] == [p] and g["llm"] == [l]


def test_lora_adapters_are_born_in_the_base_layers_dtype():
    """기본값(float32)으로 만들면 bf16 모델의 첫 행렬곱에서
    `expected mat1 and mat2 to have the same dtype` 로 죽는다.
    `tiny-vlm` 은 float32 라 이 자리가 한 번도 드러나지 않았다."""
    torch = pytest.importorskip("torch")
    from torch import nn

    from vlm_trainer.train.freeze import LoRALinear

    base = nn.Linear(8, 8, dtype=torch.bfloat16)
    lora = LoRALinear(base, r=2, alpha=4)
    assert lora.a.weight.dtype is torch.bfloat16 and lora.b.weight.dtype is torch.bfloat16
    lora(torch.zeros(1, 8, dtype=torch.bfloat16))     # 죽지 않아야 한다


def test_the_loop_asks_the_adapter_how_to_call_the_model():
    """학습 루프에 `model(input_ids, images, ...)` 라고 적으면 그 줄은 특정 백본 하나의
    서명이고, 다른 백본을 끼우면 거기서 KeyError 로 죽는다 — 굽기가 다 끝난 뒤에."""
    import inspect

    from vlm_trainer.train import loop as loop_mod

    src = inspect.getsource(loop_mod.train)
    assert "adapter.forward(" in src
    assert 'batch["images"]' not in src, "루프가 다시 배치의 모양을 알기 시작했다"


def test_the_default_forward_spreads_whatever_collate_made():
    """HuggingFace 모델은 키워드로 받는다. 서명이 다른 백본만 이것을 덮어쓴다."""
    seen = {}

    class _Model:
        def parameters(self):
            return iter(())

        def __call__(self, **kw):
            seen.update(kw)
            return {"loss": 0.0}

    BackboneAdapter.forward(_Model(), {"input_ids": 1, "pixel_values": 2}, "cpu")
    assert seen == {"input_ids": 1, "pixel_values": 2}
