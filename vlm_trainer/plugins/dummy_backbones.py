"""더미 백본 어댑터.

실제 가중치 없이 숫자만 보고한다. 예산 게이트(G4)는 파라미터 개수·레이어·hidden만
있으면 검증되므로, 실물 모델은 Phase 5에서 붙인다.
수치는 공개된 2B/7B급 VLM의 전형적인 형상을 따른 것이며 실제 모델의 값이 아니다.
"""

from __future__ import annotations

from .base import BackboneAdapter, BackboneSpec, register_backbone


@register_backbone
class Dummy2B(BackboneAdapter):
    spec_data = BackboneSpec(
        id="dummy-2b",
        params_total=2.2e9,
        params_by_group={"vision_tower": 0.30e9, "projector": 0.04e9, "llm": 1.86e9},
        n_layers=28,
        hidden=1536,
        intermediate=8960,
        vocab=151_936,
        tokens_per_tile=256,
        max_context=8192,
        tokenizer_id="dummy-2b-tok",
    )


@register_backbone
class Dummy7B(BackboneAdapter):
    spec_data = BackboneSpec(
        id="dummy-7b",
        params_total=8.3e9,
        params_by_group={"vision_tower": 0.68e9, "projector": 0.11e9, "llm": 7.51e9},
        n_layers=28,
        hidden=3584,
        intermediate=18944,
        vocab=151_936,
        tokens_per_tile=256,
        max_context=32768,
        tokenizer_id="dummy-7b-tok",
    )
