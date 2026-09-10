"""텍스트 토큰 수 — 재거나, 못 재면 보수적으로 추정한다.

G4가 이 숫자로 시퀀스 길이를 판단한다. 틀리는 방향이 중요하다: 작게 틀리면
통과시킨 학습이 컨텍스트를 넘겨 터지고, 크게 틀리면 들어갈 학습을 막을 뿐이다.
"""

from __future__ import annotations

import pytest

from vlm_trainer.train import tokens as tokens_mod
from vlm_trainer.train.config import Sequence


class FakeTokenizer:
    """공백으로 자르는 토크나이저. 실물이 없어도 배선은 검증된다."""

    def __call__(self, text, add_special_tokens=True):
        n = len(text.split())
        return {"input_ids": list(range(n + (2 if add_special_tokens else 0)))}


@pytest.fixture(autouse=True)
def clean_cache():
    tokens_mod._CACHE.clear()
    yield
    tokens_mod._CACHE.clear()


def test_no_tokenizer_is_none_not_zero():
    """0은 '텍스트가 없다'는 뜻이다. 못 셌다는 것과 섞이면 예산이 0을 믿는다."""
    assert tokens_mod.count("아무 글", "없는/토크나이저") is None


def test_an_empty_tokenizer_id_never_loads():
    assert tokens_mod.load_tokenizer("") is None


def test_estimate_leans_high():
    """비율 환산은 틀릴 때 큰 쪽으로 틀려야 한다."""
    plain = 1000 / 2.5
    assert tokens_mod.estimate(1000, 2.5) > plain
    assert tokens_mod.estimate(0, 2.5) == 0


def test_estimate_survives_a_nonsense_ratio():
    """0으로 나누지 않는다. 설정 파일에 0이 들어와도 예산 계산이 죽으면 안 된다."""
    assert tokens_mod.estimate(1000, 0.0) > 0


def test_measure_prefers_counting(monkeypatch):
    monkeypatch.setattr(tokens_mod, "load_tokenizer", lambda _id: FakeTokenizer())

    n, how = tokens_mod.measure("하나 둘 셋 넷", 12, Sequence(), "any/model")
    assert how == tokens_mod.MEASURED
    assert n == 6  # 낱말 4 + 특수 토큰 2


def test_measure_falls_back_when_it_cannot_count(monkeypatch):
    monkeypatch.setattr(tokens_mod, "load_tokenizer", lambda _id: None)

    n, how = tokens_mod.measure("하나 둘 셋 넷", 12, Sequence(chars_per_token=2.5), "any/model")
    assert how == tokens_mod.ESTIMATED
    assert n == tokens_mod.estimate(12, 2.5)


def test_no_text_means_estimate_even_with_a_tokenizer(monkeypatch):
    monkeypatch.setattr(tokens_mod, "load_tokenizer", lambda _id: FakeTokenizer())

    n, how = tokens_mod.measure("", 500, Sequence(), "any/model")
    assert how == tokens_mod.ESTIMATED and n > 0


def test_a_broken_tokenizer_does_not_take_the_budget_down(monkeypatch):
    class Broken:
        def __call__(self, *a, **k):
            raise RuntimeError("토크나이저가 깨졌다")

    monkeypatch.setattr(tokens_mod, "load_tokenizer", lambda _id: Broken())

    n, how = tokens_mod.measure("무언가", 30, Sequence(), "any/model")
    assert how == tokens_mod.ESTIMATED and n > 0


def test_the_tokenizer_is_loaded_once(monkeypatch):
    calls = []

    def fake_import(_id):
        calls.append(_id)
        return FakeTokenizer()

    monkeypatch.setattr(tokens_mod, "load_tokenizer", fake_import)
    tokens_mod.measure("가 나", 5, Sequence(), "m")
    tokens_mod.measure("다 라", 5, Sequence(), "m")
    assert len(calls) == 2  # measure는 매번 묻고, 캐시는 load_tokenizer 안에 있다

    tokens_mod._CACHE["m"] = FakeTokenizer()
    monkeypatch.undo()
    assert tokens_mod.load_tokenizer("m") is not None


# ── 예산 보고가 출처를 말한다 ───────────────────────────────────────────


def test_the_budget_names_where_the_token_count_came_from():
    """센 것과 환산한 것을 같은 말로 적으면 읽는 사람이 그 차이를 알 수 없다."""
    from vlm_trainer.engine import budget as budget_mod
    from vlm_trainer.train.config import TrainerConfig

    cfg = TrainerConfig()

    counted = budget_mod.estimate(cfg, images=1, measured_text_tokens=900, measured_how="tokenizer")
    assert counted.text_source == "토크나이저"
    assert "토크나이저로 센 값이다" in " ".join(counted.notes)

    converted = budget_mod.estimate(
        cfg, images=1, measured_text_tokens=900, measured_how="chars_per_token"
    )
    assert converted.text_source == "문자 환산"
    assert "안전 여유" in " ".join(converted.notes)

    assumed = budget_mod.estimate(cfg, images=1)
    assert assumed.text_source == "가정" and assumed.text_estimated


def test_a_declared_count_smaller_than_the_measured_one_is_refused():
    """추정이 실측보다 작으면 예산 전체가 낙관적으로 기운다 — 토크나이저가 붙어도 같다."""
    from vlm_trainer.engine import budget as budget_mod
    from vlm_trainer.train.config import Sequence, TrainerConfig

    cfg = TrainerConfig()
    cfg.sequence = Sequence(text_tokens=500)
    res = budget_mod.estimate(cfg, images=1, measured_text_tokens=1200, measured_how="tokenizer")
    assert any("작다" in e for e in res.errors)


def test_a_small_ratio_is_not_clamped():
    """바이트 토크나이저는 한글 한 글자가 3토큰이다 — 0.34가 맞는 값이지,
    걸러낼 값이 아니다. 바닥값이 이것을 덮으면 추정이 32% 낙관적으로 나온다."""
    assert tokens_mod.estimate(987, 0.34) > tokens_mod.estimate(987, 0.5)
    assert tokens_mod.estimate(987, 0.34) == int(987 / 0.34 * tokens_mod.ESTIMATE_MARGIN + 0.5)
