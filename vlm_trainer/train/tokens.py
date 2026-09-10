"""텍스트 토큰 수 — 재려면 재고, 못 재면 보수적으로 추정한다.

G4는 학습을 시작하기 전에 시퀀스 길이를 알아야 한다. 백본이 붙기 전에는 `chars_per_token`
비율로 환산할 수밖에 없었고, 그 비율은 언어와 토크나이저에 따라 두 배씩 틀린다.
토크나이저가 손에 있으면 세는 쪽이 언제나 낫다.

**추정이 실측보다 작으면 안 된다.** 셀 수 없을 때 낙관적인 값을 내면 G4가 통과시킨 학습이
컨텍스트를 넘겨 터진다. 그래서 못 세면 못 셌다고 말하고, 비율 환산에는 안전 여유를 얹는다.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# 토크나이저는 로드가 비싸다. 프로세스 안에서 한 번만 만든다.
_CACHE: Dict[str, Any] = {}

MEASURED = "tokenizer"
ESTIMATED = "chars_per_token"

# 비율 환산은 틀릴 때 작게 틀리는 쪽이 위험하다. 실측을 못 했을 때만 얹는 여유.
ESTIMATE_MARGIN = 1.15


def load_tokenizer(tokenizer_id: str) -> Optional[Any]:
    """transformers가 있고 토크나이저를 구할 수 있으면 돌려준다. 아니면 None.

    없다고 예외를 던지지 않는다 — 토크나이저가 없는 것은 정상 상태이고(코어는 stdlib + yaml),
    그때는 추정으로 내려가면 된다.
    """
    if not tokenizer_id:
        return None
    if tokenizer_id in _CACHE:
        return _CACHE[tokenizer_id]
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        _CACHE[tokenizer_id] = None
        return None
    try:  # 다운로드는 하지 않는다. 이미 받아 둔 것만 쓴다
        tok = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=True)
    except Exception:
        tok = None
    _CACHE[tokenizer_id] = tok
    return tok


def count(text: str, tokenizer_id: str) -> Optional[int]:
    """실제 토큰 수. 셀 수 없으면 None — 0이 아니다. 0은 '텍스트가 없다'는 뜻이다."""
    tok = load_tokenizer(tokenizer_id)
    if tok is None:
        return None
    try:
        return len(tok(text, add_special_tokens=True)["input_ids"])
    except Exception:
        return None


def estimate(chars: int, chars_per_token: float) -> int:
    """비율 환산 + 안전 여유. 실측이 없을 때만 쓴다."""
    if chars <= 0:
        return 0
    return int(chars / max(0.5, chars_per_token) * ESTIMATE_MARGIN + 0.5)


def measure(text: str, chars: int, cfg_seq: Any, tokenizer_id: str = "") -> Tuple[int, str]:
    """(토큰 수, 어떻게 얻었는지).

    출처를 함께 돌려주는 이유가 있다 — 예산 보고서가 "쟀다"와 "추정했다"를 같은 숫자로
    적으면 읽는 사람이 그 차이를 알 수 없다.
    """
    if text:
        n = count(text, tokenizer_id or getattr(cfg_seq, "tokenizer_id", ""))
        if n is not None:
            return n, MEASURED
    return estimate(chars, float(getattr(cfg_seq, "chars_per_token", 2.5))), ESTIMATED
