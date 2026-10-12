"""표 양식 — 사용자가 채울 빈 CSV 의 첫 줄.

화면의 "표 양식 내려받기"와 CLI 가 같은 함수를 쓴다. 엑셀에서 열면 한글이 깨지지 않도록
UTF-8 BOM 을 붙이고, 줄바꿈은 엑셀이 쓰는 CRLF 로 쓴다.
"""

from __future__ import annotations

from typing import List, Sequence

from ..core.errors import SpecError


def header(*, key: str, image: str, question: str, answers: Sequence[str],
           split: str = "", group: str = "") -> List[str]:
    cols = [key, image, question, *answers]
    if split:
        cols.append(split)
    if group:
        cols.append(group)
    empty = [c for c in cols if not str(c).strip()]
    dup = sorted({c for c in cols if cols.count(c) > 1})
    if empty or dup or not answers:
        raise SpecError(
            f"표 양식의 열 이름이 맞지 않는다 (열 {cols}).\n"
            "  안 잡혔다면: 같은 이름이 둘이면 어느 열인지 정해지지 않고, 정답 열이 없으면 정답이 비어 학습된다.\n"
            "  열 이름을 서로 다르게 하고 정답 열을 하나 이상 넣으세요.")
    return cols


def template_csv(cols: Sequence[str]) -> bytes:
    return ("﻿" + ",".join(cols) + "\r\n").encode("utf-8")
