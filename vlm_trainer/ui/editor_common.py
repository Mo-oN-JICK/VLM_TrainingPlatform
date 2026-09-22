"""Editor 본체와 믹스인이 함께 쓰는 것들.

믹스인이 `api` 를 되부르면 순환 임포트가 된다. 양쪽이 보는 것만 여기 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from ..core.errors import VlmtError

def _short(errors: Any) -> str:
    lines = [ln.strip() for ln in str(errors).splitlines() if ln.strip()]
    if lines and lines[0].endswith("게이트 위반:"):
        lines = lines[1:]  # 개수 머리말이 아니라 첫 번째 이유를 보여준다
    return lines[0] if lines else str(errors)

def _msg(exc: Exception) -> str:
    return str(exc) if isinstance(exc, VlmtError) else f"{type(exc).__name__}: {exc}"

@dataclass
class HistoryEntry:
    """한 번의 편집. 스펙이 텍스트라 시점 복원이 스냅샷 하나로 끝난다."""

    label: str
    spec_hash: str
    spec: str = ""  # canonical YAML
    at: str = ""
    diff: List[str] = field(default_factory=list)
    past: bool = False  # 이전 세션에서 남은 시점

