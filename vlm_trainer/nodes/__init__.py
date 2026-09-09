"""내장 노드 카탈로그.

이 모듈을 임포트하면 모든 내장 노드와 플러그인이 레지스트리에 등록된다.
새 노드 모듈을 추가하면 여기에도 임포트를 넣어야 CLI와 워커가 찾을 수 있다.
"""

from . import answer, dataset, expert, imaging, prompt, source, timeseries  # noqa: F401
from ..plugins import dummy_experts  # noqa: F401

__all__ = ["source", "imaging", "timeseries", "expert", "prompt", "answer", "dataset"]
