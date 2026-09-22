"""네이티브 편집기(PySide6).

**이 패키지는 선택 사항이다.** PySide6 는 150MB 가 넘고, 학습만 돌리는 PC 나 헤드리스
서버에는 있을 이유가 없다. 그래서 `vlm_trainer` 어디에서도 이것을 최상위에서 임포트하지
않는다 — `compile`/`dryrun`/`budget`/`run`/`materialize`/`train`/`view` 는 PySide6 없이
전부 돌아야 하고, 그 경계를 테스트가 지킨다.

없을 때는 조용히 실패하지 않고 무엇을 설치해야 하는지 말한다.
"""

from __future__ import annotations

from typing import Tuple

INSTALL_HINT = (
    "편집기를 열려면 PySide6 가 필요하다.\n"
    "  설치: .venv\\Scripts\\python.exe -m pip install -e .[app]\n"
    "\n"
    "  안 잡혔다면: 편집기만 못 열 뿐 나머지는 전부 동작한다 —\n"
    "    compile / dryrun / budget / run / materialize / train / view.\n"
    "  추정 낭비: 없음(아무것도 시작하지 않았다).\n"
    "  그래프를 보기만 하려면: vlmt view <project.yaml> --open"
)


def available() -> bool:
    """PySide6 가 쓸 수 있는 상태인가. 임포트를 시도해 본다 — 있다고 선언만 된 경우가 있다."""
    try:
        import PySide6  # noqa: F401
        from PySide6 import QtWidgets  # noqa: F401
    except Exception:
        return False
    return True


def launch(project_path: str, extra_modules: Tuple[str, ...] = ()) -> int:
    """편집기 창을 띄운다. PySide6 가 없으면 무엇을 해야 하는지 말하고 멈춘다."""
    if not available():
        raise SystemExit(INSTALL_HINT)
    from .window import launch as _launch

    return _launch(project_path, extra_modules)
