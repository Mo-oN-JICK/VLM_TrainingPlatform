"""사람이 읽는 표기. 같은 값이 어디서든 같게 보이게 하는 자리다.

이 파일이 있는 이유: 밀리초를 사람 말로 바꾸는 코드가 **세 벌** 있었다.
`ui/layout.py`(카드), `ui/app/progress.py`(툴바), `cli/main.py`(터미널). 셋이 우연히
일치하고 있었고, 한 곳만 고치면 같은 작업이 창에서는 `1.2s`, 터미널에서는 `1200ms` 로
보이게 된다. 그러면 사람은 둘 중 하나가 틀렸다고 읽는다.

코어에 두는 것은 의존성을 늘리지 않는다 — 표준 라이브러리조차 쓰지 않는다.
"""

from __future__ import annotations


def ms(value: float) -> str:
    """밀리초를 사람이 읽는 단위로.

    1초 미만은 밀리초, 1분 미만은 소수 한 자리 초, 그 위는 분과 초.
    4자리 밀리초(`42000ms`)는 눈으로 자릿수를 세게 만들어 쓰지 않는다.
    """
    if value < 1000:
        return f"{value:.0f}ms"
    if value < 60_000:
        return f"{value / 1000:.1f}s"
    total = int(value / 1000)
    minutes, seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    return f"{minutes // 60}h {minutes % 60:02d}m"
