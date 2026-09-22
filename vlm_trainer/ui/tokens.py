"""디자인 토큰 — 설계 문서 12의 실측값이 코드로 사는 곳.

값은 Mech-Vision 화면 캡처에서 픽셀 단위로 측정한 것이다. 캡처에서 확인할 수 없었던 것은
`OBSERVED`가 아니라 `ASSIGNED`로 표시한다. 지어낸 값과 관측한 값을 섞지 않기 위해서다.
"""

from __future__ import annotations


# ── 표면 (전부 실측) ────────────────────────────────────────────────────
SURFACE = {
    "chrome": "#121212",  # 타이틀/메뉴바
    "toolbar": "#282828",
    "panel": "#1F1F1F",
    "panel_alt": "#202020",
    "canvas": "#1A1A1A",
    "line": "#2A2A2A",  # ASSIGNED — 패널 구분선은 안티앨리어싱으로 분리 측정 불가
}

# ── 노드 카드 (전부 실측) ───────────────────────────────────────────────
NODE = {
    "bg": "#262A2F",
    "border": "#14C2AA",
    "bg_selected": "#3B5067",
    "border_selected": "#57F7E6",
    "border_selected_inner": "#447F8F",
    "title": "#FFFFFF",
    "muted": "#6F7478",
    "accent": "#0DC7AB",
}

# ── 카테고리 톤 (관측된 셋 + 같은 대역에서 배정한 것) ───────────────────
CATEGORY = {
    "Data Acquisition": "#514343",  # OBSERVED (Capture)
    "2D General Processing": "#414C5B",  # OBSERVED (Recognition/Pose)
    "Time Series Processing": "#41524C",  # ASSIGNED
    "Expert Models": "#4A4459",  # ASSIGNED
    "Adapters": "#3F4750",  # ASSIGNED
    "Prompt Assembly": "#4B4A3C",  # ASSIGNED
    "Answer Design": "#4C4147",  # ASSIGNED
    "Data Processing": "#3E4A4A",  # ASSIGNED
    "Evaluation": "#44484F",  # ASSIGNED
    "Visualization": "#464152",  # ASSIGNED
    "File": "#54576B",  # OBSERVED (Output)
    "Training": "#54576B",  # OBSERVED (Output)
    "System": "#3F4147",  # ASSIGNED
}
CATEGORY_DEFAULT = "#3F4750"

# ── 포트 타입 색 ────────────────────────────────────────────────────────
# 앞의 넷은 실측. 나머지는 같은 채도·명도 대역에서 배정했다.
PORT = {
    "Image": "#379B90",  # OBSERVED
    "ImageList": "#33999B",  # OBSERVED
    "Table": "#7F4164",  # OBSERVED (NumberList 계열)
    "Text": "#7D4B2E",  # OBSERVED (StringList 계열)
    "TimeSeries": "#4A7F9B",  # ASSIGNED
    "Regions": "#7F5B41",  # ASSIGNED
    "Schema": "#5C6B45",  # ASSIGNED
    "Sample": "#6B5A7F",  # ASSIGNED
    "Report": "#4F5B6B",  # ASSIGNED
    "Embedding": "#3F7F6A",  # ASSIGNED
    "Tokens": "#6B7F41",  # ASSIGNED
    "Mask": "#7F4F4F",  # ASSIGNED
    "Model": "#5A5A7F",  # ASSIGNED
}
PORT_DEFAULT = "#4A4A4A"

# 노드 상태 — 캡처에 나타나지 않아 전부 ASSIGNED.
# 색만으로 구분하지 않고 항상 라벨을 함께 붙인다.
STATE = {
    "pending": "#4A4A4A",
    "queued": "#4A5C6B",
    "running": "#0DC7AB",
    "success": "#3FB27F",
    "cached": "#5C7F9B",
    "failed": "#D9615A",
    "skipped": "#5A5A5A",
    "partial": "#D9A441",
}


def port_color(base: str, is_list: bool = False) -> str:
    if is_list and base == "Image":
        return PORT["ImageList"]
    return PORT.get(base, PORT_DEFAULT)


def category_color(category: str) -> str:
    return CATEGORY.get(category, CATEGORY_DEFAULT)


def wire_color(base: str, is_list: bool = False) -> str:
    """관측된 규칙: 배선 색 = 소스 포트 타입 색을 약간 어둡게 한 값."""
    c = port_color(base, is_list).lstrip("#")
    r, g, b = (int(c[i : i + 2], 16) for i in (0, 2, 4))
    return "#%02X%02X%02X" % (int(r * 0.82), int(g * 0.82), int(b * 0.82))
