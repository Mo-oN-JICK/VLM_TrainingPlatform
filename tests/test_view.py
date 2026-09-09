"""Phase 7 (읽기 전용 뷰어) — 캔버스 규약이 실제 그래프에서 지켜지는가."""

from __future__ import annotations

import os
import re

import pytest

from vlm_trainer.core.compiler import compile_project
from vlm_trainer.core.node import NodeKind
from vlm_trainer.ui import render as render_mod
from vlm_trainer.ui import tokens as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJECT = os.path.join(ROOT, "solutions", "dummy_ecg", "projects", "01_dummy", "project.yaml")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "solutions", "dummy_ecg", "data", "dummy", "index.jsonl")),
    reason="더미 데이터가 없다. python tools/make_dummy_dataset.py 를 먼저 실행하라.",
)


@pytest.fixture(scope="module")
def cg():
    return compile_project(PROJECT)


@pytest.fixture(scope="module")
def page(cg):
    return render_mod.render(cg, title="테스트")


# ── 캔버스 규약 ─────────────────────────────────────────────────────────


def test_input_is_top_lane_and_output_is_bottom(cg):
    placed = render_mod._layout(cg)
    tops = [p.y for nid, p in placed.items() if cg.nodes[nid].kind is NodeKind.INPUT]
    bots = [p.y for nid, p in placed.items() if cg.nodes[nid].kind is NodeKind.OUTPUT]
    mids = [p.y for nid, p in placed.items() if cg.nodes[nid].kind is NodeKind.PROCESSING]

    assert len(set(tops)) == 1, "Input 노드가 한 레인에 모여 있지 않다"
    assert max(tops) < min(mids), "Input이 최상단 레인이 아니다"
    assert min(bots) > max(mids), "Output이 최하단 레인이 아니다"


def test_flow_is_downward_for_every_edge(cg):
    """배선은 언제나 위에서 아래로 간다. 이 규약이 순환을 구조적으로 막는다."""
    placed = render_mod._layout(cg)
    for e in cg.edges:
        assert placed[e.src_node].y < placed[e.dst_node].y, f"{e.src} -> {e.dst} 가 위로 향한다"


def test_cards_do_not_overlap(cg):
    placed = render_mod._layout(cg)
    boxes = [(p.x, p.y, p.x + render_mod.CARD_W, p.y + render_mod.CARD_H) for p in placed.values()]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            overlap = a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
            assert not overlap, f"카드가 겹친다: {a} {b}"


# ── 3분류의 시각적 구분 ─────────────────────────────────────────────────


def test_three_kinds_are_distinguished_by_shape(cg, page):
    n_in = sum(1 for i in cg.order if cg.nodes[i].kind is NodeKind.INPUT)
    n_out = sum(1 for i in cg.order if cg.nodes[i].kind is NodeKind.OUTPUT)
    assert page.count('class="node inp"') == n_in
    assert page.count('class="node out"') == n_out
    # 포트가 있는 변이 곧 분류의 표식이다 — Input은 상단 칩이, Output은 하단 칩이 없다
    assert ".inp .ports.top,.out .ports.bot{display:none}" in page.replace("\n", "")


def test_every_edge_gets_a_wire(cg, page):
    assert page.count("<path d=") == len(cg.edges)


def test_wire_takes_the_source_port_type_color(cg, page):
    """관측된 규칙: 배선 색 = 소스 포트 타입 색을 약간 어둡게 한 값."""
    e = next(e for e in cg.edges if e.src_node == "n_ts")  # TimeSeries 출력
    assert T.wire_color("TimeSeries") in page
    assert T.wire_color("Image") in page  # 이미지 배선도 있다
    assert T.wire_color("TimeSeries") != T.wire_color("Image")


# ── 포트 칩 ─────────────────────────────────────────────────────────────


def test_chips_use_the_measured_port_colors(page):
    assert T.PORT["Image"] in page and T.PORT["ImageList"] in page
    assert T.PORT["Text"] in page and T.PORT["Table"] in page


def test_chips_show_resolved_types_not_declarations(page):
    """컴파일이 끝나면 제네릭은 남아 있지 않다. 칩에도 그렇게 보여야 한다."""
    chips = re.findall(r'<span class="t">&lt;(.*?)&gt;</span>', page)
    assert chips
    assert not [c for c in chips if c.startswith("?")], f"제네릭이 남은 칩: {set(chips)}"
    assert "ImageList" in chips and "TimeSeries" in chips


def test_every_node_and_its_ref_appear(cg, page):
    for nid in cg.order:
        assert nid in page
        assert cg.nodes[nid].ref in page


def test_header_reports_the_gate_facts(cg, page):
    assert cg.spec_hash in page
    assert f"노드 {len(cg.nodes)}" in page
    assert cg.runtime_profile in page


# ── 토큰 ────────────────────────────────────────────────────────────────


def test_measured_tokens_are_not_edited_by_accident():
    """설계 문서 12의 실측값. 바꾸려면 캡처를 다시 재야 한다."""
    assert T.SURFACE["canvas"] == "#1A1A1A"
    assert T.SURFACE["panel"] == "#1F1F1F"
    assert T.NODE["bg"] == "#262A2F"
    assert T.NODE["border"] == "#14C2AA"
    assert T.NODE["bg_selected"] == "#3B5067"
    assert T.PORT["Image"] == "#379B90"
    assert T.PORT["Table"] == "#7F4164"
    assert T.PORT["Text"] == "#7D4B2E"


def test_write_produces_a_standalone_file(cg, tmp_path):
    path = render_mod.write(cg, str(tmp_path / "g.html"))
    assert os.path.exists(path)
    body = open(path, encoding="utf-8").read()
    assert body.startswith("<!doctype html>")
    # 자체 완결이어야 한다 — 외부 스크립트나 스타일시트를 부르지 않는다
    assert "<script" not in body and "<link" not in body
