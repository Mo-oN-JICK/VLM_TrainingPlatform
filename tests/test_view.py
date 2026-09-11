"""Phase 7 (읽기 전용 뷰어) — 캔버스 규약이 실제 그래프에서 지켜지는가."""

from __future__ import annotations

import os
import re

import pytest

from vlm_trainer.core.compiler import compile_project
from vlm_trainer.ui.render import fold
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
    shown, edges = fold(cg)
    placed = render_mod._layout(shown, edges)
    tops = [p.y for nid, p in placed.items() if shown[nid].kind is NodeKind.INPUT]
    bots = [p.y for nid, p in placed.items() if shown[nid].kind is NodeKind.OUTPUT]
    mids = [p.y for nid, p in placed.items() if shown[nid].kind is NodeKind.PROCESSING]

    assert len(set(tops)) == 1, "Input 노드가 한 레인에 모여 있지 않다"
    assert max(tops) < min(mids), "Input이 최상단 레인이 아니다"
    assert min(bots) > max(mids), "Output이 최하단 레인이 아니다"


def test_flow_is_downward_for_every_edge(cg):
    """배선은 언제나 위에서 아래로 간다. 이 규약이 순환을 구조적으로 막는다.

    접힌 Procedure 상자도 예외가 아니다 — 상자 안이 여러 레인에 걸쳐 있어도
    밖에서 보이는 배선은 전부 아래로 향해야 한다."""
    shown, edges = fold(cg)
    placed = render_mod._layout(shown, edges)
    for src, dst in edges:
        a, b = src.split(":")[0], dst.split(":")[0]
        assert placed[a].y < placed[b].y, f"{src} -> {dst} 가 위로 향한다"


def test_cards_do_not_overlap(cg):
    placed = render_mod._layout(*fold(cg))
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
    """Procedure가 접혀 있으면 그 안에서 끝나는 배선은 상자 안으로 사라진다.
    밖에서 보이는 배선은 전부 그려져야 한다."""
    _, shown_edges = fold(cg)
    assert page.count("<path d=") == len(shown_edges)
    assert len(shown_edges) < len(cg.edges), "접힌 Procedure가 있으면 배선이 줄어든다"


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
    """보이는 상자 수와 게이트가 보는 노드 수를 둘 다 적는다 —
    하나만 적으면 카드 수와 헤더가 어긋나 보인다."""
    shown, edges = fold(cg)

    assert cg.spec_hash in page
    assert f"노드 {len(cg.nodes)}" in page
    assert f"상자 {len(shown)}" in page
    assert f"배선 {len(edges)}" in page
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


# ── 실행 상태와 Debug Output 규약 ───────────────────────────────────────


def _run(cg, tmp_path, **kw):
    from vlm_trainer.engine import samples as samples_mod
    from vlm_trainer.engine.runner import RunOptions, execute

    spec_dir = os.path.dirname(PROJECT)
    space = samples_mod.load(cg.sample_space, spec_dir)
    opts = RunOptions(run_id="v", cache_dir=str(tmp_path / "c"), spec_dir=spec_dir, **kw)
    return execute(cg, space, space.pick(1), opts)


def test_debug_output_off_generates_nothing(cg, tmp_path):
    """토글이 꺼져 있으면 미리보기를 만들지도 않는다 — 표시만 감추는 게 아니다."""
    rep = _run(cg, tmp_path, debug_output=False, trigger="ui")
    assert rep.previews == {}


def test_debug_output_on_renders_per_node_previews(cg, tmp_path):
    rep = _run(cg, tmp_path, debug_output=True, trigger="ui")
    assert rep.previews

    # 설계 문서 08 §8.4 — 노드 타입마다 "시각화 출력"이 다르다
    assert rep.previews["n_prompt"].kind == "prompt_render"
    assert "렌더된 최종 프롬프트" in rep.previews["n_prompt"].text
    assert rep.previews["n_answer"].kind == "answer_render"
    assert "<trend>" in rep.previews["n_answer"].text
    assert rep.previews["n_exp_ts"].kind == "regions_overlay"
    assert "지목" in rep.previews["n_exp_ts"].text
    assert rep.previews["n_stats"].kind == "table"

    # Output 노드는 부작용이 있어 미리보기하지 않는다
    for nid in cg.order:
        if cg.nodes[nid].kind is NodeKind.OUTPUT:
            assert nid not in rep.previews


def test_external_trigger_never_shows_debug_output(cg, tmp_path):
    """외부에서 트리거된 실행에서는 토글 값과 무관하게 만들지 않는다."""
    rep = _run(cg, tmp_path, debug_output=True, trigger="external")
    assert rep.previews == {}


def test_view_paints_node_states_and_debug_panel(cg, tmp_path):
    rep = _run(cg, tmp_path, debug_output=True, trigger="ui")
    page = render_mod.render(cg, report=rep)

    assert T.STATE["success"] in page
    assert "success · 1건" in page
    assert "DEBUG OUTPUT" in page.upper()
    assert "렌더된 최종 프롬프트" in page

    # 실행 전에는 전부 pending이고 Debug Output은 비어 있다고 말한다
    blank = render_mod.render(cg)
    assert "미리보기를 만들지 않았다" in blank
    assert "pending" in blank


# ── Procedure 접기 ──────────────────────────────────────────────────────


def test_a_procedure_is_drawn_as_one_box(cg):
    """25개 카드는 사람이 붙들 수 있는 수가 아니다. 캡슐화는 스펙에만 있으면 소용없다."""
    shown, edges = fold(cg)

    assert "p_crop" in shown and shown["p_crop"].inner == 4
    assert not any(k.startswith("p_crop/") for k in shown), "안쪽 노드가 밖에 남았다"
    assert len(shown) == len(cg.nodes) - 3


def test_the_box_carries_the_exposed_ports_and_their_types(cg):
    shown, _ = fold(cg)
    box = shown["p_crop"]

    assert set(box.inputs) == {"subject", "image"}
    assert set(box.outputs) == {"crops", "regions"}
    assert str(box.outputs["crops"]).startswith("ImageList"), str(box.outputs["crops"])


def test_wires_move_to_the_exposed_ports(cg):
    shown, edges = fold(cg)

    assert all(not a.startswith("p_crop/") and not b.startswith("p_crop/") for a, b in edges)
    assert any(a.startswith("p_crop:crops") for a, b in edges), "밖으로 나가는 배선이 사라졌다"
    assert len(edges) == len(cg.edges) - 3, "상자 안에서 끝나는 배선만 사라져야 한다"


def test_expanding_puts_the_inner_nodes_back(cg):
    shown, edges = fold(cg, expanded=["p_crop"])

    assert "p_crop" not in shown
    assert len([k for k in shown if k.startswith("p_crop/")]) == 4
    assert len(shown) == len(cg.nodes) and len(edges) == len(cg.edges)


def test_a_folded_box_is_not_green_while_something_inside_failed(cg):
    """상자가 초록인데 안이 빨간 것이 제일 나쁜 화면이다."""
    from vlm_trainer.engine.runner import FAILED, SUCCESS, RunReport

    rep = RunReport(order=list(cg.order))
    for nid in ("p_crop/n_exp", "p_crop/n_crop", "p_crop/n_resize"):
        rep.count(nid, SUCCESS)
    rep.count("p_crop/n_frame", FAILED)

    state, extra = render_mod.state_of_many(rep, list(cg.nodes))
    assert state == "failed"

    shown, _ = fold(cg)
    state, extra = render_mod.state_of_many(rep, shown["p_crop"].state_ids)
    assert state == "failed" and "1/4" in extra


def test_wires_touch_the_chips_they_connect(cg, page):
    """배선이 칩에서 떨어져 허공에서 시작하거나 끝나면 안 된다.

    카드의 세로 구성은 [입력 칩][카드][출력 칩]인데 **Input 노드는 입력 칩 줄이 없다**.
    그 한 줄을 좌표가 모르면 모든 배선이 CHIP_H만큼 어긋난다.
    """
    import re

    shown, edges = fold(cg)
    placed = render_mod._layout(shown, edges)

    def chip_x(nid, ports, name):
        for n, cx, cy, cw in render_mod._chips(ports, placed[nid].x, 0):
            if n == name:
                return cx + cw // 2
        raise AssertionError(f"{nid}:{name} 칩이 없다")

    starts = [m for m in re.findall(r"M (\d+) (\d+) C", page)]
    assert len(starts) == len(edges)

    for src, dst in edges:
        sn, sp = src.split(":")
        dn, dp = dst.split(":")
        s_node, d_node = shown[sn], shown[dn]

        s_top = 0 if s_node.kind is NodeKind.INPUT else render_mod.CHIP_H
        want_start = (
            chip_x(sn, s_node.outputs, sp),
            placed[sn].y + s_top + render_mod.CARD_H + render_mod.CHIP_H,
        )
        want_end = (chip_x(dn, d_node.inputs, dp), placed[dn].y)

        needle = f"M {want_start[0]} {want_start[1]} C"
        assert needle in page, f"{src} 배선이 출력 칩에서 시작하지 않는다"
        assert f"{want_end[0]} {want_end[1]}\"" in page, f"{dst} 배선이 입력 칩에서 끝나지 않는다"


def test_every_node_type_has_a_korean_name():
    """`image.crop_by_regions`가 무슨 일을 하는지는 그 이름만 봐서는 모른다."""
    from vlm_trainer.core.registry import all_defs

    # 테스트용 fixture 노드는 화면에 뜰 일이 없으므로 뺀다
    missing = [d.type for d in all_defs() if not d.doc.label and not d.type.startswith("test.")]
    assert not missing, missing


def test_the_card_shows_the_korean_name_and_keeps_the_id(cg, page):
    from vlm_trainer.core.registry import resolve as resolve_node

    shown, _ = fold(cg)
    for nid, s in shown.items():
        assert f'class="nm">{s.label}<' in page, f"{nid}의 한국어 이름이 카드에 없다"
        assert nid in page, f"{nid} 라는 id도 남아 있어야 한다"

    assert "이미지 읽기" in page
    assert resolve_node("source.image@1.0.0").doc.label == "이미지 읽기"
