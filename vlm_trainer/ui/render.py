"""읽기 전용 그래프 뷰어 — 컴파일된 그래프를 한 장의 HTML로 그린다.

편집기가 아니다. 캔버스 규약(수직 흐름, 상단 입력·하단 출력, Input 최상단·Output 최하단,
포트 타입 색, 배선 색 = 소스 포트 타입 색)이 실제 그래프에서 지켜지는지 눈으로 확인하는 수단이다.
서버도 프레임워크도 없다. 편집기는 이 위에 얹는다. 설계 문서 12.
"""

from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, Optional, Tuple

from ..core.compiler import CompiledGraph
from ..core.node import NodeKind
from ..core.registry import resolve as resolve_node
from . import tokens as T
# 배치와 상태 계산은 layout.py 에 있다. Qt 캔버스가 같은 것을 쓴다 —
# 좌표가 두 벌이 되면 앱과 `vlmt view` 가 같은 그래프를 다르게 그린다.
from .layout import (  # noqa: F401  (다시 내보낸다 — 기존 호출부를 그대로 둔다)
    CHIP_H,
    GRID,
    HEAD_H,
    PAD,
    ROW_H,
    Shown,
    _chips,
    _layout,
    _port_doc,
    _type_label,
    fold,
    state_of_many,
)

def _pio_rows(s: "Shown") -> str:
    """입력 / 처리 / 출력. 포트마다 타입 색 점을 찍는다.

    포트 이름(`schema`, `fields`)이 아니라 그 포트가 **무엇인지**를 적는다 —
    이름은 배선할 때 칩에 보이고, 여기서 읽을 것은 뜻이다.
    """
    out = []

    def row(kind: str, i: int, n: int, name: str, t: Any) -> str:
        label = kind if n == 1 else f"{kind} {i + 1}"
        base, root, is_list = _type_label(t)
        doc = _port_doc(s, kind, name) or name
        return (
            f'<div class="pio"><span class="pdot" '
            f'style="background:{T.port_color(root, is_list)}"></span>'
            f'<span class="pk">{label}</span>'
            f'<span class="pv" title="{html.escape(name)} · {html.escape(base)}">'
            f"{html.escape(doc)}</span></div>"
        )

    if s.inputs:
        for i, (name, t) in enumerate(s.inputs.items()):
            out.append(row("입력", i, len(s.inputs), name, t))
    else:
        out.append('<div class="pio none"><span class="pdot"></span>'
                   '<span class="pk">입력</span><span class="pv">없음</span></div>')

    out.append(
        f'<div class="pio proc"><span class="pdot"></span><span class="pk">처리</span>'
        f'<span class="pv" title="{html.escape(s.summary)}">'
        f"{html.escape(s.hint or s.label)}</span></div>"
    )

    if s.outputs:
        for i, (name, t) in enumerate(s.outputs.items()):
            out.append(row("출력", i, len(s.outputs), name, t))
    else:
        out.append('<div class="pio none"><span class="pdot"></span>'
                   '<span class="pk">출력</span><span class="pv">없음</span></div>')
    return "".join(out)


def render(
    cg: CompiledGraph,
    *,
    title: str = "",
    note: str = "",
    report: Any = None,
    editable: bool = False,
    compat: Optional[Dict[str, Dict[str, str]]] = None,
    banner: str = "",
    editor: Any = None,
    expanded: Any = (),
) -> str:
    # 물질화 경계는 그래프 밖의 한 줄이지만 어느 노드까지 미리 굽는지를 정한다.
    # 카드에 보이지 않으면 그 한 줄이 어디에 걸리는지 알 수 없다.
    graph = getattr(editor, "graph", None) if editor is not None else None
    boundary = set(graph.materialize.boundary) if graph is not None else set()
    # Procedure는 기본으로 **접어서** 그린다. 25개 카드는 사람이 붙들 수 있는 수가 아니다.
    # 게이트는 언제나 펼쳐진 그래프를 본다 — 접기는 보는 방식일 뿐이다.
    shown, dedges = fold(cg, expanded)
    # 사람이 옮겨 둔 자리가 있으면 그대로 쓴다. 없는 노드만 자동 배치된다.
    fixed = dict(getattr(editor, "layout", None) or {}) if editor is not None else {}
    placed = _layout(shown, dedges, fixed)
    max_lane = max((p.lane for p in placed.values()), default=0)
    width = max((p.x + p.w for p in placed.values()), default=0) + PAD
    height = max((p.y + p.h for p in placed.values()), default=0) + CHIP_H * 2 + PAD

    # ── 배선(SVG) ────────────────────────────────────────────────────
    out_pos: Dict[str, Tuple[int, int, str, bool]] = {}
    in_pos: Dict[str, Tuple[int, int]] = {}
    for nid, pl in placed.items():
        s = shown[nid]
        # 카드 안의 세로 구성: [입력 칩] [카드] [출력 칩].
        # **Input 노드는 입력 칩 줄이 아예 없다**(CSS의 `.inp .ports.top{display:none}`).
        # 그 한 줄을 좌표 계산이 모르면 배선이 칩에서 CHIP_H만큼 떨어진 허공에서 시작하고 끝난다.
        top_h = 0 if s.kind is NodeKind.INPUT else CHIP_H
        for name, cx, cy, cw in _chips(s.outputs, pl.x, pl.y + top_h + pl.h, pl.w):
            _, root, is_list = _type_label(s.outputs[name])
            out_pos[f"{nid}:{name}"] = (cx + cw // 2, cy + CHIP_H, root, is_list)
        for name, cx, cy, cw in _chips(s.inputs, pl.x, pl.y, pl.w):
            in_pos[f"{nid}:{name}"] = (cx + cw // 2, cy)

    paths = []
    for src_ref, dst_ref in dedges:
        a, b = out_pos.get(src_ref), in_pos.get(dst_ref)
        if not a or not b:
            continue
        x1, y1, root, is_list = a
        x2, y2 = b
        mid = (y1 + y2) / 2
        paths.append(
            f'<path d="M {x1} {y1} C {x1} {mid}, {x2} {mid}, {x2} {y2}" '
            f'stroke="{T.wire_color(root, is_list)}" fill="none" stroke-width="1.6"/>'
        )

    # ── 노드 카드 ────────────────────────────────────────────────────
    cards = []
    for nid, pl in placed.items():
        s = shown[nid]
        tag = {NodeKind.INPUT: "I", NodeKind.PROCESSING: "P", NodeKind.OUTPUT: "O"}[s.kind]
        shape = {NodeKind.INPUT: "inp", NodeKind.PROCESSING: "prc", NodeKind.OUTPUT: "out"}[s.kind]
        state, state_extra = state_of_many(report, s.state_ids)

        # 입력 칩은 선언이 아니라 **컴파일이 확정한 타입**을 보여준다.
        # 제네릭이 남아 있으면 그 자체가 눈에 띄어야 한다.
        in_types, out_types = s.inputs, s.outputs
        chips_in = "".join(
            f'<div class="chip cin" data-ref="{nid}:{name}" '
            f'style="left:{cx - pl.x}px;top:0;width:{cw}px;'
            f'background:{T.port_color(_type_label(in_types[name])[1], _type_label(in_types[name])[2])}">'
            f'<span class="t">&lt;{html.escape(_type_label(in_types[name])[0])}&gt;</span>'
            f'<span class="p">{html.escape(name)}</span></div>'
            for name, cx, cy, cw in _chips(in_types, pl.x, 0, pl.w)
        )
        chips_out = "".join(
            f'<div class="chip cout" data-ref="{nid}:{name}" '
            f'style="left:{cx - pl.x}px;top:0;width:{cw}px;'
            f'background:{T.port_color(_type_label(out_types[name])[1], _type_label(out_types[name])[2])}">'
            f'<span class="t">&lt;{html.escape(_type_label(out_types[name])[0])}&gt;</span>'
            f'<span class="p">{html.escape(name)}</span></div>'
            for name, cx, cy, cw in _chips(out_types, pl.x, 0, pl.w)
        )

        cards.append(
            f'<div class="node {shape}" data-node="{html.escape(nid)}" '
            f'style="left:{pl.x}px;top:{pl.y}px;width:{pl.w}px" '
            f'title="{html.escape(nid)} · {html.escape(s.ref)}">'
            f'<div class="ports top">{chips_in}</div>'
            f'<div class="card" style="height:{pl.h}px;border-top:3px solid {T.category_color(s.category)};'
            f'border-left:4px solid {T.STATE.get(state, "#4A4A4A")}">'
            f'<div class="hd"><span class="nm" title="{html.escape(nid)}">{html.escape(s.label)}</span>'
            + f'<span class="st"><span class="sdot" style="background:{T.STATE.get(state, "#4A4A4A")}"></span>{state}{" · " + html.escape(state_extra) if state_extra else ""}</span>'
            + f'<span class="badge b{tag}" title="{tag}">{tag}</span>'
            + (f'<span class="fold" title="{s.inner}개 노드가 들어 있다">&#9656;{s.inner}</span>'
               if s.inner else "")
            + ('<span class="mat" title="물질화 경계 — 여기까지 미리 굽는다">&#9640;</span>'
               if nid in boundary else "")
            + "</div>"
            + _pio_rows(s)
            + "</div>"
            + f'<div class="ports bot">{chips_out}</div>'
            + "</div>"
        )

    # ── 좌측 레일: 카테고리별 노드 수 ────────────────────────────────
    cats: Dict[str, int] = {}
    for nid in cg.order:
        cats[cg.nodes[nid].category] = cats.get(cg.nodes[nid].category, 0) + 1
    # 뷰어의 왼쪽 레일은 **이 그래프가 쓴 것**만 센다. 라이브러리 전체 목록은
    # 고를 수 있을 때만 의미가 있고, 이 페이지에서는 아무것도 고를 수 없다.
    lib = "".join(
        f'<div class="libcat"><span class="dot" style="background:{T.category_color(c)}"></span>'
        f'<span class="lbl">{html.escape(c)}</span><span class="cnt">{n}</span></div>'
        for c, n in sorted(cats.items())
    )

    # ── 우측 패널: 노드 상세(Node Quick Info) ───────────────────────
    details = []
    for nid in cg.order:
        n = cg.nodes[nid]
        d = resolve_node(n.ref)
        ins = "".join(
            f'<div class="row"><span class="k">{html.escape(p)}</span>'
            f'<span class="v">{html.escape(str(n.input_types.get(p) or t.type))}</span></div>'
            for p, t in d.inputs.items()
        )
        outs = "".join(
            f'<div class="row"><span class="k">{html.escape(p)}</span>'
            f'<span class="v">{html.escape(str(t))}</span></div>'
            for p, t in n.output_types.items()
        )
        details.append(
            f'<details><summary>{html.escape(nid)} <em>{html.escape(n.ref)}</em></summary>'
            f'<div class="doc">{html.escape(d.doc.summary)}</div>'
            + (f'<div class="hdr">입력</div>{ins}' if ins else "")
            + (f'<div class="hdr">출력</div>{outs}' if outs else "")
            + f'<div class="hdr">캐시 키</div><div class="row"><span class="v">{n.cache_key}</span></div>'
            + "</details>"
        )

    # ── Debug Output — 토글이 켜져 있을 때만 값이 있다 ──────────────
    previews = getattr(report, "previews", {}) or {}
    debug = "".join(
        f'<details open><summary>{html.escape(nid)} '
        f'<em>{html.escape(getattr(previews[nid], "kind", ""))}</em></summary>'
        f'<pre class="pv">{html.escape(getattr(previews[nid], "text", ""))}</pre></details>'
        for nid in cg.order
        if nid in previews
    )
    debug_block = (
        '<h4>Debug Output</h4><div id="debugout">'
        + (debug or '<div class="doc">위쪽 <b>Debug Output</b> 을 켜고 <b>Run</b> 을 누르면 각 단계의 중간 결과가 여기에 보입니다.</div>')
        + "</div>"
    )

    quarantine = ""
    if report is not None and getattr(report, "quarantine", None):
        rows = "".join(
            f'<div class="row"><span class="k">{html.escape(q.sample_key)}</span>'
            f'<span class="v">@{html.escape(q.node_id)} {html.escape(q.cause[:90])}</span></div>'
            for q in report.quarantine[:12]
        )
        quarantine = f'<h4>격리 {len(report.quarantine)}건</h4>{rows}'

    counts = {
        "I": sum(1 for i in cg.order if cg.nodes[i].kind is NodeKind.INPUT),
        "P": sum(1 for i in cg.order if cg.nodes[i].kind is NodeKind.PROCESSING),
        "O": sum(1 for i in cg.order if cg.nodes[i].kind is NodeKind.OUTPUT),
    }
    head = title or cg.id or "graph"

    # **스크립트를 싣지 않는다.** 이 페이지는 파일 하나로 넘겨 보는 산출물이고,
    # 누를 수 있는 것이 없어야 눌러도 안 되는 것이 없다. 편집은 `vlmt edit`(앱)이 한다.
    scripts = ""
    tools = _VIEWER_TOOLS
    spec_hash = cg.spec_hash
    # 게이트 위반 전문을 펼쳐 두면 캔버스를 밀어내 화면 비율이 무너진다.
    # 한 줄로 알리고, 읽고 싶을 때 팝업으로 연다.
    banner_html = ""
    if banner:
        lines = [ln for ln in banner.splitlines() if ln.strip()]
        first = lines[0] if lines else banner
        if first.endswith('게이트 위반:') and len(lines) > 1:
            first = lines[1]
        # 전문은 숨긴 <pre>에 둔다. <script>에 넣으면 엔티티가 날것으로 읽힌다
        # (스크립트 안쪽은 파싱되지 않아 &#x27; 이 그대로 보인다).
        banner_html = (
            f'<div class="banner" onclick="vlmtGate()" title="눌러서 전문 보기">'
            f'<span class="bmark">!</span>'
            f'<span class="btext">{html.escape(first[:110])}</span>'
            f'<span class="bmore">자세히 보기</span></div>'
            f'<pre id="gatetext" hidden>{html.escape(banner)}</pre>'
        )

    # 오른쪽은 설계 문서 12.1의 파티션: Debug Output(위) + 탭으로 나뉜 Configuration Panel(아래).
    # **뷰어는 스크립트를 싣지 않으므로 탭을 쓰지 않는다** — 눌러도 안 바뀌는 탭은 없느니만 못하다.
    details_block = "".join(details)
    # 토글이 꺼져 있으면 이 칸은 안내 한 줄뿐인데 48vh를 잡고 있었다.
    # 비어 있을 때는 줄여 두고, 내용이 생기면 그때 자리를 준다.
    has_preview = bool(getattr(report, "previews", None)) or bool(quarantine)
    dbg_pane = (
        f'<div class="dbgpane{" full" if has_preview else " slim"}">'
        f'{debug_block}{quarantine}</div>'
    )
    side = dbg_pane + f'<div class="tabbody"><h4>Node Quick Info</h4>{details_block}</div>'

    return _TEMPLATE.format(
        title=html.escape(head),
        side=side,
        scripts=scripts,
        tools=tools,
        banner=banner_html,
        css=_CSS,
        lib=lib,
        cards="".join(cards),
        paths="".join(paths),
        w=width,
        h=height,
        spec_hash=spec_hash,
        # 보이는 것과 게이트가 보는 것을 둘 다 적는다 — 접힌 상자가 몇 개를 품었는지
        # 헤더에서 사라지면 "노드 12"와 카드 9장이 어긋나 보인다.
        solution=html.escape(os.path.basename(os.path.dirname(
            os.path.dirname(os.path.dirname(getattr(editor, "path", "") or "")))) or "Solution"),
        boxes=len(shown),
        nodes=len(cg.nodes),
        edges=len(dedges),
        lanes=max_lane + 1,
        ci=counts["I"],
        cp=counts["P"],
        co=counts["O"],
        note=html.escape(note or "compile OK"),
        runline=html.escape(
            f"처리 {report.processed}건 · 캐시 {report.cache}" if report is not None else "실행 전"
        ),
        profile=_profile_control(cg, editor),
    )


def write(cg: CompiledGraph, path: str, *, title: str = "", note: str = "", report: Any = None) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(cg, title=title, note=note, report=report))
    return path


_CSS = f"""
*{{box-sizing:border-box}}
body{{margin:0;background:{T.SURFACE['chrome']};color:#D8DCDF;
     font:13px/1.45 "Segoe UI","Malgun Gothic",sans-serif}}
.menubar{{background:{T.SURFACE['chrome']};padding:6px 12px;border-bottom:1px solid {T.SURFACE['line']};
        font-size:12px;color:#8A9196}}
.toolbar{{background:{T.SURFACE['toolbar']};padding:7px 12px;border-bottom:1px solid {T.SURFACE['line']};
        display:flex;gap:14px;align-items:center;font-size:12px;
        white-space:nowrap;overflow-x:auto;overflow-y:hidden}}
.toolbar>*{{flex:0 0 auto}}
.toolbar b{{color:{T.NODE['accent']};font-weight:600}}
.side{{display:grid;grid-template-rows:auto 1fr;padding:0;overflow:hidden}}
.dbgpane{{overflow:auto;padding:10px 12px;border-bottom:1px solid {T.SURFACE['line']}}}
.dbgpane.full{{max-height:48vh}}
.dbgpane.slim{{max-height:74px;padding:8px 12px}}
.dbgpane.slim h4{{margin-bottom:2px}}
.cfg{{display:grid;grid-template-rows:auto 1fr;overflow:hidden}}
.tabs{{display:flex;flex-wrap:wrap;gap:1px;background:{T.SURFACE['line']};
      border-bottom:1px solid {T.SURFACE['line']}}}
.tab{{background:{T.SURFACE['panel_alt']};color:#8A9196;font-size:11px;padding:5px 9px;
      cursor:pointer;flex:1 1 auto;text-align:center;white-space:nowrap}}
.tab:hover{{color:#C6CCD1}}
.tab.on{{background:{T.SURFACE['panel']};color:{T.NODE['accent']};
      box-shadow:inset 0 -2px 0 {T.NODE['accent']}}}
.tabbody{{overflow:auto;padding:10px 12px}}
.projlist{{padding:0 12px 8px;border-bottom:1px solid {T.SURFACE['line']};margin-bottom:8px}}
.ptree{{font-size:11.5px}}
.ptree .sol{{color:#8A9196;padding:2px 0}}
.ptree .proj{{color:#A8B0B6;padding:2px 0 2px 14px;border-left:2px solid transparent}}
.ptree .proj.on{{color:#D8DCDF;border-left-color:{T.NODE['border']}}}
.libsearch{{width:calc(100% - 24px);margin:0 12px 6px;background:{T.SURFACE['canvas']};
      color:#D8DCDF;border:1px solid {T.SURFACE['line']};font-size:11px;padding:3px 6px}}
.libnode.hide,.rail details.hide{{display:none}}
.shell{{display:grid;grid-template-columns:220px 1fr 340px;height:calc(100vh - 62px)}}
.rail,.side{{background:{T.SURFACE['panel']};overflow:auto;padding:10px 0}}
.rail{{border-right:1px solid {T.SURFACE['line']}}}
.side{{border-left:1px solid {T.SURFACE['line']};padding:10px 12px}}
.rail h4,.side h4{{margin:6px 12px 8px;font-size:11px;letter-spacing:.12em;text-transform:uppercase;
                 color:#79828A;font-weight:600}}
.lib{{display:flex;align-items:center;gap:8px;padding:4px 12px;font-size:12.5px;color:#A8B0B6}}
.lib .dot{{width:10px;height:10px;border-radius:2px}}
.lib .n{{margin-left:auto;color:#6F7478;font-size:11px}}
.canvas{{position:relative;background:{T.SURFACE['canvas']};overflow:auto}}
/* Mech-Vision 처럼 격자를 깐다. 자리를 눈으로 맞출 기준선이 없으면
   자유 배치는 어질러지기만 한다. 굵은 선은 5칸마다. */
.stage{{position:relative;
      background-image:
        linear-gradient({T.SURFACE['line']} 1px, transparent 1px),
        linear-gradient(90deg, {T.SURFACE['line']} 1px, transparent 1px),
        linear-gradient(rgba(42,42,42,.55) 1px, transparent 1px),
        linear-gradient(90deg, rgba(42,42,42,.55) 1px, transparent 1px);
      background-size:{GRID * 5}px {GRID * 5}px,{GRID * 5}px {GRID * 5}px,{GRID}px {GRID}px,{GRID}px {GRID}px}}
.node.dragging{{opacity:.85;z-index:30;cursor:grabbing}}
.node .hd .nm{{cursor:grab}}
svg.wires{{position:absolute;inset:0;pointer-events:none}}
.node{{position:absolute}}
.ports{{position:relative;height:{CHIP_H}px}}
.chip{{position:absolute;height:{CHIP_H}px;border-radius:3px;padding:2px 6px;overflow:hidden;
      color:#fff;font-size:10px;line-height:1.15}}
.chip .t{{display:block;opacity:.92;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.chip .p{{display:block;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.card{{background:{T.NODE['bg']};border:1px solid {T.NODE['border']};border-radius:3px;
      padding:6px 9px}}
.inp .card{{border-top-left-radius:12px;border-top-right-radius:12px}}
.out .card{{border-bottom-left-radius:12px;border-bottom-right-radius:12px}}
.inp .ports.top,.out .ports.bot{{display:none}}
.hd{{display:flex;align-items:center;gap:6px;height:{HEAD_H}px}}
.hd .st{{margin-left:auto;display:flex;align-items:center;gap:4px;color:{T.NODE['muted']};
      font-size:10px;white-space:nowrap}}
.pio{{display:flex;align-items:center;gap:6px;height:{ROW_H}px;font-size:11px;
      color:#A8B0B6;overflow:hidden}}
.pio .pdot{{flex:0 0 auto;width:7px;height:7px;border-radius:2px;background:transparent}}
.pio .pk{{flex:0 0 auto;color:#6F7478;font-size:10px;min-width:34px}}
.pio .pv{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.pio.none .pv{{color:#5F6468}}
.pio.proc .pv{{color:#C6CCD1}}
.nm{{color:{T.NODE['title']};font-weight:600;font-size:13px;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap}}
/* margin-left:auto 는 카드 머리줄에서 배지를 오른쪽으로 밀기 위한 것이었는데,
   라이브러리 항목까지 잡아 이름 길이만큼 들쭉날쭉하게 밀리고 있었다. 카드 안으로 한정한다. */
.badge{{font-size:10px;border:1px solid;border-radius:2px;padding:0 4px;flex:0 0 auto}}
.hd .badge{{margin-left:auto}}
.bI{{color:{T.PORT['Image']}}} .bP{{color:#8A9196}} .bO{{color:{T.PORT['Table']}}}
.ref,.sum{{color:{T.NODE['muted']};font-size:11px;white-space:nowrap;overflow:hidden;
          text-overflow:ellipsis}}
.sdot{{width:7px;height:7px;border-radius:50%;display:inline-block}}
pre.pv{{background:{T.SURFACE['canvas']};border:1px solid {T.SURFACE['line']};border-radius:3px;
       padding:6px 8px;margin:4px 0;font-size:11px;line-height:1.4;color:#A8B0B6;
       white-space:pre-wrap;max-height:220px;overflow:auto}}
.sum{{margin-top:3px}}
details{{border-bottom:1px solid {T.SURFACE['line']};padding:6px 0}}
summary{{cursor:pointer;font-size:12.5px;color:#C6CCD1}}
summary em{{color:#6F7478;font-style:normal;font-size:11px}}
.doc{{color:#8A9196;font-size:11.5px;margin:4px 0}}
.hdr{{color:#79828A;font-size:10px;letter-spacing:.1em;text-transform:uppercase;margin:6px 0 2px}}
.row{{display:flex;gap:8px;font-size:11px}}
.row .k{{color:#A8B0B6;min-width:64px}}
.row .v{{color:#6F7478;word-break:break-all}}
.log{{background:{T.SURFACE['panel_alt']};border-top:1px solid {T.SURFACE['line']};
     padding:6px 12px;font-size:11.5px;color:#8A9196}}
.node.sel .card{{background:{T.NODE['bg_selected']};border-color:{T.NODE['border_selected']}}}
/* 지금 도는 노드. 선택(sel)과는 다른 신호라 색이 아니라 '움직임'으로 구분한다 —
   정지 화면에서 초록 테두리 하나는 완료와 헷갈리지만, 맥동하는 것은 하나뿐이다. */
.node.active{{z-index:20}}
.node.active .card{{border-color:{T.STATE['running']};
    box-shadow:0 0 0 1px {T.STATE['running']},0 0 14px -2px {T.STATE['running']};
    animation:vlmtPulse 1.25s ease-in-out infinite}}
@keyframes vlmtPulse{{
  0%,100%{{box-shadow:0 0 0 1px {T.STATE['running']},0 0 6px -2px {T.STATE['running']}}}
  50%    {{box-shadow:0 0 0 2px {T.STATE['running']},0 0 20px -1px {T.STATE['running']}}}
}}
/* 움직임을 원치 않는다고 밝힌 사용자에게는 테두리만 남긴다 */
@media (prefers-reduced-motion:reduce){{.node.active .card{{animation:none}}}}
.params{{display:none;border-bottom:1px solid {T.SURFACE['line']};padding-bottom:8px;margin-bottom:8px}}
.params.on{{display:block}}
.phd{{color:#C6CCD1;font-size:12.5px;margin:2px 0 8px}}
.phd em{{color:#6F7478;font-style:normal;font-size:11px}}
.ndet{{border-top:1px solid {T.SURFACE['line']};margin-top:6px;padding-top:4px}}
.ndet>summary{{cursor:pointer;font-size:11px;color:#8A9196;list-style:none;padding:2px 0}}
.ndet>summary::-webkit-details-marker{{display:none}}
.ndet>summary::before{{content:'\25b8 ';color:{T.NODE['accent']}}}
.ndet[open]>summary::before{{content:'\25be '}}
.ndet>summary:hover{{color:#C6CCD1}}
.wrow{{display:grid;grid-template-columns:auto 12px 1fr 18px;gap:5px;align-items:center;
      font-size:11px;color:#A8B0B6;padding:2px 0}}
.wrow .wp{{color:#D8DCDF}}
.wrow .wa{{color:{T.NODE['accent']}}}
.wrow .ws{{color:#8A9196;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.wrow .wcut{{cursor:pointer;color:#6F7478;text-align:center;line-height:18px;min-width:18px}}
.wrow .wcut:hover{{color:{T.STATE['failed']};background:{T.NODE['bg']};border-radius:2px}}
.trow{{font-size:10.5px;padding:2px 0}}
.trow .tp{{color:#D8DCDF;margin-right:5px}}
.trow code{{color:{T.NODE['accent']};font-family:ui-monospace,Consolas,monospace}}
.nact{{display:flex;gap:6px;padding:4px 0}}
.doc.dim{{color:#6F7478}}
.prow{{display:grid;grid-template-columns:1fr 148px;gap:8px;align-items:center;margin-bottom:5px}}
.prow label{{color:#A8B0B6;font-size:11.5px;display:flex;gap:4px;align-items:center}}
.prow input[type=text],.prow input[type=number]{{background:{T.SURFACE['canvas']};color:#D8DCDF;
    border:1px solid {T.SURFACE['line']};border-radius:2px;padding:3px 6px;font-size:11.5px;width:100%}}
.prow input:focus{{outline:1px solid {T.NODE['border']}}}
.mk{{font-size:9px;border:1px solid;border-radius:2px;padding:0 3px}}
.mk.r{{color:{T.PORT['Image']}}} .mk.t{{color:{T.STATE['partial']}}}
.libnode .lbl{{flex:0 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.libnode .typ{{margin-left:auto;color:#5F6468;font-size:9.5px;overflow:hidden;
      text-overflow:ellipsis;white-space:nowrap;max-width:45%}}
.libghost{{position:fixed;z-index:50;pointer-events:none;padding:3px 8px;font-size:11px;
        background:{T.NODE['bg']};border:1px solid {T.NODE['border']};color:#D8DCDF;border-radius:3px}}
.canvas.candrop{{outline:1px dashed {T.NODE['border']};outline-offset:-3px}}
.libnode{{padding:3px 12px 3px 22px;font-size:11.5px;color:#A8B0B6;cursor:pointer;
        display:flex;gap:6px;align-items:center}}
.libnode:hover{{background:{T.NODE['bg']};color:#D8DCDF}}
.rail details summary{{list-style:none;cursor:pointer;padding:4px 12px;font-size:12px;
        display:flex;gap:8px;align-items:center;color:#C6CCD1}}
.rail details summary::-webkit-details-marker{{display:none}}
.rail .dot{{width:10px;height:10px;border-radius:2px}}
.rail .n{{margin-left:auto;color:#6F7478;font-size:10.5px}}
.del{{color:#8A9196;cursor:pointer;font-size:14px;line-height:1;padding:0 2px}}
.del:hover{{color:{T.STATE['failed']}}}
.ssok{{font-size:11.5px;color:#A8B0B6;margin-bottom:2px}}
.ssbad{{font-size:11.5px;color:{T.STATE['failed']};margin-bottom:4px}}
.sscols{{font-size:10.5px;color:#6F7478;margin-bottom:6px;word-break:break-all}}
.recipe{{margin-bottom:14px}}
.recipe h4{{margin-bottom:4px}}
.rdirty{{margin-left:8px;font-size:10px;color:{T.STATE['running']};font-weight:400}}
.rstat{{font-size:11.5px;color:#A8B0B6;margin-bottom:6px}}
.rstat b{{color:#D8DCDF}}
.rwhy{{margin-left:8px;color:#C9A87A;font-size:10.5px}}
.obar{{background:{T.NODE['bg']};border-left:2px solid {T.NODE['border_selected']};
      padding:6px 8px;font-size:11.5px;color:#C6CCD1;margin-bottom:4px}}
.obar b{{color:#D8DCDF}}
.orow{{display:grid;grid-template-columns:1fr auto auto 22px;gap:6px;align-items:center;
      font-size:11.5px;color:#A8B0B6;padding:3px 8px}}
.orow.store{{display:flex;flex-wrap:wrap;gap:4px 0;padding-top:8px}}
.orow.store input{{flex:1 1 100%;margin-bottom:4px}}
.orow input{{background:{T.SURFACE['canvas']};border:1px solid {T.NODE['border']};
      color:#D8DCDF;font-size:11px;padding:2px 6px}}
.oval{{font-family:ui-monospace,Consolas,monospace;color:#57F7E6;font-size:11px}}
.rbase{{font-family:ui-monospace,Consolas,monospace;color:#6F7478;font-size:10px}}
.odrop{{cursor:pointer;color:#8A9196;text-align:center;line-height:20px;border-radius:2px;
      min-width:20px;justify-self:end}}
.odrop:hover{{color:{T.STATE['failed']};background:{T.NODE['bg']}}}
.rrow{{display:grid;grid-template-columns:1fr auto 22px;gap:6px;align-items:center;
      padding:4px 8px;font-size:11.5px;color:#A8B0B6;border-left:2px solid transparent}}
.rrow.on{{border-left-color:{T.NODE['border_selected']};background:{T.NODE['bg']};color:#D8DCDF}}
.rrow .rname{{cursor:pointer}}
.rrow .rname:hover{{color:#D8DCDF;text-decoration:underline}}
.rrow .rn{{color:#6F7478;font-size:10.5px}}
.rrow .ract{{color:{T.STATE['success']};margin-right:3px}}
.rrow .rdel{{cursor:pointer;color:#6F7478;text-align:center;line-height:20px;border-radius:2px;
      min-width:20px;justify-self:end}}
.rrow .rdel:hover{{color:{T.STATE['failed']};background:{T.NODE['bg']}}}
.rnote{{grid-column:1/4;color:#6F7478;font-size:10.5px}}
.btn.sm{{padding:1px 8px;font-size:10.5px;margin-left:6px}}
.mk.o{{background:#2B4A57;color:#8FE3F5}}
.padd{{cursor:pointer;color:#6F7478;margin-left:4px;font-size:12px;
      min-width:16px;line-height:16px;text-align:center;border-radius:2px}}
.padd:hover{{color:#57F7E6;background:{T.NODE['bg']}}}
.btn.go{{border-color:{T.STATE['running']};color:{T.STATE['running']}}}
/* 눌려 있는 토글. 켜짐/꺼짐이 한눈에 보여야 화면이 왜 움직였는지 설명이 된다 */
.btn.on{{border-color:{T.NODE['accent']};color:{T.NODE['accent']};
    background:{T.NODE['bg_selected']}}}
.btn:disabled{{opacity:.45;cursor:default}}
.sep{{width:1px;height:16px;background:{T.SURFACE['line']};margin:0 4px}}
.tgl{{display:flex;gap:4px;align-items:center;font-size:11px;color:#A8B0B6;margin-left:6px}}
.tgl input[type=number]{{width:44px;background:{T.SURFACE['canvas']};color:#D8DCDF;
      border:1px solid {T.SURFACE['line']};font-size:11px;padding:1px 4px}}
.prof{{background:{T.SURFACE['canvas']};color:#D8DCDF;border:1px solid {T.SURFACE['line']};
      font-size:11px;padding:1px 4px}}
.fold{{color:#8FE3F5;cursor:pointer;font-size:10px;margin-left:4px;padding:0 3px;
      border:1px solid {T.NODE['border']};border-radius:2px;line-height:14px}}
.fold:hover{{background:{T.NODE['bg']};color:#57F7E6}}
.node .card .hd .nm{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.mat{{color:#57F7E6;font-size:11px;margin-left:4px}}
.matt{{margin-left:auto;font-size:10.5px;color:#8FE3F5;display:flex;gap:3px;align-items:center;
      font-weight:400}}
.phd{{display:flex;gap:6px;align-items:center}}
.pvimg{{max-width:100%;display:block;margin:4px 0;border:1px solid {T.SURFACE['line']};
      background:{T.SURFACE['canvas']};image-rendering:auto}}
.runline{{margin-left:10px;font-size:11px;color:{T.STATE['running']};
      font-family:ui-monospace,Consolas,monospace}}
.runline.bad{{color:{T.STATE['failed']}}}
.hrow{{border-left:2px solid transparent;padding:4px 8px;margin-bottom:2px;cursor:pointer;
      font-size:11.5px;color:#A8B0B6;display:grid;grid-template-columns:58px 1fr 64px;gap:6px}}
.hrow:hover{{background:{T.NODE['bg']}}}
.hrow.cur{{border-left-color:{T.NODE['border']};background:{T.NODE['bg']};color:#D8DCDF}}
.hrow .ht{{color:#6F7478;font-size:10.5px}}
.hrow.past .hl{{color:#7E868C}}
.hrow.past .ht::after{{content:"·";margin-left:3px;color:{T.STATE['cached']}}}
.hrow .hh{{color:#6F7478;font-size:10px;text-align:right}}
.hrow .hdp,.hrow .hdm{{grid-column:1/4;font-family:ui-monospace,Consolas,monospace;font-size:10px;
      white-space:pre-wrap}}
.hrow .hdp{{color:#7FBF9F}} .hrow .hdm{{color:#C98A8A}}
.log b{{color:#3FB27F}}
.btn{{background:{T.SURFACE['panel']};color:#D8DCDF;border:1px solid {T.NODE['border']};
     border-radius:3px;padding:3px 12px;font-size:12px;cursor:pointer}}
.btn:hover{{background:{T.NODE['bg_selected']}}}
.hint{{margin-left:auto;color:#79828A;font-size:11.5px}}
/* 작업이 도는 동안에는 배선 안내를 접는다 — 그 순간 사람이 보는 것은 진행 상황이다 */
body.busy .hint{{display:none}}
.runline.live{{color:{T.STATE['running']};font-weight:600}}
.banner{{background:#3A2A2A;color:#E8B0B0;padding:5px 12px;font-size:12px;cursor:pointer;
      border-bottom:1px solid {T.SURFACE['line']};display:flex;gap:8px;align-items:center;
      white-space:nowrap;overflow:hidden}}
.banner:hover{{background:#452F2F}}
.bmark{{flex:0 0 auto;width:16px;height:16px;border-radius:50%;background:{T.STATE['failed']};
      color:#1A1A1A;font-weight:700;font-size:11px;text-align:center;line-height:16px}}
.btext{{overflow:hidden;text-overflow:ellipsis}}
.bmore{{margin-left:auto;flex:0 0 auto;color:#E8B0B0;text-decoration:underline;font-size:11px}}
.modal{{position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;
      justify-content:center;z-index:80}}
.modal .box{{background:{T.SURFACE['panel']};border:1px solid {T.STATE['failed']};border-radius:4px;
      max-width:760px;max-height:70vh;display:flex;flex-direction:column}}
.modal .mh{{display:flex;align-items:center;gap:8px;padding:10px 14px;
      border-bottom:1px solid {T.SURFACE['line']};color:#E8B0B0;font-size:13px;font-weight:600}}
.modal .mb{{overflow:auto;padding:12px 14px;white-space:pre-wrap;font-size:12px;color:#D8DCDF;
      line-height:1.55;font-family:ui-monospace,Consolas,monospace}}
.modal .mf{{padding:8px 14px;border-top:1px solid {T.SURFACE['line']};text-align:right}}
.modal .x{{margin-left:auto;cursor:pointer;color:#8A9196;font-size:16px}}
.modal .x:hover{{color:#E8B0B0}}
.chip.cout{{cursor:grab}}
.chip.okdrop{{outline:2px solid {T.NODE['border_selected']};outline-offset:1px}}
.chip.nodrop{{opacity:.28;cursor:not-allowed}}
#toast{{position:fixed;right:16px;bottom:40px;z-index:9;max-width:560px;padding:0;opacity:0;
       transition:opacity .15s;font-size:12.5px;border-radius:3px}}
#toast.show{{opacity:1;padding:8px 12px}}
#toast.ok{{background:#1E3A2E;color:#9BE0BC;border:1px solid #3FB27F}}
#toast.bad{{background:#3A2224;color:#F0B0B0;border:1px solid #D9615A}}
"""

_TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head>
<body>
<div class="menubar">File &nbsp; Edit &nbsp; View &nbsp; Solution &nbsp; Node &nbsp; Plugins &nbsp; Settings &nbsp; Help</div>
<div class="toolbar">
  <b>{title}</b>
  <span>상자 {boxes} · 배선 {edges} · 레인 {lanes}단 &nbsp;<em>노드 {nodes}</em></span>
  <span>I {ci} / P {cp} / O {co}</span>
  <span>프로파일 {profile}</span>
  {tools}
</div>
{banner}
<div id="toast"></div>
<div class="shell">
  <div class="rail">
    <div class="projlist">
      <h4>Projects</h4>
      <div class="ptree">
        <div class="sol">{solution}</div>
        <div class="proj on">{title}</div>
      </div>
    </div>
    <h4>이 그래프의 노드</h4>
    {lib}
  </div>
  <div class="canvas"><div class="stage" style="width:{w}px;height:{h}px">
    <svg class="wires" width="{w}" height="{h}">{paths}</svg>
    {cards}
  </div></div>
  <div class="side">{side}</div>
</div>
<div class="log"><b>{note}</b> &nbsp; {runline} &nbsp; spec_hash {spec_hash}</div>
{scripts}
</body></html>
"""


_VIEWER_TOOLS = '<span style="margin-left:auto">읽기 전용 뷰어</span>'

def _profile_control(cg: CompiledGraph, editor: Any) -> str:
    """실행 프로파일. 읽기 전용이다 — 고치는 것은 편집기(앱)의 몫이다."""
    return html.escape(cg.runtime_profile)
