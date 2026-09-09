"""읽기 전용 그래프 뷰어 — 컴파일된 그래프를 한 장의 HTML로 그린다.

편집기가 아니다. 캔버스 규약(수직 흐름, 상단 입력·하단 출력, Input 최상단·Output 최하단,
포트 타입 색, 배선 색 = 소스 포트 타입 색)이 실제 그래프에서 지켜지는지 눈으로 확인하는 수단이다.
서버도 프레임워크도 없다. 편집기는 이 위에 얹는다. 설계 문서 12.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.compiler import CompiledGraph
from ..core.node import NodeKind
from ..core.registry import resolve as resolve_node
from . import tokens as T

CARD_W = 264
CARD_H = 84
CHIP_H = 30
LANE_GAP = 52
COL_GAP = 34
PAD = 48


@dataclass
class Placed:
    id: str
    x: int
    y: int
    lane: int


def _layout(cg: CompiledGraph) -> Dict[str, Placed]:
    """위상 순서대로 레인을 쌓고, 레인 안에서는 상류의 무게중심으로 좌우를 정한다."""
    lanes: Dict[int, List[str]] = {}
    for nid in cg.order:
        lanes.setdefault(cg.nodes[nid].lane, []).append(nid)

    placed: Dict[str, Placed] = {}
    x_of: Dict[str, float] = {}
    width = max((len(v) for v in lanes.values()), default=1)

    for lane in sorted(lanes):
        ids = lanes[lane]
        def key(nid: str) -> Tuple[float, str]:
            ups = [x_of[e.src_node] for e in cg.edges if e.dst_node == nid and e.src_node in x_of]
            return (sum(ups) / len(ups) if ups else 1e9, nid)

        ids = sorted(ids, key=key)
        span = len(ids)
        offset = (width - span) / 2.0
        for i, nid in enumerate(ids):
            col = offset + i
            x_of[nid] = col
            placed[nid] = Placed(
                id=nid,
                x=int(PAD + col * (CARD_W + COL_GAP)),
                y=int(PAD + lane * (CARD_H + CHIP_H * 2 + LANE_GAP)),
                lane=lane,
            )
    return placed


def _chips(ports: Dict[str, Any], x: int, y: int) -> List[Tuple[str, int, int, int]]:
    """포트 칩의 (이름, x, y, 너비). 카드 폭을 균등 분할한다."""
    names = list(ports)
    if not names:
        return []
    w = CARD_W // len(names)
    return [(n, x + i * w, y, w - 4) for i, n in enumerate(names)]


def _type_label(t: Any) -> Tuple[str, str, bool]:
    """(표시 문자열, base 이름, 리스트 여부)."""
    base = t.display_base
    is_list = t.list_of is not None
    root = base[:-4] if is_list and base.endswith("List") else base
    return base, root, is_list


def _summary(cg: CompiledGraph, nid: str, limit: int = 3) -> str:
    p = cg.nodes[nid].params
    items = [f"{k}={v}" for k, v in list(p.items())[:limit] if v not in (None, "", [], {})]
    return ", ".join(items)


def render(cg: CompiledGraph, *, title: str = "", note: str = "") -> str:
    placed = _layout(cg)
    max_lane = max((p.lane for p in placed.values()), default=0)
    width = max((p.x for p in placed.values()), default=0) + CARD_W + PAD
    height = max((p.y for p in placed.values()), default=0) + CARD_H + CHIP_H * 2 + PAD

    # ── 배선(SVG) ────────────────────────────────────────────────────
    out_pos: Dict[str, Tuple[int, int, str, bool]] = {}
    in_pos: Dict[str, Tuple[int, int]] = {}
    for nid, pl in placed.items():
        n = cg.nodes[nid]
        for name, cx, cy, cw in _chips(n.output_types, pl.x, pl.y + CARD_H + CHIP_H):
            _, root, is_list = _type_label(n.output_types[name])
            out_pos[f"{nid}:{name}"] = (cx + cw // 2, cy + CHIP_H, root, is_list)
        d = resolve_node(n.ref)
        for name, cx, cy, cw in _chips(d.inputs, pl.x, pl.y - CHIP_H):
            in_pos[f"{nid}:{name}"] = (cx + cw // 2, cy)

    paths = []
    for e in cg.edges:
        a, b = out_pos.get(e.src), in_pos.get(e.dst)
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
        n = cg.nodes[nid]
        d = resolve_node(n.ref)
        tag = {NodeKind.INPUT: "I", NodeKind.PROCESSING: "P", NodeKind.OUTPUT: "O"}[n.kind]
        shape = {NodeKind.INPUT: "inp", NodeKind.PROCESSING: "prc", NodeKind.OUTPUT: "out"}[n.kind]

        # 입력 칩은 선언이 아니라 **컴파일이 확정한 타입**을 보여준다.
        # 제네릭이 남아 있으면 그 자체가 눈에 띄어야 한다.
        in_types = {p: n.input_types.get(p) or d.inputs[p].type for p in d.inputs}
        chips_in = "".join(
            f'<div class="chip" style="left:{cx - pl.x}px;top:0;width:{cw}px;'
            f'background:{T.port_color(_type_label(in_types[name])[1], _type_label(in_types[name])[2])}">'
            f'<span class="t">&lt;{html.escape(_type_label(in_types[name])[0])}&gt;</span>'
            f'<span class="p">{html.escape(name)}</span></div>'
            for name, cx, cy, cw in _chips(d.inputs, pl.x, 0)
        )
        chips_out = "".join(
            f'<div class="chip" style="left:{cx - pl.x}px;top:0;width:{cw}px;'
            f'background:{T.port_color(_type_label(n.output_types[name])[1], _type_label(n.output_types[name])[2])}">'
            f'<span class="t">&lt;{html.escape(_type_label(n.output_types[name])[0])}&gt;</span>'
            f'<span class="p">{html.escape(name)}</span></div>'
            for name, cx, cy, cw in _chips(n.output_types, pl.x, 0)
        )

        cards.append(
            f'<div class="node {shape}" style="left:{pl.x}px;top:{pl.y}px" '
            f'title="{html.escape(nid)} · {html.escape(n.ref)}">'
            f'<div class="ports top">{chips_in}</div>'
            f'<div class="card" style="border-top:3px solid {T.category_color(n.category)}">'
            f'<div class="hd"><span class="nm">{html.escape(nid)}</span>'
            f'<span class="badge b{tag}">{tag}</span></div>'
            f'<div class="ref">{html.escape(n.ref)}</div>'
            f'<div class="sum">{html.escape(_summary(cg, nid))}</div>'
            f"</div>"
            f'<div class="ports bot">{chips_out}</div>'
            f"</div>"
        )

    # ── 좌측 레일: 카테고리별 노드 수 ────────────────────────────────
    cats: Dict[str, int] = {}
    for nid in cg.order:
        cats[cg.nodes[nid].category] = cats.get(cg.nodes[nid].category, 0) + 1
    lib = "".join(
        f'<div class="lib"><span class="dot" style="background:{T.category_color(c)}"></span>'
        f"{html.escape(c)}<span class=\"n\">{k}</span></div>"
        for c, k in sorted(cats.items())
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

    counts = {
        "I": sum(1 for i in cg.order if cg.nodes[i].kind is NodeKind.INPUT),
        "P": sum(1 for i in cg.order if cg.nodes[i].kind is NodeKind.PROCESSING),
        "O": sum(1 for i in cg.order if cg.nodes[i].kind is NodeKind.OUTPUT),
    }
    head = title or cg.id or "graph"

    return _TEMPLATE.format(
        title=html.escape(head),
        css=_CSS,
        lib=lib,
        cards="".join(cards),
        paths="".join(paths),
        w=width,
        h=height,
        details="".join(details),
        spec_hash=cg.spec_hash,
        nodes=len(cg.nodes),
        edges=len(cg.edges),
        lanes=max_lane + 1,
        ci=counts["I"],
        cp=counts["P"],
        co=counts["O"],
        note=html.escape(note or "compile OK"),
        profile=html.escape(cg.runtime_profile),
    )


def write(cg: CompiledGraph, path: str, *, title: str = "", note: str = "") -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(cg, title=title, note=note))
    return path


_CSS = f"""
*{{box-sizing:border-box}}
body{{margin:0;background:{T.SURFACE['chrome']};color:#D8DCDF;
     font:13px/1.45 "Segoe UI","Malgun Gothic",sans-serif}}
.menubar{{background:{T.SURFACE['chrome']};padding:6px 12px;border-bottom:1px solid {T.SURFACE['line']};
        font-size:12px;color:#8A9196}}
.toolbar{{background:{T.SURFACE['toolbar']};padding:7px 12px;border-bottom:1px solid {T.SURFACE['line']};
        display:flex;gap:14px;align-items:center;font-size:12px}}
.toolbar b{{color:{T.NODE['accent']};font-weight:600}}
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
.stage{{position:relative}}
svg.wires{{position:absolute;inset:0;pointer-events:none}}
.node{{position:absolute;width:{CARD_W}px}}
.ports{{position:relative;height:{CHIP_H}px}}
.chip{{position:absolute;height:{CHIP_H - 4}px;border-radius:3px;padding:2px 6px;overflow:hidden;
      color:#fff;font-size:10px;line-height:1.15}}
.chip .t{{display:block;opacity:.92;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.chip .p{{display:block;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.card{{background:{T.NODE['bg']};border:1px solid {T.NODE['border']};border-radius:3px;
      height:{CARD_H}px;padding:7px 10px}}
.inp .card{{border-top-left-radius:12px;border-top-right-radius:12px}}
.out .card{{border-bottom-left-radius:12px;border-bottom-right-radius:12px}}
.inp .ports.top,.out .ports.bot{{display:none}}
.hd{{display:flex;align-items:center;gap:6px}}
.nm{{color:{T.NODE['title']};font-weight:600;font-size:13px;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap}}
.badge{{margin-left:auto;font-size:10px;border:1px solid;border-radius:2px;padding:0 4px}}
.bI{{color:{T.PORT['Image']}}} .bP{{color:#8A9196}} .bO{{color:{T.PORT['Table']}}}
.ref,.sum{{color:{T.NODE['muted']};font-size:11px;white-space:nowrap;overflow:hidden;
          text-overflow:ellipsis}}
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
.log b{{color:#3FB27F}}
"""

_TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head>
<body>
<div class="menubar">File &nbsp; Edit &nbsp; View &nbsp; Solution &nbsp; Node &nbsp; Plugins &nbsp; Settings &nbsp; Help</div>
<div class="toolbar">
  <b>{title}</b>
  <span>노드 {nodes} · 배선 {edges} · 레인 {lanes}단</span>
  <span>I {ci} / P {cp} / O {co}</span>
  <span>프로파일 {profile}</span>
  <span style="margin-left:auto">읽기 전용 뷰어 — 편집은 아직 없다</span>
</div>
<div class="shell">
  <div class="rail">
    <h4>Node Library</h4>
    {lib}
  </div>
  <div class="canvas"><div class="stage" style="width:{w}px;height:{h}px">
    <svg class="wires" width="{w}" height="{h}">{paths}</svg>
    {cards}
  </div></div>
  <div class="side">
    <h4>Node Quick Info</h4>
    {details}
  </div>
</div>
<div class="log"><b>{note}</b> &nbsp; spec_hash {spec_hash}</div>
</body></html>
"""
