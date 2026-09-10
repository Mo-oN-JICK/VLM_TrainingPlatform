"""읽기 전용 그래프 뷰어 — 컴파일된 그래프를 한 장의 HTML로 그린다.

편집기가 아니다. 캔버스 규약(수직 흐름, 상단 입력·하단 출력, Input 최상단·Output 최하단,
포트 타입 색, 배선 색 = 소스 포트 타입 색)이 실제 그래프에서 지켜지는지 눈으로 확인하는 수단이다.
서버도 프레임워크도 없다. 편집기는 이 위에 얹는다. 설계 문서 12.
"""

from __future__ import annotations

import html
import json
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


def state_of(report: Any, nid: str) -> Tuple[str, str]:
    """(상태 이름, 부가 라벨). 실행 보고가 없으면 pending."""
    if report is None:
        return "pending", ""
    st = report.states_of(nid)
    if not st:
        return "pending", ""
    for name in ("failed", "partial", "running", "cached", "success", "skipped"):
        if st.get(name):
            extra = f"{st[name]}건"
            ms = report.node_ms.get(nid, 0.0)
            if ms and name == "success":
                extra += f" · {ms:.0f}ms"
            return name, extra
    return "pending", ""


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
) -> str:
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
        state, state_extra = state_of(report, nid)

        # 입력 칩은 선언이 아니라 **컴파일이 확정한 타입**을 보여준다.
        # 제네릭이 남아 있으면 그 자체가 눈에 띄어야 한다.
        in_types = {p: n.input_types.get(p) or d.inputs[p].type for p in d.inputs}
        chips_in = "".join(
            f'<div class="chip cin" data-ref="{nid}:{name}" '
            f'style="left:{cx - pl.x}px;top:0;width:{cw}px;'
            f'background:{T.port_color(_type_label(in_types[name])[1], _type_label(in_types[name])[2])}">'
            f'<span class="t">&lt;{html.escape(_type_label(in_types[name])[0])}&gt;</span>'
            f'<span class="p">{html.escape(name)}</span></div>'
            for name, cx, cy, cw in _chips(d.inputs, pl.x, 0)
        )
        chips_out = "".join(
            f'<div class="chip cout" data-ref="{nid}:{name}" '
            f'style="left:{cx - pl.x}px;top:0;width:{cw}px;'
            f'background:{T.port_color(_type_label(n.output_types[name])[1], _type_label(n.output_types[name])[2])}">'
            f'<span class="t">&lt;{html.escape(_type_label(n.output_types[name])[0])}&gt;</span>'
            f'<span class="p">{html.escape(name)}</span></div>'
            for name, cx, cy, cw in _chips(n.output_types, pl.x, 0)
        )

        cards.append(
            f'<div class="node {shape}" data-node="{html.escape(nid)}" '
            f'style="left:{pl.x}px;top:{pl.y}px" '
            f'title="{html.escape(nid)} · {html.escape(n.ref)}">'
            f'<div class="ports top">{chips_in}</div>'
            f'<div class="card" style="border-top:3px solid {T.category_color(n.category)};'
            f'border-left:4px solid {T.STATE.get(state, "#4A4A4A")}">'
            f'<div class="hd"><span class="nm">{html.escape(nid)}</span>'
            f'<span class="badge b{tag}">{tag}</span>'
            + (f"<span class=\"del\" onclick=\"vlmtRemove('{nid}')\">&times;</span>" if editable else "")
            + "</div>"
            f'<div class="ref">{html.escape(n.ref)}</div>'
            f'<div class="sum">{html.escape(_summary(cg, nid))}</div>'
            f'<div class="st"><span class="sdot" style="background:{T.STATE.get(state, "#4A4A4A")}">'
            f'</span>{state}{" · " + html.escape(state_extra) if state_extra else ""}</div>'
            f"</div>"
            f'<div class="ports bot">{chips_out}</div>'
            f"</div>"
        )

    # ── 좌측 레일: 카테고리별 노드 수 ────────────────────────────────
    cats: Dict[str, int] = {}
    for nid in cg.order:
        cats[cg.nodes[nid].category] = cats.get(cg.nodes[nid].category, 0) + 1
    lib = _library_panel(editable)

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
        + (debug or '<div class="doc">토글이 꺼져 있어 미리보기를 만들지 않았다.</div>')
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

    # 편집 모드에서는 호환성 표를 페이지에 함께 실어 보낸다.
    # 드래그 중에 서버를 다시 부르지 않고도 놓을 수 있는 포트만 밝히기 위해서다.
    # 뷰어는 스크립트를 아예 싣지 않는다 — 자체 완결이면서 죽은 태그도 남기지 않는다
    scripts = ""
    if editable:
        compat_json = json.dumps(compat or {}, ensure_ascii=False)
        occupied_json = json.dumps(sorted({e.dst for e in cg.edges}))
        scripts = (
            f"<script>window.COMPAT = {compat_json}; window.OCCUPIED = {occupied_json}; "
            f"window.STATE_COLORS = {json.dumps(T.STATE)}; "
            'window.EDITABLE = "1";</script>'
            f"<script>{_EDITOR_JS}</script>"
        )
    tools = _EDITOR_TOOLS if editable else _VIEWER_TOOLS
    history = _history_panel(editor) if (editable and editor is not None) else ''
    recipe = _recipe_panel(editor) if (editable and editor is not None) else ''
    # 오버레이가 걸려 있으면 화면의 해시와 저장될 해시가 다르다. 감추지 않는다.
    spec_hash = cg.spec_hash
    base = getattr(editor, 'base', None) if editable else None
    if base is not None and base.spec_hash != cg.spec_hash:
        spec_hash = f"{cg.spec_hash} (레시피 적용) · 스펙 {base.spec_hash}"
    overlaid = set(getattr(editor, 'overlay', None) and
                   [f'{n}:{p}' for (n, p) in editor._overlay_paths()] or [])
    banner_html = f'<div class="banner">{html.escape(banner)}</div>' if banner else ""

    return _TEMPLATE.format(
        title=html.escape(head),
        params=_params_panel(cg, overlaid) if editable else '',
        history=history,
        recipe=recipe,
        scripts=scripts,
        tools=tools,
        banner=banner_html,
        css=_CSS,
        lib=lib,
        cards="".join(cards),
        paths="".join(paths),
        w=width,
        h=height,
        details="".join(details),
        debug=debug_block,
        quarantine=quarantine,
        spec_hash=spec_hash,
        nodes=len(cg.nodes),
        edges=len(cg.edges),
        lanes=max_lane + 1,
        ci=counts["I"],
        cp=counts["P"],
        co=counts["O"],
        note=html.escape(note or "compile OK"),
        runline=html.escape(
            f"처리 {report.processed}건 · 캐시 {report.cache}" if report is not None else "실행 전"
        ),
        profile=html.escape(cg.runtime_profile),
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
.st{{margin-top:3px;font-size:10.5px;color:#8A9196;display:flex;align-items:center;gap:5px}}
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
.params{{display:none;border-bottom:1px solid {T.SURFACE['line']};padding-bottom:8px;margin-bottom:8px}}
.params.on{{display:block}}
.phd{{color:#C6CCD1;font-size:12.5px;margin:2px 0 8px}}
.phd em{{color:#6F7478;font-style:normal;font-size:11px}}
.prow{{display:grid;grid-template-columns:1fr 148px;gap:8px;align-items:center;margin-bottom:5px}}
.prow label{{color:#A8B0B6;font-size:11.5px;display:flex;gap:4px;align-items:center}}
.prow input[type=text],.prow input[type=number]{{background:{T.SURFACE['canvas']};color:#D8DCDF;
    border:1px solid {T.SURFACE['line']};border-radius:2px;padding:3px 6px;font-size:11.5px;width:100%}}
.prow input:focus{{outline:1px solid {T.NODE['border']}}}
.mk{{font-size:9px;border:1px solid;border-radius:2px;padding:0 3px}}
.mk.r{{color:{T.PORT['Image']}}} .mk.t{{color:{T.STATE['partial']}}}
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
.btn:disabled{{opacity:.45;cursor:default}}
.sep{{width:1px;height:16px;background:{T.SURFACE['line']};margin:0 4px}}
.tgl{{display:flex;gap:4px;align-items:center;font-size:11px;color:#A8B0B6;margin-left:6px}}
.tgl input[type=number]{{width:44px;background:{T.SURFACE['canvas']};color:#D8DCDF;
      border:1px solid {T.SURFACE['line']};font-size:11px;padding:1px 4px}}
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
.banner{{background:#3A2A2A;color:#E8B0B0;padding:6px 12px;font-size:12px;
        border-bottom:1px solid {T.SURFACE['line']};white-space:pre-wrap}}
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
  <span>노드 {nodes} · 배선 {edges} · 레인 {lanes}단</span>
  <span>I {ci} / P {cp} / O {co}</span>
  <span>프로파일 {profile}</span>
  {tools}
</div>
{banner}
<div id="toast"></div>
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
    {recipe}
    {history}
    {params}
    {debug}
    {quarantine}
    <h4>Node Quick Info</h4>
    {details}
  </div>
</div>
<div class="log"><b>{note}</b> &nbsp; {runline} &nbsp; spec_hash {spec_hash}</div>
{scripts}
</body></html>
"""


_VIEWER_TOOLS = '<span style="margin-left:auto">읽기 전용 뷰어</span>'

_EDITOR_TOOLS = (
    '<button class="btn" onclick="vlmtUndo()">Undo</button>'
    '<button class="btn" onclick="vlmtRedo()">Redo</button>'
    '<button class="btn" onclick="vlmtSave()">Save</button>'
    '<button class="btn" onclick="location.reload()">Reload</button>'
    '<span class="sep"></span>'
    '<button class="btn go" id="runbtn" onclick="vlmtRun()">Run</button>'
    '<button class="btn" id="stopbtn" onclick="vlmtRunStop()" disabled>Stop</button>'
    '<label class="tgl" title="꺼져 있으면 미리보기를 생성조차 하지 않는다">'
    '<input type="checkbox" id="dbgout">Debug Output</label>'
    '<label class="tgl">샘플 <input type="number" id="runlimit" value="8" min="1" max="999"></label>'
    '<span class="runline" id="runline"></span>'
    '<span class="hint">출력 칩을 끌어 입력 칩에 놓는다 · 입력 칩 우클릭으로 배선 제거</span>'
)

# 편집기 스크립트. 드래그 중에는 호환되는 입력만 밝히고, 나머지에는 **놓을 수 없다**.
# 연결이 성립해도 서버가 다시 컴파일해 G1/G2를 통과해야 받아들여진다.
_EDITOR_JS = r"""
const toast = (msg, bad) => {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = bad ? 'bad show' : 'ok show';
  clearTimeout(window._tt);
  window._tt = setTimeout(() => { t.className = ''; }, 5000);
};

async function post(path, body) {
  const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'},
                              body: JSON.stringify(body)});
  return {code: r.status, data: await r.json()};
}

async function vlmtSave() {
  const {code, data} = await post('/api/save', {});
  toast(code === 200 ? '저장했다: ' + data.path : '저장 실패: ' + (data.reason || ''), code !== 200);
}

let drag = null, temp = null;
const svg = document.querySelector('svg.wires');
const stageBox = () => document.querySelector('.stage').getBoundingClientRect();

document.addEventListener('mousedown', (ev) => {
  const chip = ev.target.closest('.chip.cout');
  if (!chip || !window.EDITABLE) return;
  ev.preventDefault();
  const row = window.COMPAT[chip.dataset.ref] || {};
  drag = {src: chip.dataset.ref, row: row};

  document.querySelectorAll('.chip.cin').forEach(c => {
    const why = row[c.dataset.ref];
    if (why === '') {
      c.classList.add('okdrop');
      if (window.OCCUPIED.indexOf(c.dataset.ref) >= 0) c.title = '놓으면 기존 배선을 교체한다';
    }
    else { c.classList.add('nodrop'); c.title = why || '타입이 맞지 않는다'; }
  });

  const st = stageBox(), b = chip.getBoundingClientRect();
  temp = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  temp.setAttribute('stroke', '#57F7E6');
  temp.setAttribute('fill', 'none');
  temp.setAttribute('stroke-width', '2');
  temp.setAttribute('stroke-dasharray', '5 4');
  temp.dataset.x = b.left - st.left + b.width / 2;
  temp.dataset.y = b.top - st.top + b.height;
  svg.appendChild(temp);
});

document.addEventListener('mousemove', (ev) => {
  if (!drag || !temp) return;
  const st = stageBox();
  const x = ev.clientX - st.left, y = ev.clientY - st.top;
  const x1 = +temp.dataset.x, y1 = +temp.dataset.y, m = (y1 + y) / 2;
  temp.setAttribute('d', 'M ' + x1 + ' ' + y1 + ' C ' + x1 + ' ' + m + ', ' + x + ' ' + m + ', ' + x + ' ' + y);
});

document.addEventListener('mouseup', async (ev) => {
  if (!drag) return;
  const chip = ev.target.closest('.chip.cin');
  const src = drag.src, row = drag.row;
  drag = null;
  if (temp) { temp.remove(); temp = null; }
  document.querySelectorAll('.chip.cin').forEach(c => c.classList.remove('okdrop', 'nodrop'));
  if (!chip) return;

  const dst = chip.dataset.ref, why = row[dst];
  if (why !== '') {                       // 드롭 자체가 되지 않는다
    toast(dst + ' 에는 놓을 수 없다 — ' + (why || '타입 불일치'), true);
    return;
  }
  const {code, data} = await post('/api/connect', {from: src, to: dst});
  if (code === 200) location.reload();
  else toast('연결 거부: ' + (data.reason || ''), true);
});


async function vlmtAdd(type) {
  const {code, data} = await post('/api/add', {type: type});
  if (code === 200) { sessionStorage.setItem('sel', data.id); location.reload(); }
  else toast('추가 거부: ' + (data.reason || ''), true);
}

async function vlmtRemove(nid) {
  const {code, data} = await post('/api/remove', {node: nid});
  if (code === 200) { sessionStorage.removeItem('sel'); location.reload(); }
  else toast('삭제 거부: ' + (data.reason || ''), true);
}

// 라이브러리에서 캔버스로 끌어다 놓아도 추가된다. 클릭 추가와 결과는 같다.
// 놓은 위치는 자리를 정하지 않는다 — 캔버스는 위상 순서로 스스로 정렬한다.
// 배선 드래그와 같은 마우스 이벤트를 쓴다. HTML5 drag-and-drop을 섞으면
// 손잡이가 두 벌이 되고 한쪽만 조용히 썩는다.
let libdrag = null, libghost = null;

document.addEventListener('mousedown', (ev) => {
  const item = ev.target.closest('.libnode');
  if (!item) return;
  ev.preventDefault();                      // 텍스트 선택 드래그를 막는다
  libdrag = {type: item.dataset.type, moved: false, x: ev.clientX, y: ev.clientY};
});

document.addEventListener('mousemove', (ev) => {
  if (!libdrag) return;
  if (!libdrag.moved && Math.abs(ev.clientX - libdrag.x) + Math.abs(ev.clientY - libdrag.y) < 4) return;
  libdrag.moved = true;
  if (!libghost) {
    libghost = document.createElement('div');
    libghost.className = 'libghost';
    libghost.textContent = libdrag.type.split('@')[0];
    document.body.appendChild(libghost);
  }
  libghost.style.left = (ev.clientX + 12) + 'px';
  libghost.style.top = (ev.clientY + 12) + 'px';
  const canvas = document.querySelector('.canvas');
  if (canvas) canvas.classList.toggle('candrop', !!ev.target.closest('.canvas'));
});

document.addEventListener('mouseup', (ev) => {
  if (!libdrag) return;
  const d = libdrag;
  libdrag = null;
  if (libghost) { libghost.remove(); libghost = null; }
  const canvas = document.querySelector('.canvas');
  if (canvas) canvas.classList.remove('candrop');
  // 끌지 않았으면 클릭이다 — onclick 이 그대로 처리한다
  if (d.moved && ev.target.closest('.canvas')) vlmtAdd(d.type);
});

document.addEventListener('keydown', (ev) => {
  if (!window.EDITABLE || ev.key !== 'Delete') return;
  const sel = document.querySelector('.node.sel');
  if (sel) vlmtRemove(sel.dataset.node);
});

async function vlmtRecipe(id) {
  const {code, data} = await post('/api/recipe/select', {id: id});
  if (code === 200) location.reload(); else toast('레시피 적용 거부: ' + (data.reason || ''), true);
}
async function vlmtRecipeClear() {
  const {code, data} = await post('/api/recipe/select', {id: null});
  if (code === 200) location.reload(); else toast(data.reason || '', true);
}
async function vlmtRecipeAdd(path) {
  const {code, data} = await post('/api/recipe/add-path', {path: path});
  if (code === 200) location.reload(); else toast('축 추가 거부: ' + (data.reason || ''), true);
}
async function vlmtRecipeDrop(path) {
  const {code, data} = await post('/api/recipe/drop-path', {path: path});
  if (code === 200) location.reload(); else toast(data.reason || '', true);
}
async function vlmtRecipeStore(id) {
  const el = document.getElementById('rname');
  const {code, data} = await post('/api/recipe/store', {id: id, name: el ? el.value : ''});
  if (code === 200) location.reload(); else toast('담기 거부: ' + (data.reason || ''), true);
}
async function vlmtRecipeDelete(id) {
  const {code, data} = await post('/api/recipe/delete', {id: id});
  if (code === 200) location.reload(); else toast(data.reason || '', true);
}
async function vlmtRecipeActive(id) {
  const {code, data} = await post('/api/recipe/active', {id: id});
  if (code === 200) location.reload(); else toast(data.reason || '', true);
}

// ── 실행 ────────────────────────────────────────────────────────────────
// 편집기는 실행 경로를 따로 갖지 않는다. 서버가 `vlmt run`을 그대로 띄우고,
// 우리는 그 프로세스가 남기는 진행 스냅샷을 폴링해 카드에 칠하기만 한다.
let runTimer = null;

async function vlmtRun() {
  const limit = +(document.getElementById('runlimit') || {}).value || 8;
  const dbg = !!(document.getElementById('dbgout') || {}).checked;
  const {code, data} = await post('/api/run', {limit: limit, debug_output: dbg});
  if (code !== 200) { toast('실행 거부: ' + (data.reason || ''), true); return; }
  toast('실행 시작 · ' + data.run_id);
  // 지난 실행의 미리보기를 남겨두지 않는다. 토글이 꺼진 실행에서 옛 값이 보이면
  // "꺼져 있으면 만들지 않는다"는 규약이 화면에서 무너진다.
  const box = document.getElementById('debugout');
  if (box) box.innerHTML = '<div class="doc">' +
    (dbg ? '실행 중 — 미리보기를 기다린다.' : '토글이 꺼져 있어 미리보기를 만들지 않았다.') + '</div>';
  vlmtRunPoll();
}

async function vlmtRunStop() {
  const {code, data} = await post('/api/run/stop', {});
  if (code !== 200) toast(data.reason || '', true);
}

function vlmtPaint(states) {
  for (const [nid, st] of Object.entries(states || {})) {
    const node = document.querySelector('.node[data-node="' + CSS.escape(nid) + '"]');
    if (!node) continue;
    const color = window.STATE_COLORS[st.state] || '#4A4A4A';
    const card = node.querySelector('.card');
    if (card) card.style.borderLeftColor = color;
    const line = node.querySelector('.st');
    if (line) {
      line.innerHTML = '<span class="sdot" style="background:' + color + '"></span>' +
        st.state + (st.extra ? ' · ' + st.extra : '');
    }
  }
}

function vlmtDebugOut(previews) {
  // Debug Output 규약: 토글이 꺼져 있으면 애초에 만들어지지 않는다.
  // 여기서 감추는 것이 아니라, 받을 것이 없는 것이다.
  const box = document.getElementById('debugout');
  if (!box || !previews) return;
  const ids = Object.keys(previews);
  if (!ids.length) return;
  box.innerHTML = ids.map(nid =>
    '<details open><summary>' + nid + ' <em>' + (previews[nid].kind || '') + '</em></summary>' +
    (previews[nid].image ? '<img class="pvimg" src="' + previews[nid].image + '?t=' + Date.now() + '">' : '') +
    '<pre class="pv"></pre></details>').join('');
  ids.forEach((nid, i) => { box.querySelectorAll('pre.pv')[i].textContent = previews[nid].text || ''; });
}

async function vlmtRunPoll() {
  if (runTimer) clearTimeout(runTimer);
  const {code, data} = await post('/api/run/state', {});
  if (code !== 200) return;

  vlmtPaint(data.states);
  vlmtDebugOut(data.previews);
  const line = document.getElementById('runline');
  const run = document.getElementById('runbtn'), stop = document.getElementById('stopbtn');
  if (run) run.disabled = !!data.running;
  if (stop) stop.disabled = !data.running;

  if (line) {
    if (data.running || data.phase) {
      const q = data.quarantine_total ? ' · 격리 ' + data.quarantine_total : '';
      const tail = data.running ? '' : (data.stopped ? ' · 중지' : (data.aborted ? ' · 중단' :
                   (data.exit ? ' · 실패(exit ' + data.exit + ')' : ' · 끝')));
      line.textContent = data.run_id + ' ' + data.processed + '/' + data.total + q + tail;
      line.className = 'runline' + (data.aborted || (data.exit && !data.stopped) ? ' bad' : '');
    } else {
      line.textContent = '';
    }
  }
  if (data.aborted) toast('중단: ' + data.aborted, true);
  else if (!data.running && data.exit && !data.stopped)
    toast('실행 실패 (exit ' + data.exit + ')\n' + (data.console || ''), true);

  if (data.running) runTimer = setTimeout(vlmtRunPoll, 700);
}

// 페이지를 새로 열어도 돌고 있는 실행을 이어서 본다
window.addEventListener('DOMContentLoaded', vlmtRunPoll);

async function vlmtUndo() {
  const {code, data} = await post('/api/undo', {});
  if (code === 200) location.reload(); else toast(data.reason || '', true);
}
async function vlmtRedo() {
  const {code, data} = await post('/api/redo', {});
  if (code === 200) location.reload(); else toast(data.reason || '', true);
}
async function vlmtRewind(i) {
  const {code, data} = await post('/api/rewind', {index: i});
  if (code === 200) location.reload();
  else toast('되감기 실패: ' + (data.reason || ''), true);
}
document.addEventListener('keydown', (ev) => {
  if (!window.EDITABLE) return;
  const z = ev.key === 'z' || ev.key === 'Z';
  if ((ev.ctrlKey || ev.metaKey) && z) { ev.preventDefault(); ev.shiftKey ? vlmtRedo() : vlmtUndo(); }
});

function vlmtSelect(nid) {
  document.querySelectorAll('.node').forEach(n => n.classList.toggle('sel', n.dataset.node === nid));
  document.querySelectorAll('.params').forEach(p => p.classList.toggle('on', p.dataset.node === nid));
}

document.addEventListener('click', (ev) => {
  const card = ev.target.closest('.node');
  if (card) vlmtSelect(card.dataset.node);
});

async function vlmtParam(node, param, value, kind) {
  if (kind === 'json') {
    try { value = JSON.parse(value); }
    catch (e) { toast('값이 JSON이 아니다: ' + value, true); return; }
  }
  const {code, data} = await post('/api/param', {node: node, param: param, value: value});
  if (code === 200) { sessionStorage.setItem('sel', node); location.reload(); }
  else toast('거부: ' + (data.reason || ''), true);
}

window.addEventListener('DOMContentLoaded', () => {
  const sel = sessionStorage.getItem('sel');
  if (sel) vlmtSelect(sel);
});

document.addEventListener('contextmenu', async (ev) => {
  const chip = ev.target.closest('.chip.cin');
  if (!chip || !window.EDITABLE) return;
  ev.preventDefault();
  const {code, data} = await post('/api/disconnect', {to: chip.dataset.ref});
  if (code === 200) location.reload();
  else toast('제거 실패: ' + (data.reason || ''), true);
});
"""


def render_editor(editor: Any) -> str:
    """편집 가능한 페이지. 변경은 전부 /api/* 를 거쳐 코어로 간다."""
    from .api import compat_matrix

    cg = editor.compiled
    if cg is None:
        return (
            "<!doctype html><meta charset='utf-8'><body style='background:#1A1A1A;color:#E8B0B0;"
            "font:13px sans-serif;padding:24px'><h3>컴파일 실패</h3><pre>"
            + html.escape(editor.error)
            + "</pre></body>"
        )
    return render(
        cg,
        title=cg.name or cg.id,
        note="편집 중" + (" · 저장하지 않은 변경 있음" if editor.dirty else ""),
        editable=True,
        compat=compat_matrix(cg),
        banner=editor.error,
        editor=editor,
    )


def _library_panel(editable: bool) -> str:
    """Node Library — 카테고리별 노드 목록. 편집 모드에서는 클릭하거나 캔버스로 끌어 추가한다."""
    from ..core.registry import categories

    out = []
    for cat, defs in categories().items():
        rows = []
        for d in defs:
            tag = {NodeKind.INPUT: "I", NodeKind.PROCESSING: "P", NodeKind.OUTPUT: "O"}[d.kind]
            attrs = (
                f' data-type="{html.escape(d.ref)}"'
                f' onclick="vlmtAdd(\'{html.escape(d.ref)}\')"'
                if editable
                else ""
            )
            rows.append(
                f'<div class="libnode"{attrs} title="{html.escape(d.doc.summary)}">'
                f'<span class="badge b{tag}">{tag}</span>{html.escape(d.type)}</div>'
            )
        out.append(
            f"<details><summary><span class='dot' style='background:"
            f"{T.category_color(cat)}'></span>{html.escape(cat)}"
            f"<span class='n'>{len(defs)}</span></summary>{''.join(rows)}</details>"
        )
    return "".join(out)


def _recipe_panel(editor: Any) -> str:
    '''Parameter Recipe 패널.

    레시피는 값만 덮는 오버레이라 프로젝트 스펙을 건드리지 않는다. 그래서 적용 중이라는
    사실 자체를 눈에 띄게 말해 준다 — 화면의 값과 저장될 값이 다르기 때문이다.
    '''
    v = editor.recipe_view()
    if not v.get("path"):
        return ""

    st = v["status"]
    label = {"": "없음"}.get(st, st)
    head = (
        '<h4>Parameter Recipe'
        + ('<span class="rdirty">저장 안 됨</span>' if v["dirty"] else '')
        + '</h4>'
        + f'<div class="rstat">프로젝트 상태 <b>{html.escape(label)}</b>'
        + ('<span class="rwhy">직접 고쳐서 레시피와 어긋난다</span>'
           if st == "Customized" else '')
        + '</div>'
    )

    rows = []
    for r in v["recipes"]:
        cls = "rrow" + (" on" if r["applied"] else "")
        star = '<span class="ract" title="active — 프로젝트가 이 레시피대로다">*</span>' if r["active"] else ''
        diff = sum(1 for o in r["overrides"] if o["differs"])
        note = f'<div class="rnote">{html.escape(r["note"])}</div>' if r["note"] else ''
        rows.append(
            f'<div class="{cls}">'
            f'<span class="rname" onclick="vlmtRecipe({r["id"]})">{star}{html.escape(r["label"])}</span>'
            f'<span class="rn">{len(r["overrides"])}개'
            + (f' · 현재와 {diff}개 다름' if diff else ' · 현재와 같다') + '</span>'
            f'<span class="rdel" title="레시피 삭제" onclick="vlmtRecipeDelete({r["id"]})">&times;</span>'
            f'{note}</div>'
        )

    applied = v["applied"]
    if v["overlay"]:
        orows = []
        for o in v["overlay"]:
            base = (f'<span class="rbase" title="프로젝트 스펙의 값">스펙 {html.escape(json.dumps(o["base"], ensure_ascii=False))}</span>'
                    if o["differs"] else '')
            orows.append(
                f'<div class="orow"><span class="opath" title="{html.escape(o["path"])}">'
                f'{html.escape(o["display"])}</span>'
                f'<span class="oval">{html.escape(json.dumps(o["value"], ensure_ascii=False))}</span>'
                f'{base}'
                f'<span class="odrop" title="이 축을 레시피에서 뺀다" '
                f'onclick="vlmtRecipeDrop(\'{html.escape(o["path"])}\')">&minus;</span></div>'
            )
        applied_label = f'{applied:02d}번' if applied is not None else '이름 없는 조합'
        overlay = (
            '<div class="obar">'
            f'<b>{html.escape(applied_label)} 적용 중</b> — 화면의 값이다. '
            '<b>프로젝트에는 저장되지 않는다.</b>'
            '<button class="btn sm" onclick="vlmtRecipeClear()">벗기기</button>'
            '</div>'
            + ''.join(orows)
            + '<div class="orow store">'
            '<input type="text" id="rname" placeholder="이름(새 레시피)">'
            + (f'<button class="btn sm" onclick="vlmtRecipeStore({applied})">{applied:02d}번에 담기</button>'
               if applied is not None else '')
            + '<button class="btn sm" onclick="vlmtRecipeStore(null)">새 레시피로</button>'
            + (f'<button class="btn sm" onclick="vlmtRecipeActive({applied})">active로</button>'
               if applied is not None else '')
            + '</div>'
        )
    else:
        overlay = ('<div class="doc">파라미터 옆의 <b>+</b>를 누르면 그 값이 레시피의 축이 된다. '
                   '레시피를 누르면 그 값으로 화면이 바뀐다.</div>')

    return f'<div class="recipe">{head}{overlay}{"".join(rows)}</div>'


def _history_panel(editor: Any) -> str:
    """History 탭. 항목을 클릭하면 그 시점으로 되감는다."""
    rows = []
    for h in editor.history_view():
        cls = "hrow cur" if h["current"] else "hrow"
        if h.get("past"):
            cls += " past"  # 지난 세션에서 되살린 시점
        diff = "".join(
            f'<div class="hd{"p" if ln.startswith("+") else "m"}">{html.escape(ln)}</div>'
            for ln in h["diff"][:6]
        )
        rows.append(
            f'<div class="{cls}" onclick="vlmtRewind({h["index"]})" '
            f'title="{html.escape(h.get("when") or h["at"])}">'
            f'<span class="ht">{html.escape(h["at"])}</span>'
            f'<span class="hl">{html.escape(h["label"])}</span>'
            f'<span class="hh">{html.escape(h["spec_hash"][3:11])}</span>'
            f"{diff}</div>"
        )
    return '<h4>History</h4>' + ("".join(reversed(rows)) or '<div class="doc">기록이 없다.</div>')


def _params_panel(cg: CompiledGraph, overlaid: Any = ()) -> str:
    """노드마다 Node Parameters 블록. 카드를 고르면 그 블록만 보인다.

    레시피가 덮고 있는 파라미터는 표식을 단다 — 그 값을 고치면 스펙이 아니라
    오버레이가 바뀌기 때문이다."""
    from .api import param_meta

    blocks = []
    for nid in cg.order:
        n = cg.nodes[nid]
        rows = []
        for m in param_meta(n.ref, n.params):
            name, kind, value = m["name"], m["kind"], m["value"]
            marks = ""
            if f"{nid}:{name}" in overlaid:
                marks += ('<span class="mk o" title="레시피가 덮고 있다 — 고치면 오버레이가 바뀌고 프로젝트 스펙은 그대로다">레시피</span>')
            elif m["overridable"]:
                marks += '<span class="mk r" title="Parameter Recipe가 덮을 수 있다">recipe</span>'
                marks += (f'<span class="padd" title="이 값을 레시피의 축으로 만든다" '
                          f"onclick=\"vlmtRecipeAdd('{html.escape(nid)}.{html.escape(name)}')\">+</span>")
            if m["type_affecting"]:
                marks += '<span class="mk t" title="바꾸면 배선 타입이 다시 검사된다">type</span>'

            if kind == "bool":
                widget = (
                    f'<input type="checkbox" {"checked" if value else ""} '
                    f"onchange=\"vlmtParam('{nid}','{name}',this.checked)\">"
                )
            elif kind == "number":
                step = "1" if isinstance(value, int) else "any"
                widget = (
                    f'<input type="number" step="{step}" value="{html.escape(str(value))}" '
                    f"onchange=\"vlmtParam('{nid}','{name}',this.valueAsNumber)\">"
                )
            elif kind == "json":
                widget = (
                    f'<input type="text" value="{html.escape(json.dumps(value, ensure_ascii=False))}" '
                    f"onchange=\"vlmtParam('{nid}','{name}',this.value,'json')\">"
                )
            else:
                widget = (
                    f'<input type="text" value="{html.escape(str(value))}" '
                    f"onchange=\"vlmtParam('{nid}','{name}',this.value)\">"
                )
            rows.append(
                f'<div class="prow"><label>{html.escape(name)}{marks}</label>{widget}</div>'
            )

        blocks.append(
            f'<div class="params" data-node="{html.escape(nid)}">'
            f'<div class="phd">{html.escape(nid)} <em>{html.escape(n.ref)}</em></div>'
            + ("".join(rows) or '<div class="doc">파라미터가 없다.</div>')
            + "</div>"
        )
    return "".join(blocks)
