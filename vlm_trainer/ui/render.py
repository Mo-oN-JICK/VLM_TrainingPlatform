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


def _state_of(report: Any, nid: str) -> Tuple[str, str]:
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
        state, state_extra = _state_of(report, nid)

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
            f'<span class="badge b{tag}">{tag}</span></div>'
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
        f'<h4>Debug Output</h4>{debug}'
        if debug
        else '<h4>Debug Output</h4><div class="doc">토글이 꺼져 있어 미리보기를 만들지 않았다.</div>'
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
            'window.EDITABLE = "1";</script>'
            f"<script>{_EDITOR_JS}</script>"
        )
    tools = _EDITOR_TOOLS if editable else _VIEWER_TOOLS
    history = _history_panel(editor) if (editable and editor is not None) else ''
    banner_html = f'<div class="banner">{html.escape(banner)}</div>' if banner else ""

    return _TEMPLATE.format(
        title=html.escape(head),
        params=_params_panel(cg) if editable else '',
        history=history,
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
        spec_hash=cg.spec_hash,
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
.hrow{{border-left:2px solid transparent;padding:4px 8px;margin-bottom:2px;cursor:pointer;
      font-size:11.5px;color:#A8B0B6;display:grid;grid-template-columns:58px 1fr 64px;gap:6px}}
.hrow:hover{{background:{T.NODE['bg']}}}
.hrow.cur{{border-left-color:{T.NODE['border']};background:{T.NODE['bg']};color:#D8DCDF}}
.hrow .ht{{color:#6F7478;font-size:10.5px}}
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


def _history_panel(editor: Any) -> str:
    """History 탭. 항목을 클릭하면 그 시점으로 되감는다."""
    rows = []
    for h in editor.history_view():
        cls = "hrow cur" if h["current"] else "hrow"
        diff = "".join(
            f'<div class="hd{"p" if ln.startswith("+") else "m"}">{html.escape(ln)}</div>'
            for ln in h["diff"][:6]
        )
        rows.append(
            f'<div class="{cls}" onclick="vlmtRewind({h["index"]})">'
            f'<span class="ht">{html.escape(h["at"])}</span>'
            f'<span class="hl">{html.escape(h["label"])}</span>'
            f'<span class="hh">{html.escape(h["spec_hash"][3:11])}</span>'
            f"{diff}</div>"
        )
    return '<h4>History</h4>' + ("".join(reversed(rows)) or '<div class="doc">기록이 없다.</div>')


def _params_panel(cg: CompiledGraph) -> str:
    """노드마다 Node Parameters 블록. 카드를 고르면 그 블록만 보인다."""
    from .api import param_meta

    blocks = []
    for nid in cg.order:
        n = cg.nodes[nid]
        rows = []
        for m in param_meta(n.ref, n.params):
            name, kind, value = m["name"], m["kind"], m["value"]
            marks = ""
            if m["overridable"]:
                marks += '<span class="mk r" title="Parameter Recipe가 덮을 수 있다">recipe</span>'
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
