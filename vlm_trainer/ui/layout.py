"""캔버스 배치와 상태 계산 — 그리는 방식과 무관한 부분.

HTML도 Qt도 여기 결과를 받아 그린다. 이 파일이 따로 있는 이유는 **같은 그래프가 앱과
`vlmt view`에서 다르게 배치되면 안 되기 때문**이다. 좌표 코드가 두 벌로 갈라지는 순간
어느 쪽이 맞는지 아무도 말할 수 없게 된다.

여기에는 마크업을 만드는 코드가 없다. 상자의 크기와 자리, 포트 칩의 좌표, 카드에 칠할
상태와 그 라벨까지가 전부다. 색은 `tokens.py`가, HTML은 `render.py`가 맡는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core import humanize
from ..core.compiler import CompiledGraph
from ..core.node import NodeKind
from ..core.registry import resolve as resolve_node


CARD_W = 264
CARD_H = 84
CHIP_H = 30
LANE_GAP = 52
COL_GAP = 34
PAD = 48
GRID = 16       # 격자 한 칸. 상자를 옮기면 이 배수에 붙는다


HEAD_H = 26     # 카드 머리줄
ROW_H = 15      # 입력/처리/출력 한 줄
BODY_PAD = 10


def _card_rows(s: "Shown") -> int:
    """이 카드가 몇 줄을 쓰는가. 입력 + 처리 + 출력."""
    return max(1, len(s.inputs)) + 1 + max(1, len(s.outputs))


@dataclass
class Placed:
    id: str
    x: int
    y: int
    lane: int
    w: int = CARD_W
    h: int = CARD_H


def card_size(s: "Shown") -> Tuple[int, int]:
    """이 상자가 차지할 (폭, 높이).

    역할이 무거운 상자는 크게 그린다 — 포트가 많으면 세로로, Procedure처럼 안에 여러 노드를
    품으면 가로로 자란다. 상자 크기가 전부 같으면 무엇이 단순하고 무엇이 복잡한지 그림이
    말해 주지 못한다.
    """
    h = HEAD_H + BODY_PAD + _card_rows(s) * ROW_H
    w = CARD_W + (min(s.inner, 4) * 24 if s.inner else 0)
    return w, h


def _layout(shown: Dict[str, "Shown"], edges: List[Tuple[str, str]],
            fixed: Optional[Dict[str, Tuple[int, int]]] = None) -> Dict[str, Placed]:
    """위상 순서대로 레인을 쌓고, 레인 안에서는 상류의 무게중심으로 좌우를 정한다.

    `fixed`에 좌표가 있는 노드는 그 자리에 둔다 — 사람이 옮겨 둔 것을 다시 흩지 않는다.
    상자 크기가 제각각이므로 레인 높이는 그 레인에서 가장 높은 상자가 정한다.
    """
    fixed = fixed or {}
    lanes: Dict[int, List[str]] = {}
    for nid, s in shown.items():
        lanes.setdefault(s.lane, []).append(nid)

    size = {nid: card_size(s) for nid, s in shown.items()}
    lane_h = {
        lane: max(size[i][1] for i in ids) + CHIP_H * 2 + LANE_GAP
        for lane, ids in lanes.items()
    }
    lane_y: Dict[int, int] = {}
    y = PAD
    for lane in sorted(lanes):
        lane_y[lane] = y
        y += lane_h[lane]

    placed: Dict[str, Placed] = {}
    x_of: Dict[str, float] = {}

    for lane in sorted(lanes):
        ids = lanes[lane]

        def key(nid: str) -> Tuple[float, str]:
            ups = [
                x_of[a.split(":")[0]]
                for a, b in edges
                if b.split(":")[0] == nid and a.split(":")[0] in x_of
            ]
            return (sum(ups) / len(ups) if ups else 1e9, nid)

        ids = sorted(ids, key=key)
        x = PAD
        for nid in ids:
            w, h = size[nid]
            x_of[nid] = x
            px, py = fixed.get(nid, (x, lane_y[lane]))
            placed[nid] = Placed(id=nid, x=int(px), y=int(py), lane=lane, w=w, h=h)
            x += w + COL_GAP
    return placed


def _chips(ports: Dict[str, Any], x: int, y: int,
           card_w: int = CARD_W) -> List[Tuple[str, int, int, int]]:
    """포트 칩의 (이름, x, y, 너비). 카드 폭을 균등 분할한다."""
    names = list(ports)
    if not names:
        return []
    w = card_w // len(names)
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


@dataclass
class Shown:
    """캔버스가 실제로 그리는 것 하나.

    보통은 노드 하나지만, **접힌 Procedure면 그 안의 여러 노드를 대표하는 상자**다.
    게이트는 언제나 펼쳐진 그래프를 보고, 접기는 보는 방식일 뿐이다.
    """

    id: str
    ref: str
    kind: NodeKind
    category: str
    lane: int
    inputs: Dict[str, Any] = field(default_factory=dict)   # 이름 -> PortType
    outputs: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""       # 설정값 요약. 카드 본문이 아니라 툴팁으로 간다
    label: str = ""         # 카드에 크게 뜨는 한국어 이름
    in_docs: Dict[str, str] = field(default_factory=dict)   # 포트 이름 -> 뜻
    out_docs: Dict[str, str] = field(default_factory=dict)
    hint: str = ""          # 카드 본문 한 줄. 파라미터가 아니라 무슨 일을 하는지
    inner: int = 0          # 0이면 보통 노드, 1 이상이면 접힌 Procedure
    state_ids: List[str] = field(default_factory=list)      # 상태를 합쳐 볼 노드들


def _first_port(ref: Any) -> str:
    """노출 입력은 안쪽 여러 포트로 갈라질 수 있다. 타입은 어느 쪽이든 같으므로 첫 자리에서 읽는다."""
    return ref[0] if isinstance(ref, list) else ref


def _port_doc(s: "Shown", kind: str, name: str) -> str:
    table = s.in_docs if kind == "입력" else s.out_docs
    return table.get(name, "")


def _proc_of(cg: CompiledGraph, nid: str) -> str:
    """이 노드가 어느 Procedure 안에 있나. 접힌 상자 자신은 `cg.nodes`에 없다."""
    n = cg.nodes.get(nid)
    return (n.origin or "") if n is not None else ""


def fold(cg: CompiledGraph, expanded: Any = ()) -> Tuple[Dict[str, Shown], List[Tuple[str, str]]]:
    """(그릴 것, 배선). `expanded`에 든 Procedure만 펼쳐서 그린다.

    Procedure 안의 배선은 상자 안으로 사라지고, 밖으로 드나드는 배선은 노출 포트로 옮겨 붙는다.
    """
    expanded = set(expanded or ())
    procs = {p["id"]: p for p in cg.procedures if p["id"] not in expanded}

    # 안쪽 노드 -> 대표 상자
    rep: Dict[str, str] = {}
    for nid in cg.order:
        o = _proc_of(cg, nid)
        if o in procs:
            rep[nid] = o

    shown: Dict[str, Shown] = {}
    for nid in cg.order:
        if nid in rep:
            continue
        n = cg.nodes[nid]
        d = resolve_node(n.ref)
        shown[nid] = Shown(
            id=nid,
            ref=n.ref,
            kind=n.kind,
            category=n.category,
            lane=n.lane,
            inputs={p: n.input_types.get(p) or d.inputs[p].type for p in d.inputs},
            outputs=dict(n.output_types),
            summary=_summary(cg, nid),
            label=d.doc.label or n.ref.split("@")[0],
            hint=d.doc.hint,
            in_docs={p: port.doc for p, port in d.inputs.items()},
            out_docs={p: port.doc for p, port in d.outputs.items()},
            state_ids=[nid],
        )

    for pid, p in procs.items():
        inner = [i for i in cg.order if rep.get(i) == pid]
        if not inner:
            continue
        # 노출 포트의 **타입과 설명**을 안쪽 노드에서 그대로 들고 나온다.
        # 설명이 없으면 카드에 `schema`, `fields` 같은 날것이 뜬다.
        ins: Dict[str, Any] = {}
        in_docs: Dict[str, str] = {}
        for name, ref in (p.get("exposed_inputs") or {}).items():
            inode, iport = _first_port(ref).split(":", 1)
            full = f"{pid}/{inode}"
            if full in cg.nodes:
                d = resolve_node(cg.nodes[full].ref)
                ins[name] = cg.nodes[full].input_types.get(iport) or d.inputs[iport].type
                in_docs[name] = d.inputs[iport].doc or name
        outs: Dict[str, Any] = {}
        out_docs: Dict[str, str] = {}
        for name, ref in (p.get("exposed_outputs") or {}).items():
            onode, oport = ref.split(":", 1)
            full = f"{pid}/{onode}"
            if full in cg.nodes:
                outs[name] = cg.nodes[full].output_types.get(oport)
                out_docs[name] = resolve_node(cg.nodes[full].ref).outputs[oport].doc or name
        params = p.get("params") or {}
        shown[pid] = Shown(
            id=pid,
            ref=p.get("ref", ""),
            kind=NodeKind.PROCESSING,
            category="Procedure",
            lane=min(cg.nodes[i].lane for i in inner),
            inputs=ins,
            outputs={k: v for k, v in outs.items() if v is not None},
            summary=", ".join(f"{k}={v}" for k, v in list(params.items())[:3]),
            label=p.get("label") or p.get("ref", "").split("@")[0],
            hint=p.get("hint", ""),
            in_docs=in_docs,
            out_docs=out_docs,
            inner=len(inner),
            state_ids=inner,
        )

    # 배선을 대표 상자의 노출 포트로 옮긴다
    port_of_in: Dict[str, Tuple[str, str]] = {}
    port_of_out: Dict[str, Tuple[str, str]] = {}
    for pid, p in procs.items():
        for name, ref in (p.get("exposed_inputs") or {}).items():
            for one in (ref if isinstance(ref, list) else [ref]):
                inode, iport = one.split(":", 1)
                port_of_in[f"{pid}/{inode}:{iport}"] = (pid, name)
        for name, ref in (p.get("exposed_outputs") or {}).items():
            onode, oport = ref.split(":", 1)
            port_of_out[f"{pid}/{onode}:{oport}"] = (pid, name)

    edges: List[Tuple[str, str]] = []
    seen = set()
    for e in cg.edges:
        if rep.get(e.src_node) and rep.get(e.src_node) == rep.get(e.dst_node):
            continue  # 상자 안에서 끝나는 배선
        src = e.src
        if e.src_node in rep:
            hit = port_of_out.get(e.src)
            if hit is None:
                continue  # 노출되지 않은 내부 출력 — 상자 밖으로 나가지 않는다
            src = f"{hit[0]}:{hit[1]}"
        dst = e.dst
        if e.dst_node in rep:
            hit = port_of_in.get(e.dst)
            if hit is None:
                continue
            dst = f"{hit[0]}:{hit[1]}"
        if (src, dst) not in seen:
            seen.add((src, dst))
            edges.append((src, dst))

    # 접고 나면 안쪽 노드만 있던 레인이 통째로 빈다. 빈 줄을 남기면 상자 아홉 개가
    # 열한 줄에 걸쳐 늘어져 한눈에 안 들어온다. 순서는 지키고 번호만 촘촘히 다시 매긴다.
    used = sorted({v.lane for v in shown.values()})
    dense = {old: i for i, old in enumerate(used)}
    for v in shown.values():
        v.lane = dense[v.lane]
    return shown, edges


def state_of(report: Any, nid: str) -> Tuple[str, str]:
    """(상태 이름, 부가 라벨). 실행 보고가 없으면 pending."""
    if report is None:
        return "pending", ""

    # 지금 이 노드가 돌고 있으면 누계보다 그 사실이 먼저다. 카운트가 아직 0이어도
    # (첫 샘플의 첫 통과) 화면에는 여기가 현재 위치라고 나와야 한다.
    #
    # 두 숫자를 같이 낸다. 앞은 **이번에 들어가서 흐른 시간**이고 뒤는 지금까지의 누계다.
    # 앞만 있으면 이 노드가 전체에서 얼마나 무거운지 모르고, 뒤만 있으면 지금 한 건이
    # 유난히 오래 걸리는 중인지 알 수 없다.
    if getattr(report, "active", "") == nid:
        extra = _elapsed(report.active_since)
        total = report.node_ms.get(nid, 0.0)
        if total >= 1.0:
            extra += f" (누계 {_ms(total)})"
        return "running", extra

    st = report.states_of(nid)
    if not st:
        return "pending", ""
    for name in ("failed", "partial", "running", "cached", "success", "skipped"):
        if st.get(name):
            extra = f"{st[name]}건"
            ms = report.node_ms.get(nid, 0.0)
            # 걸린 시간은 성공했을 때만 쓸모 있는 값이 아니다. 실패한 노드도
            # 5ms 만에 터진 것과 40초를 쓰고 터진 것은 원인이 다르다.
            if ms and name in ("success", "failed", "partial"):
                extra += f" · {_ms(ms)}"
            return name, extra
    return "pending", ""



def _ms(value: float) -> str:
    """`core.humanize.ms` 의 다른 이름. 이 모듈 안의 호출부를 그대로 두기 위한 것이다."""
    return humanize.ms(value)


def _elapsed(since: float) -> str:
    import time as _time

    return _ms(max(0.0, (_time.perf_counter() - since) * 1000))


def state_of_many(report: Any, ids: List[str]) -> Tuple[str, str]:
    """접힌 상자의 상태. 안에서 하나라도 실패했으면 상자가 실패다 —
    상자가 초록인데 안이 빨간 것이 제일 나쁜 화면이다."""
    if not ids:
        return "pending", ""
    if len(ids) == 1:
        return state_of(report, ids[0])
    seen = [state_of(report, i) for i in ids]
    # running이 failed보다 앞에 온다. 실행 중에 "지금 어디"를 묻는 화면이기 때문이다 —
    # 지나간 실패는 격리 패널과 실행이 끝난 뒤의 카드 색이 계속 들고 있다.
    for name in ("running", "failed", "partial", "skipped", "cached", "success"):
        hit = [e for s, e in seen if s == name]
        if hit:
            return name, (hit[0] if len(hit) == len(seen) else f"{len(hit)}/{len(seen)} 노드")
    return "pending", ""


