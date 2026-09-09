"""편집기 API — UI가 코어를 호출하는 유일한 통로.

**UI 전용 실행 경로를 만들지 않는다.** 모든 변경은 GraphModel을 고치고 곧바로 컴파일해
G1/G2를 통과해야만 받아들여진다. 통과하지 못하면 변경 자체가 거부된다.
스펙 파일은 저장할 때만 쓰인다 — 편집 중에는 디스크의 진실이 흔들리지 않는다.
설계 문서 12, 그리고 "UI는 스펙을 편집하는 뷰일 뿐"이라는 원칙.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.compiler import CompiledGraph, compile_graph
from ..core.errors import VlmtError
from ..core.graph import Edge, GraphModel, NodeInstance
from ..core.node import NodeKind
from ..core.registry import all_defs, resolve as resolve_node
from ..core.unify import unify_ports
from ..spec.decompile import decompile, dump_yaml
from ..spec.loader import load_project


def _short(errors: Any) -> str:
    text = str(errors)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else text


def compat_matrix(cg: CompiledGraph) -> Dict[str, Dict[str, str]]:
    """출력 포트마다 어떤 입력 포트에 꽂을 수 있는지. 값이 빈 문자열이면 가능.

    이 표가 있으면 드래그 중에 서버를 다시 부르지 않고도 호환되는 포트만 밝힐 수 있다.
    실제 연결은 그래도 컴파일로 확정한다 — 표는 미리 거르는 장치일 뿐이다.
    """
    out: Dict[str, Dict[str, str]] = {}

    for snid in cg.order:
        sn = cg.nodes[snid]
        for sport, stype in sn.output_types.items():
            row: Dict[str, str] = {}
            for dnid in cg.order:
                if dnid == snid:
                    continue
                dd = resolve_node(cg.nodes[dnid].ref)
                for dport, port in dd.inputs.items():
                    ref = f"{dnid}:{dport}"
                    # 이미 배선이 있는 입력에도 놓을 수 있다 — 그 자리를 **교체**한다.
                    # 팬인은 여전히 금지되지만, 갈아끼우기를 두 단계로 만들면 편집기가 쓸모없어진다.
                    if cg.nodes[dnid].lane <= sn.lane:
                        row[ref] = "위로 향하는 배선은 만들 수 없다"
                        continue
                    res = unify_ports(stype, port.type)
                    row[ref] = "" if res.ok else ", ".join(str(m) for m in res.mismatches[:3])
            out[f"{snid}:{sport}"] = row
    return out


def param_meta(node_ref: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """파라미터마다 위젯을 고를 수 있게 값과 성질을 함께 낸다.

    타입에 영향 주는 파라미터는 표식이 필요하다 — 바꾸면 배선이 다시 검사되기 때문이다.
    """
    d = resolve_node(node_ref)
    out: List[Dict[str, Any]] = []
    for name, value in params.items():
        if isinstance(value, bool):
            kind = "bool"
        elif isinstance(value, (int, float)):
            kind = "number"
        elif isinstance(value, (list, tuple, dict)):
            kind = "json"
        else:
            kind = "text"
        out.append(
            {
                "name": name,
                "value": value,
                "kind": kind,
                "overridable": name in d.recipe_overridable,
                "type_affecting": name in d.type_affecting,
            }
        )
    return out


def occupied_inputs(cg: CompiledGraph) -> List[str]:
    """이미 배선이 있는 입력 포트. 놓으면 교체된다는 것을 UI가 알려주기 위한 것."""
    return sorted({e.dst for e in cg.edges})


@dataclass
class Editor:
    """열려 있는 프로젝트 하나. 편집은 메모리에서, 저장은 명시적으로."""

    path: str
    graph: GraphModel = field(default_factory=GraphModel)
    compiled: Optional[CompiledGraph] = None
    error: str = ""
    dirty: bool = False

    @staticmethod
    def open(path: str) -> "Editor":
        e = Editor(path=os.path.abspath(path))
        e.graph = load_project(e.path)
        e._recompile()
        return e

    # ── 내부 ────────────────────────────────────────────────────────────
    def _recompile(self) -> bool:
        try:
            self.compiled = compile_graph(self.graph)
            self.error = ""
            return True
        except Exception as exc:
            # 노드가 계약을 어기고 일반 예외를 던져도 편집기는 살아 있어야 한다.
            # 컴파일되지 않는 변경은 예외 종류와 무관하게 거부한다.
            self.error = str(exc) if isinstance(exc, VlmtError) else f"{type(exc).__name__}: {exc}"
            return False

    def _try(self, mutate) -> Dict[str, Any]:
        """변경을 적용해 보고, 게이트를 통과하지 못하면 되돌린다."""
        before_nodes = [NodeInstance(n.id, n.type, dict(n.params)) for n in self.graph.nodes]
        before_edges = list(self.graph.edges)
        before_compiled, before_error = self.compiled, self.error
        try:
            mutate()
        except Exception as exc:
            return {"ok": False, "reason": _short(exc), "detail": str(exc)}
        if self._recompile():
            self.dirty = True
            return {"ok": True}
        reason, detail = _short(self.error), self.error
        self.graph.nodes, self.graph.edges = before_nodes, before_edges
        self.compiled, self.error = before_compiled, before_error
        return {"ok": False, "reason": reason, "detail": detail}

    # ── 변경 ────────────────────────────────────────────────────────────
    def connect(self, src: str, dst: str) -> Dict[str, Any]:
        """입력 포트 하나에 배선은 하나. 이미 있으면 원자적으로 교체한다.

        거부되면 원래 배선이 그대로 남는다 — 갈아끼우다 실패해서 그래프가 깨지지 않는다.
        """
        replaced = next((e.src for e in self.graph.edges if e.dst == dst), "")

        def go() -> None:
            self.graph.edges = [e for e in self.graph.edges if e.dst != dst]
            self.graph.edges.append(Edge.parse(src, dst))

        res = self._try(go)
        if res["ok"] and replaced:
            res["replaced"] = replaced
        return res

    def disconnect(self, dst: str) -> Dict[str, Any]:
        def go() -> None:
            self.graph.edges = [e for e in self.graph.edges if e.dst != dst]

        return self._try(go)

    def set_param(self, node_id: str, param: str, value: Any) -> Dict[str, Any]:
        def go() -> None:
            n = self.graph.node(node_id)
            n.params[param] = value

        return self._try(go)

    def add_node(self, node_type: str, node_id: str = "") -> Dict[str, Any]:
        d = resolve_node(node_type)
        base = node_id or "n_" + d.type.split(".")[-1]
        nid, i = base, 1
        existing = set(self.graph.ids)
        while nid in existing:
            i += 1
            nid = f"{base}_{i}"

        def go() -> None:
            self.graph.nodes.append(NodeInstance(nid, d.ref, dict(d.default_params())))

        res = self._try(go)
        res["id"] = nid
        return res

    def remove_node(self, node_id: str) -> Dict[str, Any]:
        def go() -> None:
            self.graph.nodes = [n for n in self.graph.nodes if n.id != node_id]
            self.graph.edges = [
                e for e in self.graph.edges if e.src_node != node_id and e.dst_node != node_id
            ]

        return self._try(go)

    def save(self) -> Dict[str, Any]:
        if self.compiled is None:
            return {"ok": False, "reason": "컴파일되지 않은 그래프는 저장하지 않는다"}
        spec = decompile(self.compiled, keep_procedures=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(dump_yaml(spec))
        os.replace(tmp, self.path)
        self.dirty = False
        return {"ok": True, "path": self.path}

    # ── 조회 ────────────────────────────────────────────────────────────
    def library(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": d.ref,
                "category": d.category,
                "kind": d.kind.value,
                "summary": d.doc.summary,
                "inputs": list(d.inputs),
                "outputs": list(d.outputs),
            }
            for d in all_defs()
        ]

    def state(self) -> Dict[str, Any]:
        cg = self.compiled
        if cg is None:
            return {"ok": False, "error": self.error, "nodes": [], "edges": []}
        return {
            "ok": True,
            "id": cg.id,
            "name": cg.name,
            "spec_hash": cg.spec_hash,
            "dirty": self.dirty,
            "error": self.error,
            "nodes": [
                {
                    "id": nid,
                    "type": cg.nodes[nid].ref,
                    "kind": cg.nodes[nid].kind.value,
                    "category": cg.nodes[nid].category,
                    "lane": cg.nodes[nid].lane,
                    "params": cg.nodes[nid].params,
                    "param_meta": param_meta(cg.nodes[nid].ref, cg.nodes[nid].params),
                    "inputs": {p: str(t) for p, t in cg.nodes[nid].input_types.items()},
                    "outputs": {p: str(t) for p, t in cg.nodes[nid].output_types.items()},
                    "wired": cg.nodes[nid].inputs,
                }
                for nid in cg.order
            ],
            "edges": [{"from": e.src, "to": e.dst} for e in cg.edges],
            "compat": compat_matrix(cg),
            "occupied": occupied_inputs(cg),
        }
