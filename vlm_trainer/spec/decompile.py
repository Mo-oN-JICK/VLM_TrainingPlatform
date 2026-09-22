"""CompiledGraph -> Project Spec.

왕복 보장: canonical(compile(decompile(compile(s)))) == canonical(compile(s)).
의미는 보존되고 주석·좌표는 보존되지 않는다. 좌표는 layout.json의 몫이다.
"""

from __future__ import annotations

from typing import Any, Dict, List

import yaml

from ..core.compiler import CompiledGraph
from .loader import SPEC_VERSION


def _rev(exposed: Dict[str, str], proc_id: str) -> Dict[str, str]:
    """{"n_topk:regions": "regions"} 형태의 역맵. 키는 인라인된 전체 id 기준."""
    return {f"{proc_id}/{v}": k for k, v in exposed.items()}


def decompile(cg: CompiledGraph, *, keep_procedures: bool = True) -> Dict[str, Any]:
    node_origin = {n.id: n.origin for n in cg.nodes.values()}
    hidden: set = set()
    out_map: Dict[str, str] = {}  # "pid/inner:port" -> "pid:exposed"
    in_map: Dict[str, str] = {}
    procs: List[Dict[str, Any]] = []

    if keep_procedures:
        for p in cg.procedures:
            pid = p["id"]
            hidden |= {i for i, o in node_origin.items() if o == pid}
            for full, name in _rev(p["exposed_outputs"], pid).items():
                out_map[full] = f"{pid}:{name}"
            for full, name in _rev(p["exposed_inputs"], pid).items():
                in_map[full] = f"{pid}:{name}"
            procs.append({"id": pid, "ref": p["ref"], "params": dict(p["params"])})

    nodes = [
        {"id": n.id, "type": n.ref, "params": n.params}
        for n in (cg.nodes[i] for i in cg.order)
        if n.id not in hidden
    ]

    edges: List[Dict[str, str]] = []
    for e in cg.edges:
        src_hidden, dst_hidden = e.src_node in hidden, e.dst_node in hidden
        if src_hidden and dst_hidden:
            continue  # Procedure 내부 배선은 Procedure 안에 있다
        src = out_map.get(e.src, e.src) if src_hidden else e.src
        dst = in_map.get(e.dst, e.dst) if dst_hidden else e.dst
        edges.append({"from": src, "to": dst})

    spec: Dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "kind": "Project",
        "id": cg.id,
        "name": cg.name,
        "sample_space": cg.sample_space,
        "nodes": nodes,
        "edges": edges,
        "materialize": cg.materialize,
        "runtime_profile": cg.runtime_profile,
    }
    if cg.defaults:
        spec["defaults"] = cg.defaults
    if procs:
        spec["procedures"] = procs
    if cg.debug:
        spec["debug"] = cg.debug
    return spec


def dump_yaml(spec: Dict[str, Any]) -> str:
    return yaml.safe_dump(spec, allow_unicode=True, sort_keys=False, width=100)
