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
        # **같은 Procedure 안**의 배선만 버린다. 그것은 Procedure 파일에 있으므로
        # 프로젝트가 다시 적을 것이 아니다.
        #
        # 양쪽이 숨겨졌다는 것만으로 버리면 **Procedure 에서 Procedure 로 가는 배선이
        # 사라진다.** 그것은 프로젝트가 그은 선이고 어디에도 다시 적히지 않는다 —
        # 편집기에서 저장 한 번에 그래프가 조용히 끊긴다(실측: `p_prep:images ->
        # p_prompt:images` 가 사라져 저장한 파일이 컴파일되지 않았다).
        if src_hidden and dst_hidden and node_origin[e.src_node] == node_origin[e.dst_node]:
            continue
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
    if cg.node_modules:
        # 이것이 빠지면 저장 한 번에 커스텀 노드 선언이 사라지고, 다음 컴파일이
        # "노드를 찾을 수 없다" 로 막힌다.
        spec["node_modules"] = list(cg.node_modules)
    if procs:
        spec["procedures"] = procs
    if cg.debug:
        spec["debug"] = cg.debug
    return spec


def dump_yaml(spec: Dict[str, Any]) -> str:
    return yaml.safe_dump(spec, allow_unicode=True, sort_keys=False, width=100)
