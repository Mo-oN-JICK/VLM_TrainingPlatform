"""추론 계약 — 학습과 추론의 전처리가 어긋날 수 없게 묶는다.

학습에서 정의한 스키마와 전처리를 추론에서 손으로 다시 짜면 그 즉시 어긋난다.
그래서 학습 그래프에서 **정답 경로를 잘라내고 프롬프트 경로만 남긴 서브그래프**를
자동으로 추출해 체크포인트 옆에 함께 남긴다. 추론도 같은 노드 타입, 같은 타입 검사,
같은 Adapter 체인을 쓴다. 설계 문서 06 §6.5.
"""

from __future__ import annotations

import json
import os
from hashlib import blake2b
from typing import Any, Dict, Optional, Set

from ..core.compiler import CompiledGraph
from ..core.errors import PolicyError, StructureError
from ..core.node import NodeKind
from ..core.registry import resolve as resolve_node
from ..engine.runner import ancestors
from ..spec.loader import SPEC_VERSION

LEAK_TAGS = frozenset({"answer", "label"})
PROMPT_EXPORT = "io.prompt_export@1.0.0"


# ── 프롬프트 경로 찾기 ───────────────────────────────────────────────────


def sample_node(cg: CompiledGraph) -> str:
    """Trainer로 들어가는 Sample을 만드는 노드."""
    for nid in cg.order:
        n = cg.nodes[nid]
        if n.kind is NodeKind.OUTPUT and n.ref.startswith("train."):
            src = n.inputs.get("sample")
            if src:
                return src.split(":")[0]
    for nid in cg.order:
        if cg.nodes[nid].ref.startswith("sample.assemble"):
            return nid
    raise StructureError("Sample을 만드는 노드를 찾을 수 없다. 추론 그래프를 추출할 수 없다.")


def prompt_terminus(cg: CompiledGraph) -> str:
    """최종 프롬프트를 내는 노드.

    누설 차단 노드(taint를 지우는 노드)는 정답을 입력으로 받으므로 추론에는 없다.
    그 경우 한 단계 위의 prompt 입력을 따라 올라간다.
    """
    src = cg.nodes[sample_node(cg)].inputs.get("prompt")
    if not src:
        raise StructureError("sample.assemble의 prompt 입력이 연결되어 있지 않다.")
    nid = src.split(":")[0]
    seen: Set[str] = set()
    while nid not in seen:
        seen.add(nid)
        d = resolve_node(cg.nodes[nid].ref)
        if not d.clears_taint:
            return nid
        up = cg.nodes[nid].inputs.get("prompt")
        if not up:
            return nid
        nid = up.split(":")[0]
    return nid


def images_terminus(cg: CompiledGraph) -> Optional[str]:
    src = cg.nodes[sample_node(cg)].inputs.get("images")
    return src.split(":")[0] if src else None


def inference_nodes(cg: CompiledGraph) -> Set[str]:
    """추론에 남길 노드 집합. 정답 경로와 Output은 전부 빠진다."""
    keep = ancestors(cg, prompt_terminus(cg))
    img = images_terminus(cg)
    if img:
        keep |= ancestors(cg, img)
    keep = {i for i in keep if cg.nodes[i].kind is not NodeKind.OUTPUT}

    # 잘라낸 결과에 정답/라벨이 남아 있으면 안 된다
    leaked = [
        i
        for i in keep
        for t in cg.nodes[i].output_types.values()
        if set(t.semantic) & LEAK_TAGS
    ]
    if leaked:
        raise PolicyError(
            f"추론 그래프에 정답 경로가 남았다: {sorted(set(leaked))}\n"
            "  추론 시점에는 정답이 존재하지 않으므로 이 그래프는 돌 수 없다."
        )
    return keep


# ── 추론 그래프 스펙 ─────────────────────────────────────────────────────


def extract_inference_graph(cg: CompiledGraph, out_id: str = "n_prompt_out") -> Dict[str, Any]:
    """학습 그래프에서 프롬프트 경로만 남긴 Project 스펙(평면화)."""
    keep = inference_nodes(cg)
    p_term, i_term = prompt_terminus(cg), images_terminus(cg)

    nodes = [
        {"id": n.id, "type": n.ref, "params": n.params}
        for n in (cg.nodes[i] for i in cg.order)
        if n.id in keep
    ]
    edges = [
        {"from": e.src, "to": e.dst}
        for e in cg.edges
        if e.src_node in keep and e.dst_node in keep
    ]

    # 추론 그래프도 Output으로 끝나야 컴파일을 통과한다.
    nodes.append({"id": out_id, "type": PROMPT_EXPORT, "params": {"out_dir": "runs/{run_id}/infer"}})
    edges.append({"from": f"{p_term}:text", "to": f"{out_id}:prompt"})
    if i_term:
        port = next(iter(cg.nodes[i_term].output_types))
        edges.append({"from": f"{i_term}:{port}", "to": f"{out_id}:images"})

    return {
        "spec_version": SPEC_VERSION,
        "kind": "Project",
        "id": f"{cg.id}__infer",
        "name": f"{cg.name} (추론)",
        "sample_space": cg.sample_space,
        "nodes": nodes,
        "edges": edges,
        "materialize": {"boundary": [p_term]},
        "runtime_profile": cg.runtime_profile,
    }


# ── 계약 문서 ────────────────────────────────────────────────────────────


def _sha(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return blake2b(fh.read(), digest_size=16).hexdigest()
    except OSError:
        return "missing"


def build_contract(
    cg: CompiledGraph,
    cfg: Any,
    spec_dir: str,
    *,
    vision_tokens: int = 0,
    schema_id: str = "",
) -> Dict[str, Any]:
    keep = inference_nodes(cg)
    p_term = prompt_terminus(cg)

    image_pipeline = [
        {"type": cg.nodes[i].ref, "params": cg.nodes[i].params}
        for i in cg.order
        if i in keep and cg.nodes[i].ref.startswith(("adapt.", "image.", "ts.plot", "list."))
    ]
    slots = next(
        (cg.nodes[i].params for i in keep if cg.nodes[i].ref.startswith("prompt.image_slots")), {}
    )
    template = next(
        (cg.nodes[i].params for i in keep if cg.nodes[i].ref.startswith("text.template")), {}
    )
    knowledge = next(
        (cg.nodes[i].params for i in keep if cg.nodes[i].ref.startswith("source.text_asset")), {}
    )
    schema_node = next(
        (cg.nodes[i].params for i in cg.order if cg.nodes[i].ref.startswith("schema.define")), {}
    )

    kb_path = os.path.normpath(os.path.join(spec_dir, knowledge.get("path", ""))) if knowledge else ""
    schema_path = (
        os.path.normpath(os.path.join(spec_dir, schema_node.get("path", ""))) if schema_node else ""
    )

    return {
        "spec_hash": cg.spec_hash,
        "prompt_terminus": p_term,
        "backbone": {
            "id": getattr(cfg, "backbone", ""),
            "dtype": getattr(cfg, "dtype", ""),
            "quantization": getattr(getattr(cfg, "quantization", None), "mode", "none"),
        },
        "image_pipeline": image_pipeline,
        "image_slots": {
            "placeholder": slots.get("placeholder", "<image>"),
            "policy": slots.get("policy", "prepend"),
        },
        "prompt_template": template.get("template", ""),
        "knowledge_block": {"source": knowledge.get("path", ""), "sha": _sha(kb_path) if kb_path else ""},
        "answer_schema": {"id": schema_id, "path": schema_node.get("path", ""), "sha": _sha(schema_path) if schema_path else ""},
        "sequence": {
            "max_len": getattr(getattr(cfg, "sequence", None), "max_len", 0),
            "truncation": getattr(getattr(cfg, "sequence", None), "truncation", "forbid"),
        },
        "expected_vision_tokens_per_sample": vision_tokens,
        "parser": {"kind": "generated_from_schema", "stop": ["</verdict>"]},
    }


def write(
    out_dir: str,
    cg: CompiledGraph,
    cfg: Any,
    spec_dir: str,
    *,
    vision_tokens: int = 0,
    schema_id: str = "",
) -> Dict[str, str]:
    """체크포인트 옆에 계약과 추론 그래프를 함께 남긴다."""
    import yaml

    os.makedirs(out_dir, exist_ok=True)
    contract = build_contract(cg, cfg, spec_dir, vision_tokens=vision_tokens, schema_id=schema_id)
    graph = extract_inference_graph(cg)

    cpath = os.path.join(out_dir, "inference_contract.json")
    gpath = os.path.join(out_dir, "inference_graph.yaml")
    with open(cpath, "w", encoding="utf-8") as fh:
        json.dump(contract, fh, ensure_ascii=False, indent=2)
    with open(gpath, "w", encoding="utf-8") as fh:
        yaml.safe_dump(graph, fh, allow_unicode=True, sort_keys=False, width=100)
    return {"contract": cpath, "graph": gpath}
