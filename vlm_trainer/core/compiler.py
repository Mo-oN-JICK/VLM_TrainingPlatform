"""컴파일러 — G1/G2를 모아 실행 계획을 만든다.

게이트 우회 옵션은 여기에 절대 추가하지 않는다. 설계 문서 04 §4.4.

파이프라인: parse -> resolve -> overlay -> inline -> typecheck -> structure
            -> policy -> order -> emit
순환 검사 단계는 없다. 캔버스가 상단 입력 -> 하단 출력만 허용해 순환을 만들 수 없기 때문이다.
다만 손으로 쓴 YAML은 그 보호를 받지 않으므로, 위상 정렬 후 남은 노드를 구조 오류로 보고한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..spec import canonical
from ..spec.loader import ProcedureDef, find_procedure, load_project
from .errors import (
    GateError,
    PolicyError,
    SpecError,
    StructureError,
    TypeMismatchError,
    UnresolvedTypeError,
    render_gate_report,
)
from .graph import Edge, GraphModel, NodeInstance
from .node import NodeDef, NodeKind
from .registry import resolve as resolve_node
from .types import PortType
from .unify import apply_subst, rename_vars, unify_ports

ENGINE_ABI = "vlmt-abi-1"

# 정답/라벨이 프롬프트로 새어들어가는 것을 막는 taint 태그
LEAK_TAGS = frozenset({"label", "answer"})
PROMPT_TAG = "prompt"


class CompileFailed(GateError):
    """여러 게이트 위반을 한 번에 보고한다."""

    def __init__(self, errors: Sequence[GateError]) -> None:
        self.errors = list(errors)
        self.gate = self.errors[0].gate if self.errors else "G2"
        super().__init__(render_gate_report(self.errors))


@dataclass
class CompiledNode:
    id: str
    ref: str
    kind: NodeKind
    category: str
    params: Dict[str, Any]
    inputs: Dict[str, str] = field(default_factory=dict)  # 포트 -> "노드:포트"
    input_types: Dict[str, PortType] = field(default_factory=dict)
    output_types: Dict[str, PortType] = field(default_factory=dict)
    cache_key: str = ""
    lane: int = 0
    external_call: bool = False
    deterministic: bool = True
    origin: str = ""  # 인라인된 Procedure 인스턴스 id
    taint: frozenset = frozenset()


@dataclass
class CompiledGraph:
    id: str
    name: str
    nodes: Dict[str, CompiledNode]
    edges: List[Edge]
    order: List[str]
    lanes: Dict[str, int]
    sample_space: Dict[str, Any]
    materialize: Dict[str, Any]
    runtime_profile: str
    lock: Dict[str, Any]
    spec_hash: str = ""
    procedures: List[Dict[str, Any]] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    def by_kind(self, kind: NodeKind) -> List[CompiledNode]:
        return [self.nodes[i] for i in self.order if self.nodes[i].kind is kind]


# ── 4) inline ───────────────────────────────────────────────────────────


def _inline_procedures(
    g: GraphModel,
) -> Tuple[List[NodeInstance], List[Edge], Dict[str, str], List[Dict[str, Any]]]:
    """Procedure 인스턴스를 펼치고 id에 네임스페이스를 붙인다.

    중첩 Procedure는 아직 지원하지 않는다(Procedure 파일에 procedures 필드를 두지 않음).
    """
    nodes: List[NodeInstance] = [
        NodeInstance(n.id, n.type, dict(n.params)) for n in g.nodes
    ]
    edges: List[Edge] = list(g.edges)
    origin: Dict[str, str] = {}
    procs: List[Dict[str, Any]] = []

    for p in g.procedures:
        pd: ProcedureDef = find_procedure(p.ref, g.source_dir)
        inner: Dict[str, NodeInstance] = {}
        for n in pd.nodes:
            nid = f"{p.id}/{n.id}"
            inner[n.id] = NodeInstance(nid, n.type, dict(n.params))
            origin[nid] = p.id

        # 노출 파라미터 적용 — 노출되지 않은 내부 파라미터는 건드릴 수 없다(캡슐화)
        for key, val in (p.params or {}).items():
            if key not in pd.exposed_params:
                raise SpecError(
                    f"Procedure {p.id}({pd.ref}): 노출되지 않은 파라미터 {key!r} "
                    f"(노출된 것: {sorted(pd.exposed_params)})",
                    node=p.id,
                )
            tgt_node, tgt_param = pd.exposed_params[key]
            if tgt_node not in inner:
                raise SpecError(f"Procedure {pd.ref}: exposed_params가 없는 노드 {tgt_node!r}를 가리킨다")
            inner[tgt_node].params[tgt_param] = val

        nodes.extend(inner.values())
        for e in pd.edges:
            edges.append(
                Edge(f"{p.id}/{e.src_node}", e.src_port, f"{p.id}/{e.dst_node}", e.dst_port)
            )

        # 바깥 배선을 내부 포트로 다시 꽂는다
        def _targets(ref_node: str, ref_port: str, exposed: Dict[str, Any], what: str):
            """(노드, 포트) 목록. 입력은 안쪽 여러 포트로 갈라질 수 있다."""
            if ref_node != p.id:
                return [(ref_node, ref_port)]
            if ref_port not in exposed:
                raise SpecError(
                    f"Procedure {p.id}({pd.ref}): 노출되지 않은 {what} 포트 {ref_port!r} "
                    f"(노출된 것: {sorted(exposed)})",
                    node=p.id,
                    port=f"{p.id}:{ref_port}",
                )
            tgt = exposed[ref_port]
            out = []
            for one in tgt if isinstance(tgt, list) else [tgt]:
                tn, tp = one.rsplit(":", 1)
                out.append((f"{p.id}/{tn}", tp))
            return out

        rewired: List[Edge] = []
        for e in edges:
            srcs = _targets(e.src_node, e.src_port, pd.exposed_outputs, "출력")
            dsts = _targets(e.dst_node, e.dst_port, pd.exposed_inputs, "입력")
            for sn, sp in srcs:
                for dn, dp in dsts:
                    rewired.append(Edge(sn, sp, dn, dp))
        edges = rewired

        procs.append(
            {
                "id": p.id,
                "ref": pd.ref,
                "params": dict(p.params or {}),
                "exposed_inputs": dict(pd.exposed_inputs),
                "exposed_outputs": dict(pd.exposed_outputs),
                "exposed_params": {k: (f"{p.id}/{n}", prm) for k, (n, prm) in pd.exposed_params.items()},
            }
        )

    return nodes, edges, origin, procs


# ── 8) order ────────────────────────────────────────────────────────────


def _topo(node_ids: List[str], edges: List[Edge]) -> Tuple[List[str], List[str]]:
    indeg = {i: 0 for i in node_ids}
    adj: Dict[str, List[str]] = {i: [] for i in node_ids}
    for e in edges:
        if e.src_node in indeg and e.dst_node in indeg:
            adj[e.src_node].append(e.dst_node)
            indeg[e.dst_node] += 1
    ready = sorted(i for i in node_ids if indeg[i] == 0)
    order: List[str] = []
    while ready:
        cur = ready.pop(0)
        order.append(cur)
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort()
    leftover = [i for i in node_ids if i not in set(order)]
    return order, leftover


def _lanes(order: List[str], edges: List[Edge], kinds: Dict[str, NodeKind]) -> Dict[str, int]:
    depth = {i: 0 for i in order}
    for i in order:
        for e in edges:
            if e.dst_node == i and e.src_node in depth:
                depth[i] = max(depth[i], depth[e.src_node] + 1)
    max_d = max(depth.values(), default=0)
    for i, k in kinds.items():
        if k is NodeKind.INPUT:
            depth[i] = 0
        elif k is NodeKind.OUTPUT:
            depth[i] = max_d + 1
    return depth


# ── 컴파일 본체 ─────────────────────────────────────────────────────────


def compile_graph(
    g: GraphModel,
    *,
    recipe_overrides: Optional[Dict[str, Any]] = None,
    draft: bool = False,
) -> CompiledGraph:
    """draft=True는 **편집 중의 미완성 그래프**만을 위한 것이다.

    타입 검사(G1)와 정책 검사는 그대로 돈다. 완결성 검사(미연결 필수 포트, 도달 불가 노드,
    Output 없음, 미해결 제네릭)만 미룬다 — 노드를 놓고 배선을 잇는 사이의 상태를 허용하기 위해서다.
    저장과 실행은 언제나 strict 경로를 지난다. 게이트 우회로가 아니다.
    """
    errors: List[GateError] = []

    # 2) resolve + 4) inline
    nodes, edges, origin, procs = _inline_procedures(g)
    defs: Dict[str, NodeDef] = {}
    for n in nodes:
        defs[n.id] = resolve_node(n.type)

    # 3) overlay — 값만, 화이트리스트 안에서만
    if recipe_overrides:
        _apply_overrides(nodes, defs, recipe_overrides, procs)

    ids = [n.id for n in nodes]
    id_set = set(ids)
    inst = {n.id: n for n in nodes}

    # 배선이 가리키는 노드/포트가 실재하는지
    for e in edges:
        for nid, port, side, table in (
            (e.src_node, e.src_port, "출력", "outputs"),
            (e.dst_node, e.dst_port, "입력", "inputs"),
        ):
            if nid not in id_set:
                errors.append(StructureError(f"배선이 없는 노드를 가리킨다: {nid!r}", node=nid))
            elif port not in getattr(defs[nid], table):
                avail = sorted(getattr(defs[nid], table))
                errors.append(
                    StructureError(
                        f"{nid}({defs[nid].ref})에 {side} 포트 {port!r}가 없다 (있는 포트: {avail})",
                        node=nid,
                        port=f"{nid}:{port}",
                    )
                )
    if errors:
        raise CompileFailed(errors)

    # 팬인 금지
    seen: Dict[str, str] = {}
    for e in edges:
        if e.dst in seen:
            errors.append(
                StructureError(
                    f"입력 포트 {e.dst}에 배선이 둘 이상이다 ({seen[e.dst]}, {e.src}). "
                    "팬인은 금지된다 — 병합 노드를 명시하라.",
                    node=e.dst_node,
                    port=e.dst,
                )
            )
        seen[e.dst] = e.src

    order, leftover = _topo(ids, edges)
    if leftover:
        errors.append(
            StructureError(
                f"위상 정렬에서 남은 노드 {sorted(leftover)}. "
                "상단 입력·하단 출력 규약을 벗어난 배선(순환)이다 — 캔버스에서는 만들 수 없는 형태다."
            )
        )
        raise CompileFailed(errors)

    # 5) typecheck
    compiled: Dict[str, CompiledNode] = {}
    incoming: Dict[str, List[Edge]] = {i: [] for i in ids}
    for e in edges:
        incoming[e.dst_node].append(e)

    for nid in order:
        d = defs[nid]
        cn = CompiledNode(
            id=nid,
            ref=d.ref,
            kind=d.kind,
            category=d.category,
            params=canonical.canon_params(d.default_params(), inst[nid].params),
            external_call=d.external_call,
            deterministic=d.deterministic,
            origin=origin.get(nid, ""),
        )
        subst: Dict[str, Any] = {}
        declared_in = {p: rename_vars(port.type, nid) for p, port in d.inputs.items()}
        declared_out = {p: rename_vars(port.type, nid) for p, port in d.outputs.items()}

        taint: Set[str] = set()
        for e in incoming[nid]:
            up = compiled[e.src_node]
            src_t = up.output_types[e.src_port]
            dst_t = declared_in[e.dst_port]
            res = unify_ports(src_t, dst_t, subst)
            if not res.ok:
                errors.append(
                    TypeMismatchError(e.src, e.dst, src_t, dst_t, res.mismatches, gate="G1")
                )
            else:
                subst = res.subst
            cn.inputs[e.dst_port] = e.src
            cn.input_types[e.dst_port] = src_t
            taint |= set(up.taint)

        # 6) 미연결 필수 입력
        for port, spec_port in ({} if draft else d.inputs).items():
            if port not in cn.inputs and not spec_port.optional:
                errors.append(
                    StructureError(
                        f"{nid}({d.ref})의 필수 입력 포트 {port!r}가 연결되지 않았다 "
                        f"(기대 타입 {spec_port.type})",
                        node=nid,
                        port=f"{nid}:{port}",
                    )
                )

        # 출력 타입 확정
        impl = d.impl() if d.impl else None
        declared_raw = {p: port.type for p, port in d.outputs.items()}
        if impl is not None:
            try:
                outs = impl.infer_types(
                    {p: apply_subst(t, subst) for p, t in cn.input_types.items()},
                    d.build_params(cn.params),
                )
            except Exception:
                # 아직 아무것도 물리지 않은 노드만 선언 타입으로 둔다.
                # 입력이 이미 이어져 있다면 그것은 미완성이 아니라 진짜 오류다.
                if not draft or cn.inputs:
                    raise
                outs = dict(declared_raw)
        else:
            outs = dict(declared_raw)

        resolved_out: Dict[str, PortType] = {}
        for p, t in outs.items():
            # 노드가 선언 타입을 그대로 돌려줬다면, 변수에 노드 접두가 붙은 판본을 쓴다
            if p in declared_raw and t == declared_raw[p]:
                t = declared_out[p]
            resolved_out[p] = apply_subst(t, subst)
        cn.output_types = resolved_out

        # taint 전파: 상류 taint + 자기 출력 태그
        for t in resolved_out.values():
            taint |= set(t.semantic) & LEAK_TAGS
        cn.taint = frozenset(taint - set(d.clears_taint))

        # 7) 누설 정책 — label/answer가 prompt 생산 노드에 도달했는가
        produces_prompt = any(PROMPT_TAG in t.semantic for t in resolved_out.values())
        leaked = set(taint) & LEAK_TAGS
        if produces_prompt and leaked and not set(d.clears_taint) >= leaked:
            errors.append(
                PolicyError(
                    f"정답 누설: {nid}({d.ref})가 프롬프트를 만드는데 상류에서 "
                    f"{sorted(leaked)} 태그가 도달한다.\n"
                    "  이 검사가 없었다면: 실험은 완주하고 점수는 비현실적으로 높게 나오며, "
                    "그 사실을 아무도 눈치채지 못한다.\n"
                    "  해결: answer.leakage_guard를 거치게 하거나 해당 배선을 끊어라.",
                    node=nid,
                )
            )

        # 미해결 타입 변수
        for p, t in ({} if draft else resolved_out).items():
            if not t.is_ground():
                errors.append(
                    UnresolvedTypeError(
                        f"{nid}:{p}의 타입이 확정되지 않았다: {t} "
                        f"(미해결 변수 {sorted(t.free_vars())}). "
                        "제네릭은 compile 종료 시점에 전부 구체 타입이어야 한다.",
                        node=nid,
                        port=f"{nid}:{p}",
                    )
                )

        compiled[nid] = cn

    # 6) Output 존재 / 도달 불가 노드
    outputs = [i for i in order if defs[i].kind is NodeKind.OUTPUT]
    if draft:
        pass  # 완결성은 저장·실행 직전에 strict로 본다
    elif not outputs:
        errors.append(
            StructureError(
                "그래프에 Output 노드가 없다. 최소 하나의 Output으로 끝나야 한다.\n"
                "  이 검사가 없었다면: 몇 시간 계산 후 아무 산출물 없이 '완료'된다."
            )
        )
    else:
        useful: Set[str] = set()
        stack = list(outputs)
        while stack:
            cur = stack.pop()
            if cur in useful:
                continue
            useful.add(cur)
            stack.extend(e.src_node for e in incoming[cur])
        dead = [i for i in order if i not in useful]
        if dead:
            errors.append(
                StructureError(
                    f"어떤 Output에도 기여하지 않는 노드: {sorted(dead)}\n"
                    "  이 검사가 없었다면: 쓸모없는 전처리에 시간을 쓴다."
                )
            )

    # 7) external_call은 물질화 경계 앞에 있어야 한다
    boundary = [b for b in g.materialize.boundary if b in compiled]
    # **인라인된** 배선을 본다. 바깥 그래프의 edges만 보면 Procedure 안에서
    # 외부 모델을 부르는 노드가 통째로 빠진다.
    wired_all = {e.src_node for e in edges} | {e.dst_node for e in edges}
    externals = [i for i in order if defs[i].external_call and i in wired_all]
    # 학습 루프가 있는 그래프에서만 묻는다. 루프가 없으면 "루프 안에서 돈다"는 위험 자체가 없고,
    # 게이트가 실제로 없는 위험을 말하기 시작하면 게이트를 믿지 않게 된다.
    trains = any(not defs[i].per_sample for i in order)
    if externals and trains and not boundary and not draft:
        # 경계가 비어 있으면 아래 검사가 통째로 건너뛰어진다. 그러면 외부 모델을 부르는
        # 그래프가 아무 말 없이 통과하고, 학습 루프 안에서 전문가 모델이 VRAM을 요구한다.
        # 검사를 끄는 방법이 "경계를 안 적는 것"이어서는 안 된다.
        errors.append(
            PolicyError(
                f"외부 모델을 호출하는 노드가 있는데 물질화 경계가 비어 있다: {externals}\n"
                "  경계가 없으면 그 노드들이 학습 루프 안에서 돈다.\n"
                "  이 검사가 없었다면: 학습 중 전문가 모델이 VRAM을 요구해 OOM 또는 심한 감속이 난다.\n"
                "  materialize.boundary에 미리 구울 마지막 노드를 적어라 "
                "(편집기에서는 노드의 '물질화 경계' 토글)."
            )
        )
    if boundary:
        pre: Set[str] = set()
        stack = list(boundary)
        while stack:
            cur = stack.pop()
            if cur in pre:
                continue
            pre.add(cur)
            stack.extend(e.src_node for e in incoming[cur])
        # 아직 아무 데도 물리지 않은 노드는 경계 앞뒤가 정해지지 않았다.
        # 편집 중에만 판단을 미루고, 배선되는 순간 다시 본다.
        for nid in order:
            if draft and nid not in wired_all:
                continue
            if defs[nid].external_call and nid not in pre:
                errors.append(
                    PolicyError(
                        f"{nid}({defs[nid].ref})는 외부 모델을 호출하는데 물질화 경계 뒤에 있다.\n"
                        "  이 검사가 없었다면: 학습 루프 안에서 전문가 모델이 VRAM을 요구해 "
                        "OOM 또는 심한 감속이 난다.",
                        node=nid,
                    )
                )

    if errors:
        raise CompileFailed(errors)

    # 8) 캐시 키
    for nid in order:
        cn = compiled[nid]
        ups = sorted(compiled[e.src_node].cache_key for e in incoming[nid])
        cn.cache_key = canonical.hash_parts(ENGINE_ABI, cn.ref, cn.params, ups, cn.kind.value)

    lanes = _lanes(order, edges, {i: defs[i].kind for i in order})
    for nid, lane in lanes.items():
        compiled[nid].lane = lane

    cg = CompiledGraph(
        id=g.id,
        name=g.name,
        nodes=compiled,
        edges=edges,
        order=order,
        lanes=lanes,
        sample_space=canonical.canon_value(g.sample_space.__dict__),
        materialize=canonical.canon_value(g.materialize.__dict__),
        runtime_profile=g.runtime_profile,
        lock={compiled[i].id: compiled[i].ref for i in order},
        procedures=procs,
        defaults=canonical.canon_value(g.defaults),
        debug=canonical.canon_value(g.debug),
    )
    cg.spec_hash = canonical.hash_obj(canonical_view(cg))
    return cg


def compile_project(path: str, *, recipe_overrides: Optional[Dict[str, Any]] = None) -> CompiledGraph:
    return compile_graph(load_project(path), recipe_overrides=recipe_overrides)


def _apply_overrides(
    nodes: List[NodeInstance],
    defs: Dict[str, NodeDef],
    overrides: Dict[str, Any],
    procs: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Parameter Recipe 오버레이. 값만, 화이트리스트 안에서만.

    Procedure는 **노출한 파라미터로만** 덮을 수 있다(`p_crop.max_n`). 내부 노드를 직접
    가리키는 형태(`p_crop/n_crop.max_n`)도 받지만, 캡슐화를 지키려면 앞의 형태를 쓴다.
    """
    inst = {n.id: n for n in nodes}
    exposed: Dict[str, Dict[str, tuple]] = {p["id"]: p.get("exposed_params") or {} for p in (procs or [])}

    for path, value in overrides.items():
        if "." not in path:
            raise SpecError(f"오버라이드 경로는 '노드id.파라미터' 형식이어야 한다: {path!r}")
        nid, param = path.split(".", 1)
        if nid in exposed:  # Procedure 인스턴스의 노출 파라미터
            table = exposed[nid]
            if param not in table:
                raise PolicyError(
                    f"Procedure {nid}가 노출하지 않은 파라미터 {param!r}는 레시피가 덮을 수 없다 "
                    f"(노출된 것: {sorted(table)})",
                    node=nid,
                )
            nid, param = table[param]
        if nid not in inst:
            raise SpecError(f"오버라이드가 없는 노드를 가리킨다: {nid!r}")
        d = defs[nid]
        if param not in d.recipe_overridable:
            raise PolicyError(
                f"{nid}({d.ref}): 파라미터 {param!r}는 레시피가 덮어쓸 수 없다 "
                f"(허용: {sorted(d.recipe_overridable) or '없음'})",
                node=nid,
            )
        inst[nid].params[param] = value


def override_target(cg: CompiledGraph, path: str) -> Tuple[str, str]:
    """오버라이드 경로를 실제 (노드 id, 파라미터)로 해소한다.

    Procedure 인스턴스의 노출 파라미터(`p_crop.max_n`)는 내부 노드로 옮겨진다.
    """
    nid, param = path.split(".", 1)
    for p in cg.procedures:
        if p["id"] == nid:
            tgt = (p.get("exposed_params") or {}).get(param)
            if tgt:
                return tgt
    return nid, param


def current_value(cg: CompiledGraph, path: str) -> Any:
    nid, param = override_target(cg, path)
    node = cg.nodes.get(nid)
    return node.params.get(param) if node else None


def canonical_view(cg: CompiledGraph) -> Dict[str, Any]:
    """해시와 비교의 대상이 되는 정규 표현. 좌표·주석·UI 상태는 없다."""
    return {
        "abi": ENGINE_ABI,
        "id": cg.id,
        "sample_space": cg.sample_space,
        "materialize": cg.materialize,
        "runtime_profile": cg.runtime_profile,
        "nodes": [
            {
                "id": n.id,
                "type": n.ref,
                "kind": n.kind.value,
                "params": n.params,
                "outputs": {p: str(t) for p, t in sorted(n.output_types.items())},
            }
            for n in (cg.nodes[i] for i in cg.order)
        ],
        "edges": sorted([e.src, e.dst] for e in cg.edges),
    }
