"""그래프 실행 엔진.

노드 3분류가 정책을 결정한다.
  Input      — 캐시 키에 데이터 지문이 들어간다
  Processing — 순수 함수. 캐시 대상이고 노드 단위 미리보기가 허용된다
  Output     — 부작용을 일으키는 유일한 지점. 캐시 금지

실패는 (노드, 샘플) 단위로 격리된다. quarantine 비율이 임계를 넘으면 실행을 멈춘다 —
데이터의 20%가 조용히 사라진 채 학습이 도는 것을 막기 위해서다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..core.compiler import CompiledGraph, CompiledNode
from ..core.errors import VlmtError
from ..core.node import NodeError, NodeKind, RunCtx
from ..core.registry import resolve as resolve_node
from . import samples as samples_mod
from . import worker
from . import preview as preview_mod
from .cache import CacheStore, sample_key_hash
from .values import value_hash

PENDING, RUNNING, SUCCESS, CACHED, FAILED, SKIPPED = (
    "pending",
    "running",
    "success",
    "cached",
    "failed",
    "skipped",
)


@dataclass
class RunOptions:
    run_id: str = "dev"
    run_outputs: bool = False
    use_cache: bool = True
    cache_dir: str = ".cache"
    isolate_external: bool = True
    on_sample_error: str = "quarantine"  # quarantine | abort
    quarantine_ratio_threshold: float = 0.05
    seed: int = 20260909
    extra_modules: Tuple[str, ...] = ()
    determinism_audit: bool = False
    spec_dir: str = "."
    debug_output: bool = False
    trigger: str = "cli"  # ui | cli | external


@dataclass
class Quarantined:
    sample_key: str
    node_id: str
    cause: str
    hint: str = ""


@dataclass
class RunReport:
    order: List[str] = field(default_factory=list)
    node_state: Dict[str, Dict[str, int]] = field(default_factory=dict)
    node_ms: Dict[str, float] = field(default_factory=dict)
    quarantine: List[Quarantined] = field(default_factory=list)
    processed: int = 0
    cache: Dict[str, int] = field(default_factory=dict)
    aborted: str = ""
    last_values: Dict[str, Any] = field(default_factory=dict)
    determinism_failures: List[str] = field(default_factory=list)
    previews: Dict[str, Any] = field(default_factory=dict)  # 노드 -> Preview (토글이 켜졌을 때만)

    def count(self, node_id: str, state: str) -> None:
        self.node_state.setdefault(node_id, {})
        self.node_state[node_id][state] = self.node_state[node_id].get(state, 0) + 1

    def states_of(self, node_id: str) -> Dict[str, int]:
        return self.node_state.get(node_id, {})

    @property
    def quarantine_ratio(self) -> float:
        total = self.processed + len(self.quarantine)
        return len(self.quarantine) / total if total else 0.0


def ancestors(cg: CompiledGraph, node_id: str) -> Set[str]:
    """대상 노드와 그 상류 전부. 노드 단위 미리보기가 이 집합만 계산한다."""
    incoming: Dict[str, List[str]] = {i: [] for i in cg.order}
    for e in cg.edges:
        incoming[e.dst_node].append(e.src_node)
    out: Set[str] = set()
    stack = [node_id]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(incoming.get(cur, ()))
    return out


def _maybe_preview(rep: RunReport, opts: RunOptions, nid: str, d: Any, outputs: Dict[str, Any]) -> None:
    """Debug Output 규약 — 토글이 꺼져 있으면 **생성조차 하지 않는다**.

    외부에서 트리거된 실행에서는 값과 무관하게 만들지 않는다. 설계 문서 08 §8.5.
    """
    if not preview_mod.allowed(opts.trigger, opts.debug_output):
        return
    if d.kind is NodeKind.OUTPUT:
        return  # Output은 부작용이 있어 미리보기하지 않는다
    try:
        rep.previews[nid] = preview_mod.render(nid, d.preview or "generic", outputs)
    except Exception:
        pass  # 미리보기 실패가 실행을 막아서는 안 된다


def execute(
    cg: CompiledGraph,
    space: samples_mod.SampleSpace,
    rows: Sequence[Dict[str, Any]],
    opts: Optional[RunOptions] = None,
    targets: Optional[Set[str]] = None,
    on_sample: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> RunReport:
    opts = opts or RunOptions()
    cache = CacheStore(root=opts.cache_dir, enabled=opts.use_cache)
    rep = RunReport(order=list(cg.order))
    space_fp = samples_mod.fingerprint(space)

    plan = [i for i in cg.order if targets is None or i in targets]
    if not opts.run_outputs:
        plan = [i for i in plan if cg.nodes[i].kind is not NodeKind.OUTPUT]
    else:
        # per_sample=False Output(학습)은 샘플 루프가 아니라 train 단계에서 한 번 돈다
        plan = [i for i in plan if resolve_node(cg.nodes[i].ref).per_sample]

    for row in rows:
        key = str(row[space.key])
        values: Dict[str, Any] = {}
        failed_nodes: Set[str] = set()
        sample_failed = False

        for nid in plan:
            node = cg.nodes[nid]
            if any(src.split(":")[0] in failed_nodes for src in node.inputs.values()):
                rep.count(nid, SKIPPED)
                continue

            ctx = RunCtx(
                run_id=opts.run_id,
                node_id=nid,
                sample_key=key,
                seed=opts.seed,
                sample=row,
                root=space.root,
                spec_dir=opts.spec_dir,
            )
            d = resolve_node(node.ref)
            impl = d.impl()
            params = d.build_params(node.params)
            inputs = {p: values[src] for p, src in node.inputs.items() if src in values}
            missing = [p for p in node.inputs if p not in inputs]
            if missing:
                rep.count(nid, SKIPPED)
                continue

            # 캐시 키의 샘플 차원. Input 노드는 외부 상태의 지문이 함께 들어간다
            fp = impl.fingerprint(ctx, params) if node.kind is NodeKind.INPUT else ""
            shash = sample_key_hash(key, fp, space_fp if node.kind is NodeKind.INPUT else "")

            if node.kind is not NodeKind.OUTPUT:
                hit, cached_out = cache.get(node.cache_key, shash)
                if hit:
                    for port, v in cached_out.items():
                        values[f"{nid}:{port}"] = v
                    _maybe_preview(rep, opts, nid, d, cached_out)
                    rep.count(nid, CACHED)
                    continue

            t0 = time.perf_counter()
            try:
                isolated = opts.isolate_external and d.external_call
                if isolated:
                    out = worker.run_isolated(node.ref, node.params, inputs, ctx, opts.extra_modules)
                else:
                    out = impl.run(ctx, params, **inputs)
                if opts.determinism_audit and node.kind is NodeKind.PROCESSING:
                    again = (
                        worker.run_isolated(node.ref, node.params, inputs, ctx, opts.extra_modules)
                        if isolated
                        else impl.run(ctx, params, **inputs)
                    )
                    if value_hash(out) != value_hash(again):
                        rep.determinism_failures.append(nid)
            except NodeError as e:
                rep.count(nid, FAILED)
                failed_nodes.add(nid)
                sample_failed = True
                rep.quarantine.append(Quarantined(key, nid, e.cause, e.hint))
                if opts.on_sample_error == "abort":
                    rep.aborted = f"{nid} 실패로 중단: {e.cause}"
                    return rep
                break
            except Exception as e:  # 노드가 계약을 어기고 일반 예외를 던진 경우
                rep.count(nid, FAILED)
                failed_nodes.add(nid)
                sample_failed = True
                rep.quarantine.append(Quarantined(key, nid, f"{type(e).__name__}: {e}"))
                if opts.on_sample_error == "abort":
                    rep.aborted = f"{nid} 실패로 중단: {e}"
                    return rep
                break

            _maybe_preview(rep, opts, nid, d, out)
            rep.node_ms[nid] = rep.node_ms.get(nid, 0.0) + (time.perf_counter() - t0) * 1000
            for port, v in out.items():
                values[f"{nid}:{port}"] = v
            if node.kind is not NodeKind.OUTPUT:
                cache.put(node.cache_key, shash, out)
            rep.count(nid, SUCCESS)

        if not sample_failed:
            rep.processed += 1
            rep.last_values = values
            if on_sample is not None:
                on_sample(key, values)

        if rep.quarantine_ratio > opts.quarantine_ratio_threshold and len(rep.quarantine) >= 2:
            rep.aborted = (
                f"quarantine 비율 {rep.quarantine_ratio:.1%}가 임계 "
                f"{opts.quarantine_ratio_threshold:.1%}를 넘었다. "
                "데이터가 조용히 사라진 채 학습이 도는 것을 막기 위해 멈춘다."
            )
            break

    rep.cache = cache.stats
    return rep
