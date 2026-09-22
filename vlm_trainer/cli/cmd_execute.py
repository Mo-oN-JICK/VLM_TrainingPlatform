"""실제로 돌리는 명령 — dryrun · run · materialize · train · budget

`cli/main.py` 가 919줄이라 갈랐다. **함수 이름도 인자도 동작도 그대로다** —
`build_parser` 가 여기 있는 것을 `set_defaults(func=...)` 로 가리킨다.

공용 헬퍼(`_load_nodes`, `_space`, `_run_id` …)는 `main.py` 에 남아 있고 여기서 가져다 쓴다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .common import _parse_ids, _load_nodes, _measure_tokens, _progress_path, _recipe_overrides, _run_id, _space, _took, _train_logger, _trainer_cfg
from ..core.node import NodeKind
from ..engine.runner import RunOptions
from ..engine.runner import ancestors
from ..engine import budget as budget_mod
from ..core.compiler import compile_project
from ..train import contract as contract_mod
from ..engine import dryrun as dryrun_mod
from ..engine.runner import execute
from ..engine import materialize as materialize_mod
from ..engine import preview as preview_mod
from ..spec import recipe as recipe_mod
from ..core import registry
from ..train import shards as shards_mod
from ..engine import sweep as sweep_mod

def cmd_dryrun(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_recipe_overrides(a))
    space = _space(a, cg)
    opts = RunOptions(run_id=_run_id(a), extra_modules=tuple(a.nodes), cache_dir=a.cache_dir,
                      spec_dir=os.path.dirname(os.path.abspath(a.spec)))
    res = dryrun_mod.dryrun(cg, space, n=a.samples, opts=opts, violation_threshold=a.violation_threshold)
    print(f"샘플 공간: {len(space)}건 (인덱스 {os.path.basename(space.index_path)})")
    print(dryrun_mod.render(res))
    if a.verbose and res.measured:
        print()
        print("실측:")
        for ref, desc in res.measured.items():
            print(f"  {ref}: {desc}")
    return 0 if res.ok else 3


def cmd_run(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_recipe_overrides(a))
    space = _space(a, cg)
    if a.split:
        space = space.split(a.split)
    rows = space.rows[: a.limit] if a.limit else space.rows
    opts = RunOptions(
        run_id=_run_id(a),
        run_outputs=True,
        use_cache=not a.no_cache,
        cache_dir=a.cache_dir,
        extra_modules=tuple(a.nodes),
        spec_dir=os.path.dirname(os.path.abspath(a.spec)),
        trigger=getattr(a, "trigger", "cli"),
        cache_backend=getattr(a, "cache_backend", "local"),
        preview_dir=os.path.join("runs", _run_id(a), "preview"),
        progress_path=_progress_path(a, _run_id(a)),
    )
    # G4 — GPU를 잡기 전 마지막 문. Trainer가 있으면 예산을 먼저 본다.
    got = _trainer_cfg(a, cg)
    if got is not None and not a.skip_budget:
        cfg, tn = got
        measured, how = _measure_tokens(a, cg, cfg)
        b = budget_mod.for_graph(cg, cfg, tn, measured_text_tokens=measured, measured_how=how)
        if not b.ok and cfg.budget.policy == "fail_fast":
            print(budget_mod.render(b), file=sys.stderr)
            print("", file=sys.stderr)
            print("학습을 시작하지 않았다. Output 노드는 실행되지 않는다.", file=sys.stderr)
            return 4

    opts.debug_output = a.debug_output
    rep = execute(cg, space, rows, opts)
    took = (time.time() - rep.started) * 1000 if rep.started else 0.0
    print(f"run {opts.run_id}: {rep.processed}/{len(rows)}건 처리 · {_took(took)}")
    print(f"  캐시 {rep.cache}")
    for nid in rep.order:
        st = rep.states_of(nid)
        if st:
            ms = rep.node_ms.get(nid, 0.0)
            print(f"  {nid:<18} {st}  {_took(ms)}")
    if rep.quarantine:
        print()
        print(f"격리 {len(rep.quarantine)}건 (비율 {rep.quarantine_ratio:.1%}):")
        for q in rep.quarantine[:10]:
            print(f"  {q.sample_key} @{q.node_id}: {q.cause}")
    if a.view:
        from ..ui import render as render_mod

        out = a.view if a.view != "-" else os.path.join("runs", opts.run_id, "graph.html")
        path = render_mod.write(
            cg, out, title=cg.name or cg.id, note=f"run {opts.run_id}", report=rep
        )
        print(f"  그래프 뷰 -> {path}")

    if rep.aborted:
        print()
        print(f"중단: {rep.aborted}")
        return 3
    return 0


def cmd_materialize(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_recipe_overrides(a))
    space = _space(a, cg)
    ro = RunOptions(
        run_id=_run_id(a),
        cache_dir=a.cache_dir,
        extra_modules=tuple(a.nodes),
        spec_dir=os.path.dirname(os.path.abspath(a.spec)),
    )
    mo = materialize_mod.MaterializeOptions(
        out_dir=a.out_dir,
        shard_size=a.shard_size,
        resume=a.resume,
        limit=a.limit,
        split=a.split,
    )
    t0 = time.time()
    rep = materialize_mod.materialize(cg, space, mo, ro)
    print(f"run {ro.run_id} · {_took((time.time() - t0) * 1000)}")
    print(materialize_mod.render(rep))
    if rep.ok:
        st = shards_mod.stats(rep.out_dir)
        print(
            f"  산출물: shard {st.shards}개 · 샘플 {st.samples}건 · 이미지 {st.images}장 "
            f"(샘플당 최대 {st.max_images_per_sample}) · 평균 {st.avg_chars:.0f}자"
        )
    return 0 if rep.ok else 5


def cmd_train(a: argparse.Namespace) -> int:
    from ..engine.journal import Journal
    from ..train import loop as loop_mod

    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_recipe_overrides(a))
    got = _trainer_cfg(a, cg)
    if got is None:
        raise SystemExit("이 그래프에는 Trainer 노드가 없다.")
    cfg, tn = got
    run_id = _run_id(a)
    spec_dir = os.path.dirname(os.path.abspath(a.spec))

    # G4 — GPU를 잡기 전 마지막 문
    mat_dir = a.materialized or os.path.join("runs", run_id, "materialized")
    measured = 0
    if not a.skip_budget:
        measured, how = _measure_tokens(a, cg, cfg)
        b = budget_mod.for_graph(cg, cfg, tn, measured_text_tokens=measured, measured_how=how)
        if not b.ok and cfg.budget.policy == "fail_fast":
            print(budget_mod.render(b), file=sys.stderr)
            print("", file=sys.stderr)
            print("학습을 시작하지 않았다.", file=sys.stderr)
            return 4
        measured = b.s_vision

    out_dir = os.path.join("runs", run_id, "train")
    t_train = time.time()
    journal = Journal(os.path.join("runs", run_id, "journal.jsonl"))
    rep = loop_mod.train(
        cfg,
        os.path.abspath(mat_dir),
        os.path.abspath(out_dir),
        device=a.device_torch,
        journal=journal,
        resume=a.resume,
        max_steps=a.max_steps,
        on_log=_train_logger(a, run_id),
    )
    took = (time.time() - t_train) * 1000
    rep.contract = contract_mod.write(
        os.path.abspath(out_dir), cg, cfg, spec_dir, vision_tokens=measured
    )
    print(loop_mod.render(rep))
    print(f"  전체 {_took(took)}")
    return 0 if rep.ok else 6


def cmd_budget(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_recipe_overrides(a))
    got = _trainer_cfg(a, cg)
    if got is None:
        print("이 그래프에는 Trainer 노드가 없다. 예산 검사 대상이 아니다.")
        return 0
    cfg, tn = got
    measured, how = (0, "") if a.no_measure else _measure_tokens(a, cg, cfg)
    res = budget_mod.for_graph(cg, cfg, tn, measured_text_tokens=measured, measured_how=how)
    print(budget_mod.render(res))
    return 0 if res.ok else 4


def cmd_preview(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_recipe_overrides(a))
    if a.node not in cg.nodes:
        raise SystemExit(f"노드 {a.node!r}가 그래프에 없다. 가능한 id: {', '.join(cg.order)}")
    node = cg.nodes[a.node]
    if node.kind is NodeKind.OUTPUT:
        raise SystemExit(f"{a.node}는 Output 노드다. 부작용이 있으므로 미리보기하지 않는다.")

    space = _space(a, cg)
    rows = [r for r in space.rows if str(r[space.key]) == a.sample] if a.sample else space.pick(1)
    if not rows:
        raise SystemExit(f"샘플 {a.sample!r}를 찾을 수 없다")

    opts = RunOptions(run_id=_run_id(a), cache_dir=a.cache_dir, extra_modules=tuple(a.nodes),
                      use_cache=not a.no_cache, spec_dir=os.path.dirname(os.path.abspath(a.spec)))
    rep = execute(cg, space, rows, opts, targets=ancestors(cg, a.node))
    outs = {p: v for p, v in ((p, rep.last_values.get(f"{a.node}:{p}")) for p in node.output_types)
            if v is not None}
    if not outs:
        print(f"{a.node}: 값이 생성되지 않았다 (상류 실패 여부를 확인하라)")
        for q in rep.quarantine[:5]:
            print(f"  {q.sample_key} @{q.node_id}: {q.cause}")
        return 3

    out_dir = os.path.join("runs", opts.run_id, "previews") if a.save else ""
    kind = registry.resolve(node.ref).preview or "generic"
    pv = preview_mod.render(a.node, kind, outs, out_dir)
    executed = sum(1 for n in rep.order if rep.states_of(n).get("success"))
    cached = sum(1 for n in rep.order if rep.states_of(n).get("cached"))
    print(f"미리보기 {a.node} ({node.ref}) · 샘플 {rows[0][space.key]}")
    print(f"  상류 {len(ancestors(cg, a.node))}개 중 실행 {executed} · 캐시 {cached}")
    print(f"  종류: {pv.kind}")
    print(pv.text)
    if pv.image_path:
        print(f"  이미지 저장: {pv.image_path}")
    return 0


def cmd_sweep(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    book = recipe_mod.load(a.spec)
    if a.sweep:
        recipes = recipe_mod.expand(book, a.sweep)
    elif a.recipes:
        recipes = [book.get(i) for i in _parse_ids(a.recipes)]
    else:
        recipes = [book.recipes[i] for i in sorted(book.recipes)]
    if not recipes:
        raise SystemExit("돌릴 레시피가 없다.")

    print(f"스윕 시작: 레시피 {len(recipes)}개 · 단일 GPU 순차 큐")
    rep = sweep_mod.run(
        a.spec,
        recipes,
        run_root=a.run_root,
        trainer_config=a.trainer_config or None,
        limit=a.limit,
        shard_size=a.shard_size,
        device=a.device,
        torch_device=a.device_torch,
        max_steps=a.max_steps,
        train=not a.no_train,
        cache_dir=a.cache_dir,
        on_event=print,
    )
    print()
    print(sweep_mod.render(rep))
    return 0 if not rep.rejected else 7


