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




# 파이프라인이 단계마다 직접 꽂는 인자. 나머지는 전부 `pipeline` 파서가 받아야 한다 —
# 하나라도 빠지면 그 단계에서 AttributeError 로 죽고, 하필 굽기가 끝난 뒤에 죽는다.
STAGE_SET_ARGS = frozenset({"split", "run_id", "resume", "skip_budget"})


def cmd_pipeline(a: argparse.Namespace) -> int:
    """준비 → 굽기 → 학습 → 추론을 한 번에.

    **여기에 실행 코드는 없다.** 위에 있는 `cmd_materialize` · `cmd_train` · `cmd_run` 을
    그 순서로 부를 뿐이고, 각 단계는 터미널에서 그 명령을 직접 쳤을 때와 글자 하나까지
    같은 일을 한다. 부른 명령을 그대로 찍어 주는 것도 그래서다 — 어느 단계가 이상하면
    그 줄만 복사해 따로 돌려 볼 수 있어야 한다.
    """
    import copy

    from ..engine import pipeline as pipe

    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_recipe_overrides(a))
    run_id = _run_id(a)
    snap = pipe.snapshot_path(run_id)

    # 굽다 만 것부터 판단한다. 멈출 자리라면 아무것도 계산하지 않고 멈춘다.
    left = pipe.leftover(run_id)
    resume = bool(a.resume)
    if left and not a.resume and not a.fresh:
        # 말없이 이어받으면 예전 파라미터로 구운 샘플이 새 학습에 섞이고,
        # 말없이 지우면 몇 시간이 날아간다. 양쪽 다 나쁘므로 묻는다.
        answer = None
        if sys.stdin is not None and sys.stdin.isatty():
            print(f"굽다 만 것이 있다: shard {left.shards}개 · 샘플 {left.samples}건")
            print(f"  {left.out_dir}")
            try:
                answer = input("  이어 받을까? [y/N] ").strip().lower().startswith("y")
            except (EOFError, KeyboardInterrupt):
                # 터미널처럼 보였지만 읽을 것이 없다. 그 경우에도 대신 고르지 않는다.
                print()
                answer = None
        if answer is None:
            print(f"굽다 만 것이 있다: shard {left.shards}개 · 샘플 {left.samples}건", file=sys.stderr)
            print(f"  {left.out_dir}", file=sys.stderr)
            print("  이어 받으려면 --resume, 처음부터 구우려면 --fresh 를 준다.", file=sys.stderr)
            print("  물어볼 수 없는 자리라 대신 고르지 않는다 — 말없이 이어받으면 예전",
                  file=sys.stderr)
            print("  파라미터로 구운 샘플이 섞이고, 말없이 지우면 구운 시간이 날아간다.",
                  file=sys.stderr)
            return 8
        resume = answer

    p = pipe.plan(cg, run_id, bake_split=_bake_split(a, cg))
    print(f"pipeline {run_id} · {pipe.headline(p)}")
    p.started = time.time()
    pipe.write(p, snap)

    for stage in p.stages:
        if stage.key == "prepare" and a.skip_budget:
            stage.state, stage.note = pipe.SKIPPED, "--skip-budget"
            pipe.write(p, snap)
            continue

        ns = copy.copy(a)
        ns.run_id = run_id
        ns.split = stage.split
        ns.resume = resume if stage.key == "bake" else bool(a.resume)
        # 예산은 준비 단계에서 이미 봤다. 단계마다 다시 보면 dry-run 실측이 매번 돈다.
        ns.skip_budget = stage.key != "prepare"
        stage.state = pipe.RUNNING
        pipe.write(p, snap)

        t0 = time.time()
        print()
        print(f"[{stage.label}] vlmt {stage.command} {os.path.basename(a.spec)}"
              + (f" --split {stage.split}" if stage.split else "")
              + (" --resume" if ns.resume and stage.key == "bake" else ""))
        try:
            code = _STAGE_FUNCS[stage.command](ns)
        except SystemExit as exc:            # 하위 명령이 못 돌겠다고 한 경우
            code, stage.note = 1, str(exc)
            print(stage.note, file=sys.stderr)
        stage.ms = (time.time() - t0) * 1000
        stage.state = pipe.DONE if code == 0 else pipe.FAILED
        pipe.write(p, snap)

        if code != 0:
            # 뒤 단계는 앞 단계의 산출물을 읽는다. 굽기가 깨졌는데 학습을 시작하면
            # 반쯤 구운 shard 로 몇 시간을 태우고 나서야 안다.
            for rest in p.stages:
                if rest.state == pipe.PENDING:
                    rest.state = pipe.SKIPPED
                    rest.note = f"{stage.label} 단계가 끝나지 않았다"
            p.finished = time.time()
            pipe.write(p, snap)
            print()
            print(pipe.render(p))
            print(f"  {stage.label} 단계에서 멈췄다 (종료 코드 {code})", file=sys.stderr)
            return code

    p.finished = time.time()
    pipe.write(p, snap)
    print()
    print(pipe.render(p))
    print(f"  전체 {_took((p.finished - p.started) * 1000)} · runs/{run_id}")
    return 0


# 단계 이름 -> 위에 있는 명령. 파이프라인이 새 실행 경로를 만들지 않는다는 것이
# 이 표로 드러난다.
def _bake_split(a: argparse.Namespace, cg: Any) -> str:
    """굽기가 돌 split. 기본은 **학습용만** 굽는 것이다.

    학습은 구워진 shard 를 전부 읽는다. 검증 샘플까지 구워 두면 그것도 학습에 들어가고,
    그러면 "검증 10장이 뭐라 답하는지 본다" 가 아무 의미도 없어진다. 이미 외운 것을
    다시 물어보는 셈이다. 조용히 그렇게 되는 것이 나빠서 고른 이유를 찍어 준다.

    split 이 하나뿐인 그래프는 나눌 것이 없으니 그대로 전부 굽는다.
    """
    want = (a.bake_split or "auto").strip()
    if want == "all":
        return ""
    if want != "auto":
        return want
    names = {str(r.get("_split", "")) for r in _space(a, cg).rows}
    if len(names) > 1 and "train" in names:
        print("굽기는 train split 만 굽는다 — 검증 샘플을 구우면 학습이 그것까지 읽는다")
        print("  전부 구우려면 --bake-split all")
        return "train"
    return ""


_STAGE_FUNCS = {
    "budget": cmd_budget,
    "materialize": cmd_materialize,
    "train": cmd_train,
    "run": cmd_run,
}


def cmd_export(a: argparse.Namespace) -> int:
    """`모델 Export` — LoRA 를 합친 한 덩어리 + 계약.

    **계약을 반드시 함께 넣는다.** 프롬프트를 어떤 형식으로 넣어야 하는지 모르면
    가중치만 있어도 쓸 수 없고, 받은 사람은 답이 안 나오는 이유를 모델 탓으로 돌린다.
    """
    from ..train import checkpoint as ckpt_mod

    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_recipe_overrides(a))
    got = _trainer_cfg(a, cg)
    if got is None:
        raise SystemExit("이 그래프에는 Trainer 노드가 없다. 내보낼 모델이 없다.")
    cfg, _ = got

    run_id = _run_id(a)
    model_dir = a.model_dir or os.path.join("runs", run_id, "train")
    out_dir = a.out or os.path.join("runs", run_id, "export")

    t0 = time.time()
    rep = ckpt_mod.export(os.path.abspath(model_dir), cfg.backbone, out_dir)
    print(ckpt_mod.render(rep))
    print(f"  {_took((time.time() - t0) * 1000)}")
    return 0 if rep.ok else 9
