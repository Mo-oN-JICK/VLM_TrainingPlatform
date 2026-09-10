"""vlmt — 커맨드라인 진입점.

UI 없이 스펙만으로 전부 할 수 있어야 한다. UI 전용 실행 경로는 존재하지 않는다.
Phase 2 범위: compile / decompile / nodes / show / dryrun / run / preview.
budget(G4) · materialize · sweep은 Phase 3 이후에 붙는다.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ..core import registry
from ..core.compiler import CompileFailed, canonical_view, compile_project, current_value
from ..core.errors import VlmtError
from ..core.node import NodeKind
from ..engine import budget as budget_mod
from ..engine import materialize as materialize_mod
from ..engine import sweep as sweep_mod
from ..engine import dryrun as dryrun_mod
from ..engine import preview as preview_mod
from ..engine import samples as samples_mod
from ..engine.runner import RunOptions, ancestors, execute
from ..spec import recipe as recipe_mod
from ..train import tokens as tokens_mod
from ..spec.decompile import decompile, dump_yaml
from ..train.config import TrainerConfig
from ..train import shards as shards_mod
from ..train import contract as contract_mod


def _load_nodes(modules: List[str]) -> None:
    registry.load_builtin_nodes()
    for m in modules or []:
        importlib.import_module(m)


def _recipe_overrides(a: argparse.Namespace) -> Dict[str, Any]:
    """--recipe N 의 오버라이드에 --set 을 얹는다. 손으로 준 값이 마지막에 이긴다."""
    out: Dict[str, Any] = {}
    rid = getattr(a, "recipe", None)
    if rid:
        book = recipe_mod.load(a.spec)
        out.update(book.overrides_for(int(rid)))
    out.update(_overrides(getattr(a, "set", []) or []))
    return out


def _overrides(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set 은 경로=값 형식이어야 한다: {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def cmd_compile(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_recipe_overrides(a))
    print(f"compile OK  {cg.id or os.path.basename(a.spec)}")
    print(f"  노드 {len(cg.nodes)}개 · 배선 {len(cg.edges)}개 · 레인 {max(cg.lanes.values()) + 1}단")
    print(f"  spec_hash {cg.spec_hash}")
    if a.verbose:
        print()
        width = max(len(i) for i in cg.order)
        for nid in cg.order:
            n = cg.nodes[nid]
            tag = {NodeKind.INPUT: "I", NodeKind.PROCESSING: "P", NodeKind.OUTPUT: "O"}[n.kind]
            print(f"  [{tag}] lane{n.lane} {nid:<{width}}  {n.ref}")
            for port, t in sorted(n.output_types.items()):
                print(f"          {port}: {t}")
    if a.out:
        payload = {
            "spec_hash": cg.spec_hash,
            "order": cg.order,
            "lanes": cg.lanes,
            "nodes": canonical_view(cg)["nodes"],
            "cache_keys": {i: cg.nodes[i].cache_key for i in cg.order},
        }
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"  -> {a.out}")
    return 0


def cmd_decompile(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    cg = compile_project(a.spec)
    text = dump_yaml(decompile(cg, keep_procedures=not a.flatten))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"decompile -> {a.out}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_nodes(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    cats = registry.categories()
    if not cats:
        print("등록된 노드가 없다. --nodes MODULE 로 노드 모듈을 지정하라.")
        return 0
    for cat, defs in cats.items():
        print(f"\n▸ {cat}")
        for d in defs:
            tag = {NodeKind.INPUT: "I", NodeKind.PROCESSING: "P", NodeKind.OUTPUT: "O"}[d.kind]
            print(f"   [{tag}] {d.ref}")
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    d = registry.resolve(a.type)
    print(f"{d.ref}   [{d.kind.value}]   {d.category}")
    if d.doc.summary:
        print(f"\n{d.doc.summary}")
    if d.doc.scenario:
        print(f"사용 시나리오: {d.doc.scenario}")
    print("\n입력 포트")
    for p, port in d.inputs.items() or {}:
        print(f"  {p}: {port.type}")
    if not d.inputs:
        print("  (없음 — 그래프의 시작점)")
    print("출력 포트")
    for p, port in d.outputs.items():
        print(f"  {p}: {port.type}")
    if not d.outputs:
        print("  (없음 — 그래프의 종결점)")
    print("파라미터")
    for k, v in d.default_params().items():
        mark = " *recipe" if k in d.recipe_overridable else ""
        mark += " *type" if k in d.type_affecting else ""
        print(f"  {k} = {v!r}{mark}")
    if d.external_call:
        print("\n외부 모델을 호출한다 — 물질화 경계 앞에 있어야 한다.")
    return 0


def _space(a: argparse.Namespace, cg) -> samples_mod.SampleSpace:
    return samples_mod.load(cg.sample_space, os.path.dirname(os.path.abspath(a.spec)))


def _progress_path(a: argparse.Namespace, run_id: str) -> str:
    """진행 상황 스냅샷의 자리. CLI 실행도 남긴다 — 다른 터미널이나 편집기가 지켜볼 수 있다."""
    p = getattr(a, "progress", "")
    if p == "off":
        return ""
    return p or os.path.join("runs", run_id, "progress.json")


def _run_id(a: argparse.Namespace) -> str:
    import time

    return getattr(a, "run_id", "") or time.strftime("%Y%m%dT%H%M%S")


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


def _trainer_cfg(a: argparse.Namespace, cg) -> Optional[tuple]:
    """(TrainerConfig, train_node_id). Trainer 노드가 없으면 None."""
    nodes = budget_mod.train_nodes(cg)
    if not nodes:
        return None
    tn = nodes[0]
    path = cg.nodes[tn].params.get("config_path") or "trainer.yaml"
    spec_dir = os.path.dirname(os.path.abspath(a.spec))
    cfg = TrainerConfig.load(os.path.normpath(os.path.join(spec_dir, path)))
    if getattr(a, "device", ""):
        cfg.budget.device = a.device
    for kv in getattr(a, "what_if", None) or []:
        k, _, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        if k == "images":
            cfg.vision.max_images_per_sample = int(v)
        elif k == "tiles":
            cfg.vision.max_tiles = int(v)
            cfg.vision.tiling = int(v) > 1
        elif k == "max_len":
            cfg.sequence.max_len = int(v)
        elif k == "backbone":
            cfg.backbone = v
        elif k == "quantization":
            cfg.quantization.mode = v
        elif k == "per_device":
            for st in cfg.stages:
                st.per_device = int(v)
        else:
            raise SystemExit(
                f"--what-if 에 알 수 없는 키: {k!r} "
                "(images|tiles|max_len|backbone|quantization|per_device)"
            )
    return cfg, tn


def _tokenizer_id(cfg) -> str:
    """백본이 보고한 토크나이저. 없으면 백본 id 자체를 쓴다(hf:<id>가 곧 토크나이저다)."""
    from ..plugins.base import resolve_backbone

    try:
        spec = resolve_backbone(cfg.backbone).spec()
    except Exception:
        return ""
    return spec.tokenizer_id or (cfg.backbone[3:] if cfg.backbone.startswith("hf:") else "")


def _measure_tokens(a: argparse.Namespace, cg, cfg) -> Tuple[int, str]:
    """dry-run이 실제로 만든 텍스트를 센다. 토크나이저가 없으면 비율로 추정한다.

    정적 추정이 실측보다 낙관적이면 G4가 잡는다.
    """
    space = _space(a, cg)
    opts = RunOptions(
        run_id=_run_id(a),
        extra_modules=tuple(a.nodes),
        cache_dir=a.cache_dir,
        spec_dir=os.path.dirname(os.path.abspath(a.spec)),
    )
    res = dryrun_mod.dryrun(cg, space, n=1, opts=opts)
    if not res.measured_chars:
        return 0, ""
    return tokens_mod.measure(res.measured_text, res.measured_chars, cfg.sequence, _tokenizer_id(cfg))


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
    print(f"run {opts.run_id}: {rep.processed}/{len(rows)}건 처리")
    print(f"  캐시 {rep.cache}")
    for nid in rep.order:
        st = rep.states_of(nid)
        if st:
            ms = rep.node_ms.get(nid, 0.0)
            print(f"  {nid:<18} {st}  {ms:.0f}ms")
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
    rep = materialize_mod.materialize(cg, space, mo, ro)
    print(f"run {ro.run_id}")
    print(materialize_mod.render(rep))
    if rep.ok:
        st = shards_mod.stats(rep.out_dir)
        print(
            f"  산출물: shard {st.shards}개 · 샘플 {st.samples}건 · 이미지 {st.images}장 "
            f"(샘플당 최대 {st.max_images_per_sample}) · 평균 {st.avg_chars:.0f}자"
        )
    return 0 if rep.ok else 5


def cmd_infer_graph(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_overrides(a.set))
    spec = contract_mod.extract_inference_graph(cg)
    text = dump_yaml(spec)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"추론 그래프 -> {a.out}")
        print(f"  프롬프트 종단: {contract_mod.prompt_terminus(cg)}")
        print(f"  노드 {len(spec['nodes'])}개 (정답 경로는 잘라냈다)")
    else:
        sys.stdout.write(text)
    return 0


def _train_logger(a: argparse.Namespace, run_id: str):
    """콘솔에 찍고, 같은 사실을 진행 파일에도 남긴다.

    실행(`run`)과 같은 통로다 — 다른 터미널이나 편집기가 학습을 지켜볼 수 있어야 하는데,
    stdout만 있으면 그 사실이 프로세스 안에 갇힌다.
    """
    path = _progress_path(a, run_id)
    if path:
        path = os.path.join(os.path.dirname(path), "train_progress.json")
    state = {"run_id": run_id, "phase": "train", "stage": "", "step": 0, "loss": 0.0, "at": 0.0}

    def log(stage: str, step: int, loss: float) -> None:
        print(f"    {stage} step {step:>4} loss {loss:.4f}")
        if not path:
            return
        state.update(stage=stage, step=int(step), loss=float(loss), at=time.time())
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError:
            pass  # 관찰이 학습을 죽여서는 안 된다

    return log


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
    rep.contract = contract_mod.write(
        os.path.abspath(out_dir), cg, cfg, spec_dir, vision_tokens=measured
    )
    print(loop_mod.render(rep))
    return 0 if rep.ok else 6


def cmd_new(a: argparse.Namespace) -> int:
    """빈 Solution/Project 껍데기를 만들고, 원하면 편집기를 연다."""
    from ..spec import scaffold

    proj, made = scaffold.new_project(
        a.dir,
        solution_id=a.solution_id,
        project_id=a.project_id,
        name=a.name,
        index=a.index,
        profile=a.profile,
    )
    for f in made:
        print(f"  만듦: {f}")
    print()
    print("그래프는 비어 있다. 노드를 놓기 전에는 컴파일되지 않는다 — Output이 하나는 있어야 한다.")
    print(f"  편집기: vlmt edit {proj}")
    print(f"  샘플 인덱스: project.yaml의 sample_space.index가 가리키는 JSONL을 먼저 만들어라")
    if a.edit:
        from ..ui import server as server_mod

        server_mod.serve(proj, port=a.port, open_browser=True, extra_modules=tuple(a.nodes))
    return 0


def cmd_edit(a: argparse.Namespace) -> int:
    from ..ui import server as server_mod

    _load_nodes(a.nodes)
    server_mod.serve(a.spec, port=a.port, open_browser=a.open, extra_modules=tuple(a.nodes))
    return 0


def cmd_backbones(a: argparse.Namespace) -> int:
    from ..plugins.base import all_backbones, resolve_backbone

    _load_nodes(a.nodes)
    if a.add:
        s = resolve_backbone(a.add if a.add.startswith("hf:") else "hf:" + a.add).spec()
        print(f"등록: {s.id}")
    print(f"{'백본':<38}{'파라미터':>10}{'레이어':>7}{'hidden':>8}{'토큰/타일':>10}{'컨텍스트':>10}")
    for s in all_backbones():
        print(
            f"{s.id:<38}{s.params_total / 1e9:>9.2f}B{s.n_layers:>7}{s.hidden:>8}"
            f"{s.tokens_per_tile:>10}{s.max_context:>10}"
        )
    return 0


def cmd_view(a: argparse.Namespace) -> int:
    from ..ui import render as render_mod

    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_recipe_overrides(a))
    out = a.out or os.path.join("runs", "view", f"{cg.id or 'graph'}.html")
    path = render_mod.write(cg, out, title=cg.name or cg.id, note=f"compile OK · {len(cg.nodes)} nodes")
    print(f"그래프 뷰 -> {path}")
    print(f"  레인 {max(cg.lanes.values()) + 1}단 · 노드 {len(cg.nodes)} · 배선 {len(cg.edges)}")
    if a.open:
        import webbrowser

        webbrowser.open(f"file:///{path.replace(os.sep, '/')}")
    return 0


def cmd_recipe(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    book = recipe_mod.load(a.spec)
    if not book.recipes and a.action != "expand":
        print(f"레시피가 없다: {book.path}")
        return 0

    if a.action == "list":
        cg = compile_project(a.spec)
        paths = {p for r in book.recipes.values() for p in r.overrides}
        st = recipe_mod.status(book, {p: current_value(cg, p) for p in paths})
        print(f"{book.path}")
        print(f"  활성: {st if st else '없음'}"
              + ("  (프로젝트를 직접 수정해 레시피와 어긋난다)" if st == recipe_mod.CUSTOMIZED else ""))
        for rid in sorted(book.recipes):
            r = book.recipes[rid]
            mark = "*" if str(rid) == st else " "
            print(f"  {mark}{rid:>3}  {r.name:<16} {r.note}")
        for name, sw in book.sweeps.items():
            total = 1
            for v in sw.axes.values():
                total *= len(v)
            print(f"   sweep {name}: {sw.strategy} · 축 {len(sw.axes)}개 · 조합 {total}개 · 번호 {sw.id_range}")
        return 0

    if a.action == "show":
        r = book.get(int(a.recipe_id))
        print(f"{r.id:>3}  {r.name}  {r.note}")
        for k, v in sorted(r.overrides.items()):
            print(f"    {book.display_name(k):<24} = {v!r}   [{k}]")
        return 0

    if a.action == "diff":
        x, y = book.get(int(a.recipe_id)), book.get(int(a.other))
        rows = recipe_mod.diff(x, y)
        print(f"{x.id} {x.name}  vs  {y.id} {y.name}")
        for k, va, vb in rows:
            print(f"    {book.display_name(k):<24} {va!r}  ->  {vb!r}")
        if not rows:
            print("    차이 없음")
        return 0

    if a.action == "expand":
        expanded = recipe_mod.expand(book, a.sweep)
        out_dir = os.path.dirname(os.path.abspath(a.spec))
        rep, _ = sweep_mod.plan(a.spec, expanded, device=a.device)
        lock = recipe_mod.save_lock(book, expanded, out_dir)
        print(f"전개: {len(expanded)}개 (번호 {expanded[0].id}~{expanded[-1].id})")
        print(f"  예산 통과: {len(rep.queued)}개  ·  제외: {len(rep.rejected)}개")
        for e in rep.rejected:
            print(f"    {e.recipe.label}: {e.reason}")
        print(f"  -> {lock}")
        return 0

    if a.action == "set-active":
        import yaml as _yaml

        with open(book.path, "r", encoding="utf-8") as fh:
            data = _yaml.safe_load(fh) or {}
        data["active"] = int(a.recipe_id)
        with open(book.path, "w", encoding="utf-8") as fh:
            _yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False, width=100)
        print(f"활성 레시피: {a.recipe_id}")
        return 0

    raise SystemExit(f"알 수 없는 action: {a.action}")


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


def _parse_ids(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vlmt", description="VLM Trainer 플랫폼 CLI")
    p.add_argument("--nodes", action="append", default=[], help="추가로 임포트할 노드 모듈")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="스펙을 컴파일하고 4중 게이트 중 G1/G2를 통과시킨다")
    c.add_argument("spec")
    c.add_argument("--set", action="append", default=[], help="파라미터 오버라이드 (노드id.파라미터=값)")
    c.add_argument("--out", help="compiled.json 저장 경로")
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=cmd_compile)

    d = sub.add_parser("decompile", help="컴파일된 그래프를 스펙으로 되돌린다")
    d.add_argument("spec")
    d.add_argument("--flatten", action="store_true", help="Procedure를 펼친 채로 출력")
    d.add_argument("--out")
    d.set_defaults(func=cmd_decompile)

    n = sub.add_parser("nodes", help="노드 라이브러리를 카테고리별로 나열한다")
    n.set_defaults(func=cmd_nodes)

    s = sub.add_parser("show", help="노드 하나의 포트와 파라미터를 보여준다 (Node Quick Info)")
    s.add_argument("type")
    s.set_defaults(func=cmd_show)

    dr = sub.add_parser("dryrun", help="G3 — 실제 샘플 몇 건을 전 노드에 통과시켜 실측 검증한다")
    dr.add_argument("spec")
    dr.add_argument("--samples", type=int, default=3)
    dr.add_argument("--set", action="append", default=[])
    dr.add_argument("--cache-dir", default=".cache")
    dr.add_argument("--violation-threshold", type=float, default=0.02)
    dr.add_argument("--run-id", default="")
    dr.add_argument("-v", "--verbose", action="store_true")
    dr.set_defaults(func=cmd_dryrun)

    r = sub.add_parser("run", help="그래프를 실행한다 (Output 노드 포함)")
    r.add_argument("spec")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--split", default="")
    r.add_argument("--set", action="append", default=[])
    r.add_argument("--cache-dir", default=".cache")
    r.add_argument("--cache-backend", default="local",
                   help="캐시 저장 백엔드 (local | memory). 키 계산은 백엔드와 무관하다")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--run-id", default="")
    r.add_argument("--device", default="", help="예산 프로파일 (rtx3060_12gb | rtx4090_24gb)")
    r.add_argument("--skip-budget", action="store_true",
                   help="예산 검사를 건너뛴다(Trainer 없는 그래프 전용)")
    r.add_argument("--view", default="", help="실행 상태를 칠한 그래프 HTML을 남긴다 ('-'면 runs/<id>/graph.html)")
    r.add_argument("--debug-output", action="store_true",
                   help="Debug Output 토글. 꺼져 있으면 미리보기를 생성조차 하지 않는다")
    r.add_argument("--trigger", default="cli", choices=("cli", "ui", "external"),
                   help="누가 이 실행을 띄웠나. Debug Output 규약이 여기에 달려 있다")
    r.add_argument("--progress", default="",
                   help="진행 상황 스냅샷 경로 (기본 runs/<run-id>/progress.json, 'off'면 남기지 않는다)")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("budget", help="G4 — 학습 전에 단계별 VRAM과 시퀀스 길이를 산정한다")
    b.add_argument("spec")
    b.add_argument("--set", action="append", default=[])
    b.add_argument("--device", default="", help="rtx3060_12gb | rtx4090_24gb | a100_40gb")
    b.add_argument("--what-if", action="append", default=[],
                   help="images=2 tiles=4 max_len=2048 backbone=dummy-7b quantization=none per_device=4")
    b.add_argument("--no-measure", action="store_true", help="dry-run 실측 없이 가정값으로 계산")
    b.add_argument("--cache-dir", default=".cache")
    b.add_argument("--run-id", default="")
    b.set_defaults(func=cmd_budget)

    m = sub.add_parser("materialize", help="물질화 경계까지 구워 shard로 남긴다 (학습은 이것만 읽는다)")
    m.add_argument("spec")
    m.add_argument("--out-dir", default="", help="기본값 runs/<run_id>/materialized")
    m.add_argument("--shard-size", type=int, default=64)
    m.add_argument("--resume", action="store_true", help="커밋된 shard를 인정하고 남은 샘플만 굽는다")
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--split", default="")
    m.add_argument("--set", action="append", default=[])
    m.add_argument("--cache-dir", default=".cache")
    m.add_argument("--run-id", default="")
    m.set_defaults(func=cmd_materialize)

    nw = sub.add_parser("new", help="빈 Solution/Project 껍데기를 만든다")
    nw.add_argument("dir", help="만들 디렉터리 (없으면 만든다)")
    nw.add_argument("--solution-id", default="", help="기본값은 디렉터리 이름")
    nw.add_argument("--project-id", default="", help="기본값은 01_<solution-id>")
    nw.add_argument("--name", default="", help="사람이 읽는 이름")
    nw.add_argument("--index", default="data/index.jsonl",
                    help="샘플 인덱스 JSONL의 자리 (Solution 루트 기준)")
    nw.add_argument("--profile", default="windows_single_gpu",
                    help="실행 프로파일 (windows_single_gpu | linux_multi_gpu)")
    nw.add_argument("--edit", action="store_true", help="만든 뒤 편집기를 연다")
    nw.add_argument("--port", type=int, default=8770)
    nw.set_defaults(func=cmd_new)

    ed = sub.add_parser("edit", help="그래프 편집기를 연다 (127.0.0.1 로컬 서버)")
    ed.add_argument("spec")
    ed.add_argument("--port", type=int, default=8770)
    ed.add_argument("--open", action="store_true")
    ed.set_defaults(func=cmd_edit)

    bb = sub.add_parser("backbones", help="등록된 백본과 그 형상을 보여준다 (가중치는 열지 않는다)")
    bb.add_argument("--add", default="", help="hf:<경로 또는 모델 id>를 config.json만 읽어 등록한다")
    bb.set_defaults(func=cmd_backbones)

    vw = sub.add_parser("view", help="컴파일된 그래프를 한 장의 HTML로 그린다 (읽기 전용)")
    vw.add_argument("spec")
    vw.add_argument("--set", action="append", default=[])
    vw.add_argument("--out", default="")
    vw.add_argument("--open", action="store_true", help="브라우저로 연다")
    vw.set_defaults(func=cmd_view)

    rc = sub.add_parser("recipe", help="Parameter Recipe 관리 (list/show/diff/expand/set-active)")
    rc.add_argument("action", choices=["list", "show", "diff", "expand", "set-active"])
    rc.add_argument("spec")
    rc.add_argument("--recipe-id", default="1")
    rc.add_argument("--other", default="2", help="diff 대상")
    rc.add_argument("--sweep", default="", help="expand 할 스윕 이름")
    rc.add_argument("--device", default="")
    rc.set_defaults(func=cmd_recipe)

    sw = sub.add_parser("sweep", help="레시피 여러 개를 순차로 돌린다 (물질화는 지문이 같으면 공유)")
    sw.add_argument("spec")
    sw.add_argument("--recipes", default="", help="예: 1,2,3 또는 10-21")
    sw.add_argument("--sweep", default="", help="스윕 이름으로 전개해서 돌린다")
    sw.add_argument("--run-root", default="runs/sweeps")
    sw.add_argument("--trainer-config", default="")
    sw.add_argument("--limit", type=int, default=0)
    sw.add_argument("--shard-size", type=int, default=64)
    sw.add_argument("--device", default="", help="예산 프로파일")
    sw.add_argument("--device-torch", default="auto")
    sw.add_argument("--max-steps", type=int, default=0)
    sw.add_argument("--no-train", action="store_true", help="물질화와 예산까지만")
    sw.add_argument("--cache-dir", default=".cache")
    sw.set_defaults(func=cmd_sweep)

    ig = sub.add_parser("infer-graph", help="학습 그래프에서 정답 경로를 잘라낸 추론 그래프를 뽑는다")
    ig.add_argument("spec")
    ig.add_argument("--set", action="append", default=[])
    ig.add_argument("--out", default="")
    ig.set_defaults(func=cmd_infer_graph)

    tr = sub.add_parser("train", help="물질화된 shard로 다단계 학습을 실행한다")
    tr.add_argument("spec")
    tr.add_argument("--materialized", default="", help="기본값 runs/<run_id>/materialized")
    tr.add_argument("--set", action="append", default=[])
    tr.add_argument("--device", default="", help="예산 프로파일")
    tr.add_argument("--device-torch", default="auto", help="cuda | cpu | auto")
    tr.add_argument("--resume", action="store_true")
    tr.add_argument("--max-steps", type=int, default=0)
    tr.add_argument("--skip-budget", action="store_true")
    tr.add_argument("--what-if", action="append", default=[])
    tr.add_argument("--cache-dir", default=".cache")
    tr.add_argument("--run-id", default="")
    tr.set_defaults(func=cmd_train)

    pv = sub.add_parser("preview", help="노드 하나만 실행해 시각화 출력을 본다")
    pv.add_argument("spec")
    pv.add_argument("--node", required=True)
    pv.add_argument("--sample", default="")
    pv.add_argument("--set", action="append", default=[])
    pv.add_argument("--cache-dir", default=".cache")
    pv.add_argument("--no-cache", action="store_true")
    pv.add_argument("--save", action="store_true", help="이미지 미리보기를 runs/<id>/previews에 저장")
    pv.add_argument("--run-id", default="")
    pv.set_defaults(func=cmd_preview)
    return p


def _subparsers(parser: argparse.ArgumentParser) -> List[argparse.ArgumentParser]:
    out: List[argparse.ArgumentParser] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            out.extend(action.choices.values())
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    for p in _subparsers(parser):
        if any(x.dest == "spec" for x in p._actions) and not any(x.dest == "recipe" for x in p._actions):
            p.add_argument("--recipe", default="", help="Parameter Recipe 번호로 값 오버레이")
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CompileFailed as e:
        print(str(e), file=sys.stderr)
        return 2
    except VlmtError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
