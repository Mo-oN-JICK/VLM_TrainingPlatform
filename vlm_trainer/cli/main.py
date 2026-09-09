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
from typing import Any, Dict, List, Optional

from ..core import registry
from ..core.compiler import CompileFailed, canonical_view, compile_project
from ..core.errors import VlmtError
from ..core.node import NodeKind
from ..engine import dryrun as dryrun_mod
from ..engine import preview as preview_mod
from ..engine import samples as samples_mod
from ..engine.runner import RunOptions, ancestors, execute
from ..spec.decompile import decompile, dump_yaml


def _load_nodes(modules: List[str]) -> None:
    registry.load_builtin_nodes()
    for m in modules or []:
        importlib.import_module(m)


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
    cg = compile_project(a.spec, recipe_overrides=_overrides(a.set))
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


def _run_id(a: argparse.Namespace) -> str:
    import time

    return getattr(a, "run_id", "") or time.strftime("%Y%m%dT%H%M%S")


def cmd_dryrun(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_overrides(a.set))
    space = _space(a, cg)
    opts = RunOptions(run_id=_run_id(a), extra_modules=tuple(a.nodes), cache_dir=a.cache_dir)
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
    cg = compile_project(a.spec, recipe_overrides=_overrides(a.set))
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
    )
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
    if rep.aborted:
        print()
        print(f"중단: {rep.aborted}")
        return 3
    return 0


def cmd_preview(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    cg = compile_project(a.spec, recipe_overrides=_overrides(a.set))
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
                      use_cache=not a.no_cache)
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
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--run-id", default="")
    r.set_defaults(func=cmd_run)

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


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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
