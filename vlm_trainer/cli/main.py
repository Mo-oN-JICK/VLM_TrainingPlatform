"""vlmt — 커맨드라인 진입점.

UI 없이 스펙만으로 전부 할 수 있어야 한다. UI 전용 실행 경로는 존재하지 않는다.
Phase 1 범위: compile / decompile / nodes / show.
dryrun · budget · materialize · run · preview는 Phase 2 이후에 붙는다.
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
