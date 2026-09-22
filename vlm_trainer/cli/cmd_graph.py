"""그래프를 읽고 보여 주는 명령 — compile · decompile · nodes · show · view

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

from .common import _load_nodes, _overrides, _recipe_overrides
from ..core.node import NodeKind
from ..core.compiler import canonical_view
from ..core.compiler import compile_project
from ..train import contract as contract_mod
from ..spec.decompile import decompile
from ..spec.decompile import dump_yaml
from ..core import registry

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


def cmd_backbones(a: argparse.Namespace) -> int:
    from ..plugins.base import all_backbones, resolve_backbone

    _load_nodes(a.nodes)
    if a.fetch:
        # 가중치는 시켜야 받는다. `--add`는 config.json만 읽어 형상을 답하고(예산 게이트가
        # 쓰는 길), 실제 GB가 디스크에 떨어지는 것은 이 한 줄뿐이다.
        from ..plugins.hf_backbone import fetch as fetch_weights

        ref = a.fetch if a.fetch.startswith("hf:") else "hf:" + a.fetch
        print(f"내려받는 중: {ref}  (수 GB. 한 번 받으면 캐시에 남는다)")
        where = fetch_weights(ref)
        print(f"  받음: {where}")
        a.add = a.add or a.fetch
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


