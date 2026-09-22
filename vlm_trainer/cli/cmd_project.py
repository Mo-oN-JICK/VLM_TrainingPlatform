"""프로젝트를 만들고 고치는 명령 — new · edit · recipe

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

from .common import _load_nodes
from ..core.compiler import compile_project
from ..core.compiler import current_value
from ..spec import recipe as recipe_mod
from ..engine import sweep as sweep_mod

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
    print("  샘플 인덱스: project.yaml의 sample_space.index가 가리키는 JSONL을 먼저 만들어라")
    if a.edit:
        from ..ui import app as app_mod

        return app_mod.launch(proj, extra_modules=tuple(a.nodes))
    return 0


def cmd_edit(a: argparse.Namespace) -> int:
    """그래프 편집기를 연다. 창 하나로 뜨는 네이티브 앱이다.

    이름을 `edit` 그대로 둔 이유: `editor.bat`, `new-project.bat --edit`, 문서가 전부
    이 이름을 쓴다. 안이 웹에서 Qt 로 바뀐 것은 부르는 쪽이 알 필요가 없다.
    """
    from ..ui import app as app_mod

    _load_nodes(a.nodes)
    return app_mod.launch(a.spec, extra_modules=tuple(a.nodes))


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


