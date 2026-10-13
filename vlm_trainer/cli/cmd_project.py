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
    print("그래프는 비어 있습니다. 노드를 놓기 전에는 컴파일되지 않습니다 — Output이 하나는 있어야 합니다.")
    print(f"  편집기: vlmt edit {proj}")
    print("  샘플 인덱스: project.yaml의 sample_space.index가 가리키는 JSONL을 먼저 만드세요")
    if a.edit:
        from ..ui import app as app_mod

        return app_mod.launch(proj, extra_modules=tuple(a.nodes))
    return 0


def cmd_new_line(a: argparse.Namespace) -> int:
    """빈 5노드 프로젝트를 만든다. 표와 정답 항목은 화면의 `데이터` 에서 넣는다."""
    from ..spec import scaffold

    path, made = scaffold.new_line(a.dir, line_id=a.id, name=a.name)
    for f in made:
        print(f"  만듦: {f}")
    print(f"\n  편집기: vlmt edit {path}   (열면 `데이터` 가 먼저 뜹니다)")
    if a.edit:
        from ..ui import app as app_mod

        return app_mod.launch(path)
    return 0


def cmd_edit(a: argparse.Namespace) -> int:
    """그래프 편집기를 연다. 창 하나로 뜨는 네이티브 앱이다.

    이름을 `edit` 그대로 둔 이유: `editor.bat`, `new-project.bat --edit`, 문서가 전부
    이 이름을 쓴다. 안이 웹에서 Qt 로 바뀐 것은 부르는 쪽이 알 필요가 없다.
    """
    from ..ui import app as app_mod

    _load_nodes(a.nodes)
    return app_mod.launch(a.spec, extra_modules=tuple(a.nodes), canvas=getattr(a, "canvas", False))


def cmd_table_template(a: argparse.Namespace) -> int:
    """사용자가 채울 빈 표(CSV)의 첫 줄을 만든다."""
    from ..spec import table

    cols = table.header(key=a.key, image=a.image, question=a.question,
                        answers=[c.strip() for c in a.answers.split(",") if c.strip()],
                        split=a.split, group=a.group)
    with open(a.out, "wb") as fh:
        fh.write(table.template_csv(cols))
    print(f"표 양식 -> {a.out}")
    print(f"  열: {', '.join(cols)}")
    return 0


def cmd_schema_build(a: argparse.Namespace) -> int:
    """정답 형태 표(CSV: 항목,이름,종류,허용값)에서 스키마 파일을 만든다."""
    from ..answer import table

    schema = table.build(table.read_rows(a.rows), schema_id=a.id, render=a.render, max_tokens=a.max_tokens)
    text = table.dump(schema)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"정답 형태 -> {a.out} (항목 {len(schema['steps'])}개)")
    else:
        sys.stdout.write(text)
    return 0


def cmd_schema_rows(a: argparse.Namespace) -> int:
    """스키마 파일을 정답 형태 표(CSV)로 되돌린다. 표로 나타낼 수 없는 항목이 있으면 알린다."""
    from ..answer import table

    got = table.rows_from_schema(a.schema)
    data = table.rows_csv(got["rows"])
    if a.out:
        with open(a.out, "wb") as fh:
            fh.write(data)
        print(f"정답 형태 표 -> {a.out} (항목 {len(got['rows'])}개)")
    else:
        sys.stdout.write(data.decode("utf-8-sig"))
    if got["unsupported"]:
        print("표로 나타낼 수 없는 것: " + ", ".join(got["unsupported"]) + " — 표로 고쳐 저장하면 사라집니다", file=sys.stderr)
    return 0


def cmd_new_node(a: argparse.Namespace) -> int:
    """노드 하나의 뼈대. 빈 파일부터 시작하면 대부분 한두 가지를 빠뜨린다."""
    from ..spec import node_scaffold as ns

    made = ns.new_node(a.path, a.type, a.kind, category=a.category, label=a.label,
                       root=a.root or "")
    print(f"만들었습니다: {made.module_path}")
    print(f"          {made.test_path}")
    print()
    print("다음 두 가지:")
    print(f"  1) 스펙에 적습니다 — node_modules: [{made.module_name}]")
    print(f"  2) 검사합니다      — vlmt check-node {made.module_name}")
    return 0


def cmd_check_node(a: argparse.Namespace) -> int:
    """등록 검사가 보지 못하는 것을 실제로 돌려 본다."""
    from ..spec import node_scaffold as ns

    rep = ns.check_module(a.module, root=a.root or os.getcwd())
    print(ns.render(rep))
    return 0 if rep.ok else 1


def cmd_recipe(a: argparse.Namespace) -> int:
    _load_nodes(a.nodes)
    book = recipe_mod.load(a.spec)
    if not book.recipes and a.action != "expand":
        print(f"레시피가 없습니다: {book.path}")
        return 0

    if a.action == "list":
        cg = compile_project(a.spec)
        paths = {p for r in book.recipes.values() for p in r.overrides}
        st = recipe_mod.status(book, {p: current_value(cg, p) for p in paths})
        print(f"{book.path}")
        print(f"  활성: {st if st else '없음'}"
              + ("  (프로젝트를 직접 수정해 레시피와 어긋납니다)" if st == recipe_mod.CUSTOMIZED else ""))
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




def _run_dir(run: str) -> str:
    """실행 id 또는 폴더 경로."""
    return run if os.path.isdir(run) else os.path.join("runs", run)


def cmd_judge(a: argparse.Namespace) -> int:
    """검증 건 하나에 사람 판정(ok | ng | none)을 남긴다."""
    from ..answer import judge

    judge.judge(_run_dir(a.run), a.sample_id, a.verdict)
    print(f"{a.sample_id}: {a.verdict}")
    return 0


def cmd_judgments(a: argparse.Namespace) -> int:
    """검증 건마다 항목별 O/X 와 사람 판정을 보인다. 점수 합계는 내지 않는다."""
    from ..answer import judge

    got = judge.review(_run_dir(a.run))
    for it in got["items"]:
        mark = "".join("O" if f["match"] else "X" for f in it["fields"])
        flag = " 빈 답" if it["empty"] else (" 형식 어긋남" if it["broken"] else "")
        flag += "".join(f" 범위 밖:{f['step']}" for f in it["fields"] if f["out_of_range"])
        print(f"{it['id']}	{mark}	{it['verdict']}{flag}")
    c = got["count"]
    print(f"OK {c['ok']} · NG {c['ng']} · 미판정 {c['none']}  (항목 순서: {', '.join(got['steps'])})")
    return 0


def _table_of(spec: str) -> dict:
    import yaml

    from ..engine import table_edit

    with open(spec, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return table_edit.target(data, os.path.dirname(os.path.abspath(spec)))


def cmd_row_set(a: argparse.Namespace) -> int:
    """표(CSV)의 한 건의 칸을 고친다. 처음 고치면 `표.bak.csv` 로 원본을 남긴다."""
    from ..core.errors import SpecError
    from ..engine import table_edit

    edits = {}
    for kv in a.edits:
        col, sep, val = kv.partition("=")
        if not sep:
            raise SpecError(f"{kv!r}를 읽을 수 없습니다.\n  안 잡혔다면: 어느 열을 무슨 값으로 고칠지 알 수 없습니다.\n  열이름=값 꼴로 적으세요.")
        edits[col.strip()] = val
    t = _table_of(a.spec)
    changed = table_edit.set_cells(t["path"], t["key"], a.sample_id, edits)
    print(f"{a.sample_id}: " + (", ".join(changed) + " 고침" if changed else "바뀐 것 없음"))
    return 0


def cmd_row_exclude(a: argparse.Namespace) -> int:
    """한 건을 표에서 제외한다(`제외` 열에 x). 지우지 않고, --undo 로 되살린다."""
    from ..engine import table_edit

    t = _table_of(a.spec)
    changed = table_edit.set_excluded(t["path"], t["key"], a.sample_id, not a.undo)
    print(f"{a.sample_id}: " + (("되살림" if a.undo else "제외함") if changed else "바뀐 것 없음"))
    return 0
