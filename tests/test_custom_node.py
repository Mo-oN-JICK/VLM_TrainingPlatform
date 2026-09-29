"""저장소 밖에서 온 노드 — 캐시가 옛 결과를 조용히 돌려주지 않게 한다.

캐시는 노드 **이름**(`type@version`)으로 옛 결과를 찾는다. 내장 노드는 코드를 고칠 때
`ENGINE_ABI` 를 올려 캐시를 통째로 버리지만, 남이 만든 노드에는 그 레버가 없다.
버전을 안 올리고 코드만 고치면 **캐시가 옛 결과를 돌려준다.** 고친 사람에게는
"코드를 바꿨는데 결과가 안 변한다" 로 보이고, 예외도 경고도 없다.

커스텀 노드를 열기 전에 이것부터 막아야 한다. 안 그러면 이 도구가 막으려고 만든 바로
그 실패(조용히 틀린 채 계속 돈다)를 우리가 직접 만드는 꼴이 된다.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")

# 저장소 밖 노드 한 개. `%s` 자리에 코드를 더 넣어 "고친 판" 을 만든다.
NODE = '''from dataclasses import dataclass

from vlm_trainer.core.node import Node, NodeDoc, NodeKind, Port, RunCtx
from vlm_trainer.core.registry import register
from vlm_trainer.core.types import text


@dataclass
class P:
    suffix: str = "!"


@register(type="my.tag", version="1.0.0", category="Prompt Assembly",
          kind=NodeKind.PROCESSING,
          inputs={"t": Port(text("prompt"), "in")},
          outputs={"t": Port(text("prompt"), "out")},
          params=P, doc=NodeDoc(label="tag"))
class Tag(Node):
    def run(self, ctx: RunCtx, params, **inputs):
        return {"t": str(inputs["t"]) + params.suffix%s}
'''


def _write_node(dirpath, extra: str = "") -> None:
    io.open(os.path.join(dirpath, "mynode.py"), "w", encoding="utf-8",
            newline="\n").write(NODE % extra)


def _probe(dirpath, script: str) -> str:
    """별도 프로세스에서 돌린다 — 레지스트리가 프로세스마다 새로 채워져야
    "코드를 고치고 다시 켰다" 를 흉내 낼 수 있다."""
    p = os.path.join(dirpath, "probe.py")
    io.open(p, "w", encoding="utf-8", newline="\n").write(textwrap.dedent(script))
    # `PYTHONUTF8` 이 없으면 자식이 한국어 오류를 cp949 로 쓰고, 읽는 쪽이
    # UnicodeDecodeError 로 죽는다 — 정작 보려던 메시지가 사라진다.
    env = {**os.environ, "PYTHONPATH": dirpath + os.pathsep + ROOT, "PYTHONUTF8": "1"}
    r = subprocess.run([PY, "probe.py"], cwd=dirpath, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip()


@pytest.fixture
def work(tmp_path):
    _write_node(str(tmp_path))
    return str(tmp_path)


# ── 지문 ────────────────────────────────────────────────────────────────
def test_a_builtin_node_carries_no_source_fingerprint():
    """내장 노드는 `ENGINE_ABI` 가 그 역할을 한다. 매 실행마다 소스를 읽을 이유가 없고,
    여기에 지문이 붙으면 기존 캐시가 전부 무효가 된다."""
    from vlm_trainer.core import registry

    registry.load_builtin_nodes()
    assert registry.resolve("infer.vlm@1.0.0").impl_fingerprint == ""


def test_a_node_from_outside_the_repo_carries_one(work):
    got = _probe(work, """
        import sys
        from vlm_trainer.core import registry
        registry.load_builtin_nodes()
        import mynode  # noqa
        print(registry.resolve("my.tag@1.0.0").impl_fingerprint)
    """)
    assert got and got != "", "저장소 밖 노드에 지문이 없다"


def test_changing_the_code_changes_the_fingerprint_without_a_version_bump(work):
    """**버전은 1.0.0 그대로다.** 사람이 올리는 것을 기다리면 잊는다."""
    script = """
        from vlm_trainer.core import registry
        registry.load_builtin_nodes()
        import mynode  # noqa
        print(registry.resolve("my.tag@1.0.0").impl_fingerprint)
    """
    before = _probe(work, script)
    _write_node(work, " + params.suffix")
    after = _probe(work, script)
    assert before != after, "코드를 고쳤는데 지문이 그대로다 — 캐시가 옛 결과를 준다"


def test_an_unreadable_source_is_recorded_as_unreadable_not_as_builtin():
    """소스를 못 읽는 경우(대화형 세션, 압축 배포)를 조용히 내장 노드처럼 취급하면
    딱 이 위험이 되살아난다. 못 읽었다는 사실 자체를 지문에 남긴다."""
    from vlm_trainer.core.registry import _source_fingerprint

    class _NoSource:
        __module__ = "somewhere_else"
        __qualname__ = "_NoSource"

    got = _source_fingerprint(_NoSource)
    assert got.startswith("unreadable:"), got


# ── 캐시 키 ─────────────────────────────────────────────────────────────
def test_the_cache_key_and_everything_downstream_changes(work):
    """노드 자신만 다시 계산하면 부족하다. 그 값을 받은 뒤쪽도 전부 다시 계산돼야 한다."""
    import shutil

    shutil.copytree(os.path.join(ROOT, "solutions"), os.path.join(work, "solutions"))
    script = """
        import io, yaml
        from vlm_trainer.core import registry
        from vlm_trainer.core.compiler import compile_project
        registry.load_builtin_nodes()
        import mynode  # noqa

        spec = "solutions/vlm_open/projects/01_open/project.yaml"
        d = yaml.safe_load(io.open(spec, encoding="utf-8"))
        d["nodes"].append({"id": "n_tag", "type": "my.tag@1.0.0", "params": {}})
        d["edges"] = [e for e in d["edges"]
                      if not (e["from"] == "n_guard:prompt" and e["to"] == "n_sample:prompt")]
        d["edges"] += [{"from": "n_guard:prompt", "to": "n_tag:t"},
                       {"from": "n_tag:t", "to": "n_sample:prompt"}]
        p2 = "solutions/vlm_open/projects/01_open/_probe.yaml"
        io.open(p2, "w", encoding="utf-8").write(yaml.safe_dump(d, allow_unicode=True))

        cg = compile_project(p2)
        print(cg.nodes["n_tag"].cache_key)
        print(cg.nodes["n_sample"].cache_key)
        print(cg.spec_hash)
    """
    a = _probe(work, script).split("\n")
    _write_node(work, " + params.suffix")
    b = _probe(work, script).split("\n")

    assert a[0] != b[0], "노드 자신의 캐시 키가 그대로다"
    assert a[1] != b[1], "하류 노드의 캐시 키가 그대로다 — 뒤쪽이 옛 값을 재사용한다"
    # 스펙은 한 글자도 안 바뀌었다. `spec_hash` 가 이것을 못 보는 것이 정상이고,
    # 그래서 캐시 키와 `bake_key` 가 따로 필요하다.
    assert a[2] == b[2], "스펙이 안 바뀌었는데 spec_hash 가 달라졌다"


# ── 재개 ────────────────────────────────────────────────────────────────
def test_resume_is_refused_when_the_custom_node_changed(work):
    """`--resume` 은 `spec_hash` 만 봤다. 커스텀 노드를 고쳐도 스펙은 그대로라
    통과했고, **옛 코드로 구운 shard 와 새 코드로 구운 shard 가 한 학습에 섞였다.**"""
    import shutil

    shutil.copytree(os.path.join(ROOT, "solutions"), os.path.join(work, "solutions"))
    script = """
        import io, yaml
        from vlm_trainer.core import registry
        from vlm_trainer.core.compiler import compile_project
        from vlm_trainer.engine import materialize as mat
        registry.load_builtin_nodes()
        import mynode  # noqa

        spec = "solutions/vlm_open/projects/01_open/project.yaml"
        d = yaml.safe_load(io.open(spec, encoding="utf-8"))
        d["nodes"].append({"id": "n_tag", "type": "my.tag@1.0.0", "params": {}})
        d["edges"] = [e for e in d["edges"]
                      if not (e["from"] == "n_guard:prompt" and e["to"] == "n_sample:prompt")]
        d["edges"] += [{"from": "n_guard:prompt", "to": "n_tag:t"},
                       {"from": "n_tag:t", "to": "n_sample:prompt"}]
        p2 = "solutions/vlm_open/projects/01_open/_probe.yaml"
        io.open(p2, "w", encoding="utf-8").write(yaml.safe_dump(d, allow_unicode=True))
        print(mat.bake_key(compile_project(p2)))
    """
    before = _probe(work, script)
    _write_node(work, " + params.suffix")
    after = _probe(work, script)
    assert before != after, "굽는 방식이 달라졌는데 bake_key 가 그대로다"


def test_the_journal_records_the_bake_key_so_resume_can_compare():
    """기록하지 않으면 비교할 것이 없다. 옛 저널에는 없으므로 **있을 때만** 비교한다 —
    없다고 거부하면 이 변경 전에 구운 것이 전부 못 쓰게 된다."""
    import inspect

    from vlm_trainer.engine import materialize as mat
    from vlm_trainer.engine.journal import Replay

    assert hasattr(Replay(), "bake_key")
    src = inspect.getsource(mat.materialize)
    assert "bake_key=now_bake" in src, "저널에 bake_key 를 남기지 않는다"
    assert "replay.bake_key and" in src, "옛 저널(bake_key 없음)을 거부하고 있다"


def test_the_sweep_and_the_resume_check_use_the_same_key():
    """두 곳이 다르게 셈하면, 스윕이 공유한 bake 를 재개가 거부하거나 그 반대가 된다."""
    import inspect

    from vlm_trainer.engine import sweep

    assert "bake_key" in inspect.getsource(sweep.materialize_key)


# ── 스펙이 자기 노드를 데리고 다닌다 ────────────────────────────────────
#
# 전에는 `vlmt --nodes mymod ...` 플래그뿐이었다. 그래서 project.yaml 만 받은 사람은
# "노드를 찾을 수 없다" 로 막혔고, 무엇을 더 받아야 하는지 파일 어디에도 없었다.
# `editor.bat` 은 그 플래그를 안 넘기므로 **앱에는 커스텀 노드가 아예 안 보였다.**

SOLUTION_NODE = '''from dataclasses import dataclass

from vlm_trainer.core.node import Node, NodeDoc, NodeKind, Port, RunCtx
from vlm_trainer.core.registry import register
from vlm_trainer.core.types import text


@dataclass
class P:
    suffix: str = "!"


@register(type="my.tag", version="1.0.0", category="Prompt Assembly",
          kind=NodeKind.PROCESSING,
          inputs={"t": Port(text("prompt"), "in")},
          outputs={"t": Port(text("prompt"), "out")},
          params=P, doc=NodeDoc(label="꼬리표 붙이기"))
class Tag(Node):
    def run(self, ctx: RunCtx, params, **inputs):
        return {"t": str(inputs["t"]) + params.suffix}
'''


def _solution_with_custom_node(tmp_path, *, declare: bool = True, module: str = "my_nodes.crop"):
    """`<solution>/my_nodes/crop.py` 를 둔 과제 하나. 프로젝트 스펙이 그것을 선언한다."""
    import shutil

    import yaml

    shutil.copytree(os.path.join(ROOT, "solutions"), os.path.join(tmp_path, "solutions"))
    sol = os.path.join(tmp_path, "solutions", "vlm_open")
    os.makedirs(os.path.join(sol, "my_nodes"), exist_ok=True)
    io.open(os.path.join(sol, "my_nodes", "__init__.py"), "w").close()
    io.open(os.path.join(sol, "my_nodes", "crop.py"), "w", encoding="utf-8",
            newline="\n").write(SOLUTION_NODE)

    spec = os.path.join(sol, "projects", "01_open", "project.yaml")
    d = yaml.safe_load(io.open(spec, encoding="utf-8"))
    if declare:
        d["node_modules"] = [module]
    d["nodes"].append({"id": "n_tag", "type": "my.tag@1.0.0", "params": {}})
    d["edges"] = [e for e in d["edges"]
                  if not (e["from"] == "n_guard:prompt" and e["to"] == "n_sample:prompt")]
    d["edges"] += [{"from": "n_guard:prompt", "to": "n_tag:t"},
                   {"from": "n_tag:t", "to": "n_sample:prompt"}]
    io.open(spec, "w", encoding="utf-8").write(yaml.safe_dump(d, allow_unicode=True))
    return "solutions/vlm_open/projects/01_open/project.yaml"


def test_a_declared_module_is_imported_without_any_flag(tmp_path):
    """이것이 요점이다. 플래그 없이 컴파일된다."""
    rel = _solution_with_custom_node(str(tmp_path))
    out = _probe(str(tmp_path), f"""
        from vlm_trainer.core import registry
        from vlm_trainer.core.compiler import compile_project
        registry.load_builtin_nodes()
        cg = compile_project({rel!r})
        print(len(cg.nodes))
    """)
    assert int(out) == 15


def test_without_the_declaration_it_still_fails(tmp_path):
    """선언이 일을 하고 있다는 증거. 이것이 통과하면 위 테스트는 아무것도 안 본 것이다."""
    rel = _solution_with_custom_node(str(tmp_path), declare=False)
    with pytest.raises(AssertionError) as e:
        _probe(str(tmp_path), f"""
            from vlm_trainer.core import registry
            from vlm_trainer.core.compiler import compile_project
            registry.load_builtin_nodes()
            compile_project({rel!r})
        """)
    assert "my.tag" in str(e.value)


def test_a_missing_module_says_where_it_looked(tmp_path):
    """스펙 파일이 파이썬을 임포트한다. 실패를 조용히 넘기면 그다음 오류가
    "노드를 찾을 수 없다" 인데, 거기서는 무엇이 빠졌는지 안 나온다."""
    rel = _solution_with_custom_node(str(tmp_path), module="my_nodes.없는것")
    with pytest.raises(AssertionError) as e:
        _probe(str(tmp_path), f"""
            from vlm_trainer.core.compiler import compile_project
            compile_project({rel!r})
        """)
    msg = str(e.value)
    assert "임포트할 수 없다" in msg and "뒤진 곳" in msg


def test_the_declaration_survives_a_save(tmp_path):
    """저장 한 번에 선언이 사라지면 다음 컴파일이 막힌다.
    `p_prep:images -> p_prompt:images` 가 사라지던 것과 같은 부류다."""
    rel = _solution_with_custom_node(str(tmp_path))
    out = _probe(str(tmp_path), f"""
        import io, yaml
        from vlm_trainer.core import registry
        from vlm_trainer.ui.api import Editor
        registry.load_builtin_nodes()
        ed = Editor.open({rel!r})
        ed.save()
        print(yaml.safe_load(io.open({rel!r}, encoding="utf-8")).get("node_modules"))
    """)
    assert "my_nodes.crop" in out


def test_the_app_library_lists_the_custom_node(tmp_path):
    """`editor.bat` 은 `--nodes` 를 넘기지 않는다. 선언이 없으면 앱에서 그 상자를
    끌어다 놓을 수가 없다."""
    rel = _solution_with_custom_node(str(tmp_path))
    out = _probe(str(tmp_path), f"""
        from vlm_trainer.core import registry
        from vlm_trainer.ui.api import Editor
        registry.load_builtin_nodes()
        ed = Editor.open({rel!r})
        print(any(n["type"].startswith("my.tag") for n in ed.library()))
    """)
    assert out == "True"


def test_the_solution_root_is_found_by_structure_not_by_a_marker_file(tmp_path):
    """`solution.yaml` 의 존재로 판단하면 안 된다 — 코드가 그 파일을 읽지도 않고,
    `vlm_open` 에는 아예 없다. 없는 표지를 기준으로 삼으면 멀쩡한 과제에서 조용히
    못 찾는다."""
    from vlm_trainer.spec.loader import solution_root

    d = os.path.join("X", "solutions", "vlm_open", "projects", "01_open")
    assert solution_root(d).endswith(os.path.join("solutions", "vlm_open"))
    assert not os.path.exists(os.path.join(ROOT, "solutions", "vlm_open", "solution.yaml"))


def test_an_empty_declaration_does_not_move_the_spec_hash():
    """`node_modules` 를 안 쓰는 스펙의 `spec_hash` 가 이 변경으로 달라지면,
    구워 둔 것과 재개가 통째로 무효가 된다."""
    from vlm_trainer.core import registry
    from vlm_trainer.core.compiler import canonical_view, compile_project

    registry.load_builtin_nodes()
    cg = compile_project(os.path.join(ROOT, "solutions", "vlm_open",
                                      "projects", "01_open", "project.yaml"))
    assert not cg.node_modules
    assert "node_modules" not in canonical_view(cg)


# ── 뼈대와 검사 ─────────────────────────────────────────────────────────
#
# 빈 파일부터 시작하면 대부분 한두 가지를 빠뜨린다. 빠뜨린 것이 등록에서 걸리면 다행이고,
# 안 걸리면 조용히 틀린 결과가 나온다.

@pytest.mark.parametrize("kind", ["input", "processing", "output"])
def test_a_fresh_node_passes_its_own_checks(tmp_path, kind):
    """만들자마자 통과해야 한다. 뼈대가 등록 거부부터 만나게 하면 아무도 안 쓴다."""
    from vlm_trainer.spec import node_scaffold as ns

    made = ns.new_node(str(tmp_path / "nodes" / f"{kind}.py"), f"my.{kind}", kind,
                       label=kind, root=str(tmp_path))
    assert os.path.exists(made.module_path) and os.path.exists(made.test_path)
    assert made.module_name == f"nodes.{kind}"

    out = _probe(str(tmp_path), f"""
        from vlm_trainer.spec import node_scaffold as ns
        rep = ns.check_module("nodes.{kind}", root={str(tmp_path)!r})
        print(rep.ok)
        print("|".join(c.name for c in rep.checks if not c.ok))
    """)
    ok, failed = out.split("\n")[0], out.split("\n")[1] if "\n" in out else ""
    assert ok == "True", f"갓 만든 노드가 검사에 걸린다: {failed}"


def test_the_generated_test_file_runs(tmp_path):
    """테스트 파일도 함께 만든다. 명령은 손으로 부르는 것이고 테스트는 잊지 않는다."""
    from vlm_trainer.spec import node_scaffold as ns

    ns.new_node(str(tmp_path / "nodes" / "crop.py"), "my.crop", "processing",
                label="자르기", root=str(tmp_path))
    r = subprocess.run([PY, "-m", "pytest", "-q"], cwd=str(tmp_path),
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONUTF8": "1",
                            "PYTHONPATH": str(tmp_path) + os.pathsep + ROOT})
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize(
    "name,mutate,expect",
    [
        ("순수하지 않다",
         ('return {"value": str(inputs["value"]) + params.example}',
          'import random\n        return {"value": str(random.random())}'),
         "두 번 돌려 같은 값"),
        ("선언과 다른 타입",
         ('return {"value": str(inputs["value"]) + params.example}',
          'return {"value": 12345}'),
         "선언한 타입대로"),
        ("선언한 포트를 안 낸다",
         ('return {"value": str(inputs["value"]) + params.example}', "return {}"),
         "선언한 타입대로"),
        ("이름이 비었다", ('label="t",', 'label="",'), "사람이 읽을 이름"),
        ("실행 중 터진다",
         ('return {"value": str(inputs["value"]) + params.example}',
          'raise ValueError("터짐")'),
         "실행"),
    ],
)
def test_the_check_actually_catches_a_broken_node(tmp_path, name, mutate, expect):
    """통과만 하는 검사는 검사가 아니다. **등록만으로는 알 수 없는 것**들이다 —
    정말 순수한가, 정말 선언한 타입을 내는가, 정말 도는가."""
    from vlm_trainer.spec import node_scaffold as ns

    made = ns.new_node(str(tmp_path / "nodes" / "t.py"), "my.t", "processing",
                       label="t", root=str(tmp_path))
    src = io.open(made.module_path, encoding="utf-8").read()
    old, new = mutate
    assert old in src, "뼈대가 바뀌어 이 시험이 낡았다"
    io.open(made.module_path, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))

    out = _probe(str(tmp_path), f"""
        from vlm_trainer.spec import node_scaffold as ns
        rep = ns.check_module("nodes.t", root={str(tmp_path)!r})
        print(rep.ok)
        print("|".join(c.name for c in rep.checks if not c.ok))
    """)
    ok, failed = (out.split("\n") + [""])[:2]
    assert ok == "False", f"{name}: 검사가 통과시켰다"
    assert expect in failed, f"{name}: 다른 것이 걸렸다 — {failed}"


def test_the_check_refuses_to_invent_values_it_cannot_build(tmp_path):
    """표본 값을 지어낼 수 없는 포트 타입이면 실행 검사를 **건너뛴다**.
    지어낸 값으로 통과시키면 검사가 거짓말을 한다."""
    from vlm_trainer.core.types import ANY, image, regions, text
    from vlm_trainer.spec import node_scaffold as ns

    assert ns.sample_value(text()) == "샘플"
    assert ns.sample_value(image(frame=ANY)) is not None
    assert ns.sample_value(regions()) is None, "만들 수 없는 것을 만들어 냈다"


def test_a_node_placed_inside_the_package_is_flagged(tmp_path):
    """`vlm_trainer.` 아래에 두면 내장 노드로 취급되어 구현 지문이 안 붙는다 —
    코드를 고쳐도 캐시가 옛 결과를 돌려준다."""
    from vlm_trainer.core.registry import _source_fingerprint

    class _Inside:
        __module__ = "vlm_trainer.nodes.mine"
        __qualname__ = "_Inside"

    assert _source_fingerprint(_Inside) == ""


def test_new_node_refuses_a_bad_type_name(tmp_path):
    from vlm_trainer.core.errors import SpecError
    from vlm_trainer.spec import node_scaffold as ns

    with pytest.raises(SpecError, match="범주.이름"):
        ns.new_node(str(tmp_path / "x.py"), "crop", "processing")
    with pytest.raises(SpecError, match="kind"):
        ns.new_node(str(tmp_path / "y.py"), "my.crop", "sideways")


def test_new_node_does_not_overwrite(tmp_path):
    """덮어쓰면 고쳐 둔 노드가 날아간다."""
    from vlm_trainer.core.errors import SpecError
    from vlm_trainer.spec import node_scaffold as ns

    ns.new_node(str(tmp_path / "a.py"), "my.a", "processing")
    with pytest.raises(SpecError, match="이미 있는 파일"):
        ns.new_node(str(tmp_path / "a.py"), "my.a", "processing")
