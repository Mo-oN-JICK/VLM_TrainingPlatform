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
    r = subprocess.run([PY, "probe.py"], cwd=dirpath, capture_output=True,
                       text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONPATH": dirpath + os.pathsep + ROOT})
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
