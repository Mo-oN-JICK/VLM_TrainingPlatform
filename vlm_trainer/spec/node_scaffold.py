"""새 노드의 뼈대와, 그 노드가 계약을 지키는지 보는 검사.

빈 파일부터 시작하면 대부분 한두 가지를 빠뜨린다. 빠뜨린 것이 등록에서 걸리면 다행이고,
안 걸리면 조용히 틀린 결과가 나온다 — 이 저장소가 막으려고 만든 바로 그 실패다.

그래서 둘을 함께 둔다. **뼈대는 빠뜨릴 자리를 미리 채워 주고, 검사는 등록만으로는
알 수 없는 것을 실제로 돌려 본다.**
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.errors import SpecError

KINDS = ("input", "processing", "output")

# 3분류마다 포트 모양이 정해져 있다(설계 문서 02 §2.4). 뼈대가 그것을 지킨 채로 나와야
# 처음 만드는 사람이 등록 거부부터 만나지 않는다.
_PORTS = {
    "input": ("", '        "value": Port(text(), "내보낼 값"),'),
    "processing": ('        "value": Port(text(), "받을 값"),', '        "value": Port(text(), "내보낼 값"),'),
    "output": ('        "value": Port(text(), "받을 값"),', ""),
}

TEMPLATE = '''"""{label} — {type_}

이 노드가 무엇을 하는지 한 문단으로 적는다. **무엇을 하는가**보다 **왜 이 노드가 따로
있어야 하는가**를 적는 편이 나중에 읽는 사람에게 쓸모 있다.
"""

from dataclasses import dataclass
from typing import Any, Dict

from vlm_trainer.core.node import Node, NodeDoc, NodeError, NodeKind, Port, RunCtx
from vlm_trainer.core.registry import register
from vlm_trainer.core.types import text


@dataclass
class {cls}Params:
    """파라미터는 **dataclass 여야 한다**(계약 C3). 앱이 이 선언을 보고 설정 화면을
    자동으로 만든다 — 선언이 없으면 화면에 아무것도 안 뜬다."""

    example: str = "!"


@register(
    type="{type_}",
    version="1.0.0",
    category="{category}",
    kind=NodeKind.{kind_const},
    {ports}
    params={cls}Params,
    recipe_overridable=["example"],
    doc=NodeDoc(
        label="{label}",
        hint="이 상자가 무슨 일을 하는지 한 줄. **이 도구를 처음 보는 사람** 기준으로.",
        summary="정확한 참조 문장. 용어를 써도 된다.",
        scenario="언제 쓰는가. 안 쓰면 무엇이 곤란한가.",
    ),
)
class {cls}(Node):
    """**구현은 모듈 최상위에 둔다.** 함수 안에 정의하면 Windows spawn 워커가
    임포트할 수 없고, 등록 시점에 거부된다."""
{fingerprint}
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        # 같은 입력에는 같은 출력을 낸다. 파일·네트워크·전역 난수·시계에 손대지 않는다.
        # 무작위가 필요하면 `ctx.rng()` 를 쓰고 `deterministic=False` 를 선언한다.
        #
        # 실패는 `NodeError` 로 던진다. 네 가지를 담는다 —
        # 불일치 필드 / 안 잡혔다면 언제 어디서 터졌을지 / 추정 낭비 / 구체적 해결 배선.
{body}
'''

_FINGERPRINT = '''
    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        """Input 은 이것을 구현해야 한다(계약 C4). 바깥 세계가 바뀌었는지 알려 주는
        유일한 통로다 — 없으면 원본 파일이 바뀌어도 캐시가 옛 값을 돌려준다."""
        return f"{params.example}"
'''

_BODY = {
    "input": '        return {"value": params.example}',
    "processing": '        return {"value": str(inputs["value"]) + params.example}',
    "output": (
        '        # Output 은 **부작용의 자리**다. 파일을 쓰거나 밖으로 보낸다.\n'
        '        # 값을 내보내야 하면 출력 포트를 선언해도 된다 — 반드시 끝일 필요는 없다.\n'
        '        return {}'
    ),
}

TEST_TEMPLATE = '''"""{type_} 이 계약을 지키는지.

`vlmt check-node {module}` 이 보는 것과 같은 성질을 여기에 고정해 둔다. 명령은 손으로
부르는 것이고, 테스트는 잊지 않는다.
"""

from vlm_trainer.core import registry
from vlm_trainer.core.node import RunCtx


def _node():
    registry.load_builtin_nodes()
    import {module}  # noqa: F401  — 등록 부작용. **지우지 마라**

    return registry.resolve("{type_}@1.0.0")


def test_it_registers():
    d = _node()
    assert d.doc.label, "카드에 뜰 이름이 없다"


def test_same_input_gives_the_same_output():
    """순수하지 않으면 캐시가 거짓말을 한다."""
    d = _node()
    ctx = RunCtx(run_id="t", node_id="n1")
    p = d.build_params(d.default_params())
    {call}
    assert first == second
'''


@dataclass
class NodeFiles:
    module_path: str = ""
    test_path: str = ""
    module_name: str = ""
    type_: str = ""


def _class_name(type_: str) -> str:
    tail = type_.split(".")[-1]
    return "".join(p.capitalize() or "_" for p in tail.split("_")) or "MyNode"


def module_name_for(path: str, root: str) -> str:
    """파일 경로를 임포트 이름으로. `node_modules` 에 적을 값이 이것이다."""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    stem = rel[:-3] if rel.endswith(".py") else rel
    return stem.replace(os.sep, ".").replace("/", ".")


def new_node(path: str, type_: str, kind: str, *, category: str = "Data Processing",
             label: str = "", root: str = "") -> NodeFiles:
    """노드 하나와 그 테스트 파일을 만든다. 이미 있으면 덮지 않는다."""
    kind = (kind or "processing").lower()
    if kind not in KINDS:
        raise SpecError(f"kind 는 {', '.join(KINDS)} 중 하나여야 한다 (받은 값: {kind!r})")
    if "." not in type_:
        raise SpecError(
            f"노드 타입은 `범주.이름` 꼴이어야 한다 (받은 값: {type_!r}, 예: my.crop_by_rule)")

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise SpecError(f"이미 있는 파일이다: {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    cls = _class_name(type_)
    src_in, src_out = _PORTS[kind]
    ports = "\n".join(
        x for x in (
            f"    inputs={{\n{src_in}\n    }}," if src_in else "",
            f"    outputs={{\n{src_out}\n    }}," if src_out else "",
        ) if x
    ).lstrip()

    body = TEMPLATE.format(
        label=label or cls, type_=type_, cls=cls, category=category,
        kind_const=kind.upper(), ports=ports,
        fingerprint=_FINGERPRINT if kind == "input" else "",
        body=_BODY[kind],
    )
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)

    root = root or os.path.dirname(path)
    module = module_name_for(path, root)
    test_path = os.path.join(os.path.dirname(path), f"test_{os.path.basename(path)}")
    call = (
        'first = d.impl().run(ctx, p, value="x")["value"]\n'
        '    second = d.impl().run(ctx, p, value="x")["value"]'
        if kind != "input" else
        'first = d.impl().run(ctx, p)["value"]\n    second = d.impl().run(ctx, p)["value"]'
    )
    if kind == "output":
        call = ('first = d.impl().run(ctx, p, value="x")\n'
                '    second = d.impl().run(ctx, p, value="x")')
    if not os.path.exists(test_path):
        with open(test_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(TEST_TEMPLATE.format(type_=type_, module=module, call=call))

    return NodeFiles(module_path=path, test_path=test_path, module_name=module, type_=type_)


# ── 검사 ────────────────────────────────────────────────────────────────
@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class CheckReport:
    module: str = ""
    types: List[str] = field(default_factory=list)
    checks: List[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def sample_value(t: Any) -> Any:
    """선언한 포트 타입에 맞는 **최소한의 값**. 검사에 쓸 입력을 만든다.

    실제 데이터가 아니다 — 노드가 선언한 모양을 지키는지 보려는 것이고, 그러려면
    무엇이든 타입에 맞는 값 하나가 필요하다. 만들 수 없는 타입이면 `None` 을 돌려
    그 노드의 실행 검사는 건너뛴다. **지어낸 값으로 통과시키지 않는다.**
    """
    from ..core.types import BaseKind

    if t.list_of is not None:
        inner = sample_value(t.list_of.item) if hasattr(t.list_of, "item") else None
        return [inner] if inner is not None else None

    base = t.base
    if base is BaseKind.TEXT:
        return "샘플"
    if base is BaseKind.TABLE:
        return {"col": "값"}
    if base is BaseKind.SAMPLE:
        return {"prompt": "샘플", "answer": "샘플"}
    if base is BaseKind.MODEL:
        return {"dir": ".", "backbone": "tiny-vlm"}
    if base is BaseKind.IMAGE:
        try:
            import numpy as np
        except ImportError:
            return None
        return np.zeros((8, 8, 3), dtype="uint8")
    if base is BaseKind.TOKENS:
        return [1, 2, 3]
    return None


def check_module(module: str, *, root: str = "") -> CheckReport:
    """모듈 하나를 임포트하고, 그 안에서 등록된 노드들을 실제로 돌려 본다.

    등록 검사(`@register`)는 **선언**만 본다. 여기서 보는 것은 선언만으로 알 수 없는
    세 가지다 — 정말 순수한가, 정말 선언한 타입을 내는가, 정말 다른 프로세스에서
    임포트되는가. 셋 다 틀려도 그래프는 돌고, 틀린 채로 돈다.
    """
    import importlib
    import sys

    from ..core import registry
    from ..core.node import NodeKind, RunCtx
    from ..engine import values as values_mod

    rep = CheckReport(module=module)
    registry.load_builtin_nodes()
    before = {(d.type, d.version) for d in registry.all_defs()}

    added = ""
    if root and root not in sys.path:
        sys.path.insert(0, root)
        added = root
    try:
        try:
            importlib.import_module(module)
        except Exception as e:                      # noqa: BLE001 — 무엇이든 그대로 보고한다
            rep.checks.append(Check("임포트", False, f"{type(e).__name__}: {e}"))
            return rep
    finally:
        if added and added in sys.path:
            sys.path.remove(added)
    rep.checks.append(Check("임포트", True, module))

    fresh = [d for d in registry.all_defs() if (d.type, d.version) not in before]
    rep.types = [d.ref for d in fresh]
    if not fresh:
        rep.checks.append(
            Check("노드 등록", False,
                  "이 모듈이 등록한 노드가 없다 — @register 데코레이터가 붙어 있는지 보라"))
        return rep
    rep.checks.append(Check("노드 등록", True, ", ".join(rep.types)))

    for d in fresh:
        rep.checks.append(Check(
            f"{d.type}: 저장소 밖 노드로 인식", bool(d.impl_fingerprint),
            "구현 지문이 있다 — 코드를 고치면 캐시가 갈린다" if d.impl_fingerprint
            else "지문이 없다. `vlm_trainer.` 아래에 두면 내장 노드로 취급되어, "
                 "코드를 고쳐도 캐시가 옛 결과를 돌려준다"))

        rep.checks.append(Check(
            f"{d.type}: 사람이 읽을 이름", bool(d.doc.label),
            d.doc.label or "NodeDoc(label=...) 이 비었다 — 카드에 타입 이름이 그대로 뜬다"))

        rep.checks.append(Check(
            f"{d.type}: spawn 워커가 임포트 가능", _spawn_ok(d.ref, module, root),
            "다른 프로세스에서 같은 노드를 찾았다"))

        inputs = {name: sample_value(p.type) for name, p in d.inputs.items()}
        if any(v is None for v in inputs.values()):
            rep.checks.append(Check(
                f"{d.type}: 실행", True,
                "건너뜀 — 이 포트 타입의 표본 값을 지어낼 수 없다. "
                "지어낸 값으로 통과시키지 않는다"))
            continue

        ctx = RunCtx(run_id="check", node_id="n1")
        params = d.build_params(d.default_params())
        try:
            first = d.impl().run(ctx, params, **inputs)
            second = d.impl().run(ctx, params, **inputs)
        except Exception as e:                      # noqa: BLE001
            rep.checks.append(Check(f"{d.type}: 실행", False, f"{type(e).__name__}: {e}"))
            continue
        rep.checks.append(Check(f"{d.type}: 실행", True, "돌았다"))

        if d.deterministic and d.kind is not NodeKind.OUTPUT:
            same = values_mod.value_hash(first) == values_mod.value_hash(second)
            rep.checks.append(Check(
                f"{d.type}: 두 번 돌려 같은 값", same,
                "순수하다" if same else
                "같은 입력에 다른 값이 나왔다. 캐시가 거짓말을 하게 된다 — "
                "무작위가 필요하면 deterministic=False 를 선언하라"))

        bad: List[str] = []
        for port, declared in ((p, d.outputs[p].type) for p in d.outputs):
            if port not in first:
                bad.append(f"{port}: 선언했는데 내보내지 않았다")
                continue
            bad += values_mod.check_against(declared, first[port], where=f"{d.type}:{port}")
        rep.checks.append(Check(
            f"{d.type}: 선언한 타입대로 내보낸다", not bad,
            "; ".join(bad[:3]) if bad else "선언과 실제가 맞는다"))

    return rep


def _spawn_ok(ref: str, module: str, root: str) -> bool:
    """정말 다른 프로세스에서 임포트되는가. 등록 검사는 이름만 보고 실제로 띄우지는 않는다."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    try:
        ex = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        try:
            return bool(ex.submit(_child_resolve, ref, module, root).result(timeout=90))
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    except Exception:                               # noqa: BLE001
        return False


def _child_resolve(ref: str, module: str, root: str) -> bool:
    """자식 프로세스에서 돈다. **모듈 최상위 함수여야** spawn 이 임포트할 수 있다."""
    import importlib
    import sys

    from ..core import registry

    if root and root not in sys.path:
        sys.path.insert(0, root)
    registry.load_builtin_nodes()
    importlib.import_module(module)
    return registry.resolve(ref) is not None


def render(rep: CheckReport) -> str:
    lines = [f"노드 검사: {rep.module}"]
    for c in rep.checks:
        mark = "v" if c.ok else "x"
        lines.append(f"  {mark} {c.name}" + (f"  —  {c.detail}" if c.detail else ""))
    lines.append("")
    lines.append("통과." if rep.ok else "고칠 것이 있다.")
    if rep.ok and rep.types:
        lines.append(f"  스펙에 적어라: node_modules: [{rep.module}]")
    return "\n".join(lines)
