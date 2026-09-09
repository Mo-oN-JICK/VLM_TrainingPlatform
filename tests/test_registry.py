"""Phase 0 완료 조건 — 3분류와 포트 형상의 강제는 등록 시점에 일어난다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from vlm_trainer.core.errors import RegistrationError
from vlm_trainer.core.node import Node, NodeKind, Port, RunCtx
from vlm_trainer.core.registry import register, resolve
from vlm_trainer.core.types import image, text


# 등록 실패를 검증하려면 클래스가 모듈 최상위에 있어야 한다(spawn 요구와 같은 이유).
class _Dummy(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {}


class _DummyWithFingerprint(Node):
    def fingerprint(self, params: Any) -> str:
        return "x"

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {}


@dataclass
class _P:
    a: int = 1


def test_input_node_may_not_have_input_ports():
    with pytest.raises(RegistrationError, match="Input 노드는 입력 포트가 없고"):
        register(
            type="bad.input_with_inputs",
            version="1.0.0",
            category="X",
            kind=NodeKind.INPUT,
            inputs={"x": Port(text())},
            outputs={"y": Port(text())},
        )(_DummyWithFingerprint)


def test_processing_node_needs_both_sides():
    with pytest.raises(RegistrationError, match="Processing 노드는"):
        register(
            type="bad.processing_no_output",
            version="1.0.0",
            category="X",
            kind=NodeKind.PROCESSING,
            inputs={"x": Port(text())},
        )(_Dummy)


def test_output_node_may_not_have_output_ports():
    with pytest.raises(RegistrationError, match="Output 노드는"):
        register(
            type="bad.output_with_outputs",
            version="1.0.0",
            category="X",
            kind=NodeKind.OUTPUT,
            inputs={"x": Port(text())},
            outputs={"y": Port(text())},
        )(_Dummy)


def test_input_node_must_implement_fingerprint():
    with pytest.raises(RegistrationError, match="fingerprint"):
        register(
            type="bad.input_no_fingerprint",
            version="1.0.0",
            category="X",
            kind=NodeKind.INPUT,
            outputs={"y": Port(text())},
        )(_Dummy)


def test_external_call_only_on_processing():
    with pytest.raises(RegistrationError, match="external_call"):
        register(
            type="bad.external_output",
            version="1.0.0",
            category="X",
            kind=NodeKind.OUTPUT,
            inputs={"x": Port(text())},
            external_call=True,
        )(_Dummy)


def test_recipe_whitelist_must_name_real_params():
    with pytest.raises(RegistrationError, match="recipe_overridable"):
        register(
            type="bad.unknown_overridable",
            version="1.0.0",
            category="X",
            kind=NodeKind.PROCESSING,
            inputs={"x": Port(text())},
            outputs={"y": Port(text())},
            params=_P,
            recipe_overridable=["nope"],
        )(_Dummy)


def test_type_affecting_must_be_subset_of_overridable():
    with pytest.raises(RegistrationError, match="type_affecting"):
        register(
            type="bad.type_affecting",
            version="1.0.0",
            category="X",
            kind=NodeKind.PROCESSING,
            inputs={"x": Port(text())},
            outputs={"y": Port(text())},
            params=_P,
            recipe_overridable=[],
            type_affecting=["a"],
        )(_Dummy)


def test_locally_defined_nodes_are_rejected_for_spawn():
    class _Local(Node):  # 함수 안에 정의 -> spawn 워커가 임포트할 수 없다
        def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
            return {}

    with pytest.raises(RegistrationError, match="spawn"):
        register(
            type="bad.local_class",
            version="1.0.0",
            category="X",
            kind=NodeKind.PROCESSING,
            inputs={"x": Port(text())},
            outputs={"y": Port(text())},
        )(_Local)


def test_resolve_pins_and_defaults_to_latest():
    d = resolve("test.crop@1.0.0")
    assert d.type == "test.crop" and d.kind is NodeKind.PROCESSING
    assert resolve("test.crop").version == "1.0.0"
    with pytest.raises(RegistrationError, match="찾을 수 없다"):
        resolve("test.crop@9.9.9")


def test_defaults_are_materialized_not_omitted():
    d = resolve("test.crop")
    assert d.default_params() == {"padding_ratio": 0.15, "max_n": 3}
