"""노드 계약.

3분류(Input/Processing/Output)와 포트 형상은 1:1로 대응하며 등록 시점에 강제된다.
설계 문서 02 §2.4 ~ §2.6.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from .errors import VlmtError
from .types import PortType


class NodeKind(str, Enum):
    INPUT = "Input"
    PROCESSING = "Processing"
    OUTPUT = "Output"


@dataclass(frozen=True)
class Port:
    type: PortType
    doc: str = ""

    @property
    def optional(self) -> bool:
        return self.type.optional


@dataclass(frozen=True)
class NodeDoc:
    """Node Quick Info 탭에 그대로 들어간다."""

    summary: str = ""
    scenario: str = ""
    ports: str = ""


class NodeError(VlmtError):
    """노드 실행 실패. 엔진은 이 타입만 격리한다."""

    def __init__(
        self,
        node_id: str,
        cause: str,
        *,
        port: str = "",
        sample_key: str = "",
        hint: str = "",
    ) -> None:
        super().__init__(f"[{node_id}{':' + port if port else ''}] {cause}")
        self.node_id, self.port, self.sample_key, self.cause, self.hint = (
            node_id,
            port,
            sample_key,
            cause,
            hint,
        )


@dataclass
class RunCtx:
    """노드가 바깥 세계에 닿는 유일한 통로.

    Processing 노드는 여기 없는 것(파일 시스템, 네트워크, 전역 RNG, 시계)에
    절대 접근하지 않는다.
    """

    run_id: str = "dev"
    node_id: str = ""
    sample_key: str = ""
    scratch: Optional[str] = None
    seed: int = 0
    now: str = "1970-01-01T00:00:00Z"
    sample: Dict[str, Any] = field(default_factory=dict)  # sample_space의 현재 행
    root: str = "."  # 인덱스 파일 기준 디렉터리. Input 노드만 경로를 만든다

    def path(self, rel: str) -> str:
        import os

        return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(self.root, rel))

    def rng(self, salt: str = ""):
        import random

        return random.Random(f"{self.seed}|{self.node_id}|{self.sample_key}|{salt}")


class Node:
    """모든 노드의 기반. 하위 클래스는 run()만 구현하면 된다."""

    definition: "NodeDef"  # register()가 채운다

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def infer_types(
        self, inputs: Dict[str, PortType], params: Any
    ) -> Dict[str, PortType]:
        """선언한 출력 타입을 반환한다. 심볼 dim 해소가 필요하면 재정의한다."""
        return {name: p.type for name, p in self.definition.outputs.items()}

    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        """Input 노드 전용. 외부 상태(경로 + 크기 + mtime)의 지문."""
        raise NotImplementedError

    def preview(self, ctx: RunCtx, params: Any, outputs: Dict[str, Any]) -> Any:
        return None


@dataclass
class NodeDef:
    type: str
    version: str
    category: str
    kind: NodeKind
    inputs: Dict[str, Port]
    outputs: Dict[str, Port]
    params: Optional[type] = None
    recipe_overridable: tuple = ()
    type_affecting: tuple = ()
    preview: str = ""
    external_call: bool = False
    deterministic: bool = True
    clears_taint: tuple = ()   # 이 노드를 거치면 제거되는 semantic taint (예: label)
    doc: NodeDoc = field(default_factory=NodeDoc)
    impl: Optional[Type[Node]] = None

    @property
    def ref(self) -> str:
        return f"{self.type}@{self.version}"

    def default_params(self) -> Dict[str, Any]:
        if self.params is None or not dataclasses.is_dataclass(self.params):
            return {}
        out: Dict[str, Any] = {}
        for f in dataclasses.fields(self.params):
            if f.default is not dataclasses.MISSING:
                out[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                out[f.name] = f.default_factory()  # type: ignore[misc]
            else:
                out[f.name] = None
        return out

    def build_params(self, values: Dict[str, Any]) -> Any:
        if self.params is None:
            return dict(values)
        known = {f.name for f in dataclasses.fields(self.params)}
        unknown = set(values) - known
        if unknown:
            raise KeyError(
                f"{self.ref}: 알 수 없는 파라미터 {sorted(unknown)} "
                f"(사용 가능: {sorted(known)})"
            )
        return self.params(**values)
