"""편집 가능한 그래프 모델.

UI와 스펙 로더가 모두 이 모델을 만들고, 컴파일러가 이것만 소비한다.
좌표는 여기에 없다 — layout.json은 sidecar이고 의미에 관여하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NodeInstance:
    id: str
    type: str  # "image.crop_by_regions@1.0.0" 또는 버전 없는 이름
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcedureInstance:
    id: str
    ref: str  # "expert_region_crop@1.2.0"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str

    @property
    def src(self) -> str:
        return f"{self.src_node}:{self.src_port}"

    @property
    def dst(self) -> str:
        return f"{self.dst_node}:{self.dst_port}"

    @staticmethod
    def parse(frm: str, to: str) -> "Edge":
        a, b = _split(frm), _split(to)
        return Edge(a[0], a[1], b[0], b[1])


def _split(ref: str) -> Tuple[str, str]:
    if ":" not in ref:
        raise ValueError(f"포트 참조는 '노드id:포트명' 형식이어야 한다: {ref!r}")
    node, port = ref.rsplit(":", 1)
    return node, port


@dataclass
class SampleSpace:
    index: str = ""
    key: str = "sample_id"
    splits: Dict[str, Any] = field(default_factory=dict)
    filter: str = ""


@dataclass
class Materialize:
    boundary: List[str] = field(default_factory=list)
    format: str = "webdataset"
    shard_size_mb: int = 512
    out_dir: str = "runs/{run_id}/materialized"


@dataclass
class GraphModel:
    id: str = ""
    name: str = ""
    nodes: List[NodeInstance] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    procedures: List[ProcedureInstance] = field(default_factory=list)
    sample_space: SampleSpace = field(default_factory=SampleSpace)
    materialize: Materialize = field(default_factory=Materialize)
    defaults: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    runtime_profile: str = "windows_single_gpu"
    debug: Dict[str, Any] = field(default_factory=dict)
    source_dir: Optional[str] = None  # procedure 상대 경로 해소용

    # ── 조회 ────────────────────────────────────────────────────────────
    def node(self, node_id: str) -> NodeInstance:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def incoming(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges if e.dst_node == node_id]

    def outgoing(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges if e.src_node == node_id]

    @property
    def ids(self) -> List[str]:
        return [n.id for n in self.nodes] + [p.id for p in self.procedures]
