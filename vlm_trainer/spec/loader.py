"""스펙 로더. YAML → GraphModel / ProcedureDef.

project.yaml만이 의미를 갖는다. layout.json은 읽지 않는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from ..core.errors import SpecError
from ..core.graph import (
    Edge,
    GraphModel,
    Materialize,
    NodeInstance,
    ProcedureInstance,
    SampleSpace,
)

SPEC_VERSION = 1


def _read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise SpecError(f"스펙 파일이 없다: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SpecError(f"스펙 최상위는 매핑이어야 한다: {path}")
    return data


def _edges(raw: Any, where: str) -> List[Edge]:
    out: List[Edge] = []
    for i, e in enumerate(raw or []):
        if not isinstance(e, dict) or "from" not in e or "to" not in e:
            raise SpecError(f"{where}: edges[{i}]는 {{from, to}} 매핑이어야 한다 — {e!r}")
        try:
            out.append(Edge.parse(str(e["from"]), str(e["to"])))
        except ValueError as exc:
            raise SpecError(f"{where}: edges[{i}] {exc}") from None
    return out


def _nodes(raw: Any, where: str) -> List[NodeInstance]:
    out: List[NodeInstance] = []
    for i, n in enumerate(raw or []):
        if not isinstance(n, dict) or "id" not in n or "type" not in n:
            raise SpecError(f"{where}: nodes[{i}]에 id 또는 type이 없다 — {n!r}")
        params = n.get("params") or {}
        if not isinstance(params, dict):
            raise SpecError(f"{where}: nodes[{i}].params는 매핑이어야 한다")
        out.append(NodeInstance(id=str(n["id"]), type=str(n["type"]), params=dict(params)))
    return out


def load_project(path: str) -> GraphModel:
    path = os.path.abspath(path)
    return build_project(_read_yaml(path), os.path.dirname(path), os.path.basename(path))


def build_project(data: Dict[str, Any], source_dir: str, where: str = "spec") -> GraphModel:
    """이미 읽어 둔 매핑에서 GraphModel을 만든다. History 되감기가 이 경로를 쓴다."""
    kind = data.get("kind", "Project")
    if kind != "Project":
        raise SpecError(f"{where}: kind는 Project여야 한다 (현재 {kind!r})")
    if int(data.get("spec_version", SPEC_VERSION)) != SPEC_VERSION:
        raise SpecError(f"{where}: 지원하지 않는 spec_version {data.get('spec_version')!r}")

    ss = data.get("sample_space") or {}
    mt = data.get("materialize") or {}

    g = GraphModel(
        id=str(data.get("id", "")),
        name=str(data.get("name", "")),
        nodes=_nodes(data.get("nodes"), where),
        edges=_edges(data.get("edges"), where),
        procedures=[
            ProcedureInstance(
                id=str(p["id"]), ref=str(p["ref"]), params=dict(p.get("params") or {})
            )
            for p in (data.get("procedures") or [])
        ],
        sample_space=SampleSpace(
            index=str(ss.get("index", "")),
            key=str(ss.get("key", "sample_id")),
            splits=dict(ss.get("splits") or {}),
            filter=str(ss.get("filter", "")),
        ),
        materialize=Materialize(
            boundary=list(mt.get("boundary") or []),
            format=str(mt.get("format", "webdataset")),
            shard_size_mb=int(mt.get("shard_size_mb", 512)),
            out_dir=str(mt.get("out_dir", "runs/{run_id}/materialized")),
        ),
        defaults=dict(data.get("defaults") or {}),
        runtime_profile=str(data.get("runtime_profile", "windows_single_gpu")),
        debug=dict(data.get("debug") or {}),
        source_dir=source_dir,
    )

    dup = [i for i in g.ids if g.ids.count(i) > 1]
    if dup:
        raise SpecError(f"{where}: 중복된 노드 id {sorted(set(dup))}")
    return g


@dataclass
class ProcedureDef:
    name: str
    version: str
    summary: str = ""
    nodes: List[NodeInstance] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    # 이름 -> ["node:port", ...]. 입력 하나가 안쪽 여러 포트를 먹일 수 있다 —
    # 스키마처럼 한 값이 두 노드에 다 필요한 경우가 흔하다. 팬인과는 다른 이야기다.
    exposed_inputs: Dict[str, List[str]] = field(default_factory=dict)
    exposed_outputs: Dict[str, str] = field(default_factory=dict)
    exposed_params: Dict[str, tuple] = field(default_factory=dict)  # 이름 -> (node, param)
    label: str = ""   # 캔버스에 뜨는 한국어 이름
    hint: str = ""    # 카드 본문 한 줄
    path: str = ""

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


def _exposed_ports(raw: Any, where: str, kind: str) -> Dict[str, Any]:
    """출력은 한 자리, 입력은 여러 자리를 받을 수 있다."""
    out: Dict[str, Any] = {}
    for name, v in (raw or {}).items():
        port = v.get("port") if isinstance(v, dict) else v
        if not port:
            raise SpecError(f"{where}: exposed_{kind}[{name}]에 port가 없다")
        if kind == "outputs":
            if isinstance(port, list):
                raise SpecError(
                    f"{where}: exposed_outputs[{name}]는 포트 하나여야 한다 — "
                    "출력이 두 곳에서 나오면 어느 값인지 정해지지 않는다"
                )
            out[str(name)] = str(port)
        else:
            out[str(name)] = [str(x) for x in (port if isinstance(port, list) else [port])]
    return out


def load_procedure(path: str) -> ProcedureDef:
    data = _read_yaml(path)
    if data.get("kind") != "Procedure":
        raise SpecError(f"{path}: kind는 Procedure여야 한다")
    where = os.path.basename(path)
    params: Dict[str, tuple] = {}
    for name, v in (data.get("exposed_params") or {}).items():
        if not isinstance(v, dict) or "node" not in v or "param" not in v:
            raise SpecError(f"{where}: exposed_params[{name}]는 {{node, param}}이어야 한다")
        params[str(name)] = (str(v["node"]), str(v["param"]))

    return ProcedureDef(
        name=str(data.get("name", "")),
        version=str(data.get("version", "0.0.0")),
        summary=str(data.get("summary", "")),
        nodes=_nodes(data.get("nodes"), where),
        edges=_edges(data.get("edges"), where),
        exposed_inputs=_exposed_ports(data.get("exposed_inputs"), where, "inputs"),
        exposed_outputs=_exposed_ports(data.get("exposed_outputs"), where, "outputs"),
        exposed_params=params,
        label=str(data.get("label", "")),
        hint=str(data.get("hint", "")),
        path=path,
    )


def find_procedure(ref: str, source_dir: Optional[str]) -> ProcedureDef:
    """`name@semver`를 프로시저 파일로 해소한다. 버전 핀은 정확히 일치해야 한다."""
    if "@" not in ref:
        raise SpecError(f"Procedure 참조는 name@semver로 핀 고정해야 한다: {ref!r}")
    name, version = ref.split("@", 1)
    roots: List[str] = []
    if source_dir:
        d = os.path.abspath(source_dir)
        for up in range(4):
            roots.append(os.path.join(d, "procedures"))
            d = os.path.dirname(d)
    for root in roots:
        cand = os.path.join(root, f"{name}.yaml")
        if os.path.exists(cand):
            pd = load_procedure(cand)
            if pd.version != version:
                raise SpecError(
                    f"Procedure {ref}: 파일 버전은 {pd.version}이다 ({cand}). "
                    "참조는 핀 고정이므로 자동 승격하지 않는다."
                )
            return pd
    raise SpecError(f"Procedure {ref}를 찾을 수 없다 (검색 경로: {roots})")
