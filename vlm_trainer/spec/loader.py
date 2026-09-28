"""스펙 로더. YAML → GraphModel / ProcedureDef.

project.yaml만이 의미를 갖는다. layout.json은 읽지 않는다.
"""

from __future__ import annotations

import importlib
import os
import sys
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


def _node_modules(raw: Any, where: str) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or any(not isinstance(m, str) for m in raw):
        raise SpecError(f"{where}: node_modules는 문자열 목록이어야 한다 (예: [my_nodes.crop])")
    return [m.strip() for m in raw if m.strip()]


def solution_root(spec_dir: str) -> str:
    """과제(solution) 루트. 노드는 보통 과제 단위로 공유되므로 여기도 찾아본다.

    규약은 `<solution>/projects/<project>/project.yaml` 이다. 부모 폴더 이름이
    `projects` 면 그 위가 루트다 — **`solution.yaml` 의 존재로 판단하지 않는다.**
    그 파일은 지금 코드가 읽지 않고, 실제로 `vlm_open` 에는 아예 없다. 없는 표지를
    기준으로 삼으면 멀쩡한 과제에서 조용히 못 찾는다.
    """
    parent = os.path.dirname(spec_dir)
    if os.path.basename(parent).lower() == "projects":
        return os.path.dirname(parent)
    for cur in (parent, os.path.dirname(parent)):
        if cur and os.path.exists(os.path.join(cur, "solution.yaml")):
            return cur
    return ""


def import_node_modules(g: GraphModel) -> List[str]:
    """스펙이 선언한 저장소 밖 노드 모듈을 임포트한다. 임포트한 모듈 이름을 돌려준다.

    **여기서 임포트해야 CLI·편집기·테스트가 같은 길을 탄다.** 플래그로만 알려 주면
    부르는 곳마다 잊을 수 있고, 잊은 곳에서는 "노드를 찾을 수 없다" 로 막힌다.

    찾는 자리는 스펙 폴더와 그 위의 solution 루트다. `my_nodes/crop.py` 를 프로젝트
    옆에 두면 그대로 잡힌다. 이미 설치된 패키지면 그것도 잡힌다.

    스펙 파일이 파이썬을 임포트한다는 뜻이므로, **실패를 조용히 넘기지 않는다** —
    모듈이 없으면 어느 자리를 뒤졌는지까지 말한다.
    """
    if not g.node_modules:
        return []

    d = os.path.abspath(g.source_dir or ".")
    roots: List[str] = [d, solution_root(d)]
    roots = [r for i, r in enumerate(roots) if r and r not in roots[:i]]

    added = [r for r in roots if r not in sys.path]
    sys.path[:0] = added
    try:
        for m in g.node_modules:
            if m in sys.modules:
                continue
            try:
                importlib.import_module(m)
            except ImportError as e:
                where_looked = ", ".join(roots)
                raise SpecError(
                    f"node_modules의 {m!r}을 임포트할 수 없다 — {e}\n"
                    f"  뒤진 곳: {where_looked}\n"
                    "  안 잡혔다면: 이 스펙이 쓰는 노드가 등록되지 않은 채\n"
                    "  컴파일에 들어가고 '노드를 찾을 수 없다'로 막힌다.\n"
                    "  무엇이 빠졌는지는 거기서 안 나온다.\n"
                    "  파일 이름과 위치를 확인하라 (예: my_nodes/crop.py -> my_nodes.crop)."
                ) from None
    finally:
        for r in added:
            if r in sys.path:
                sys.path.remove(r)
    return list(g.node_modules)


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
    g = build_project(_read_yaml(path), os.path.dirname(path), os.path.basename(path))
    # 스펙이 선언한 노드 모듈을 **여기서** 임포트한다. 이 한 줄이 CLI·편집기·테스트를
    # 같은 길에 올린다 — 부르는 쪽마다 따로 챙기면 챙기지 않은 곳이 생긴다.
    import_node_modules(g)
    return g


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
        node_modules=_node_modules(data.get("node_modules"), where),
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
