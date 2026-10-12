"""Line(5노드 한 줄) -> Project 스펙. 실행 전에 기존 그래프로 펼친다.

**팔레트(칸)는 이 층의 약속일 뿐 실행 때 흐르는 값이 아니다.** 스테이션이 `reads`/`writes` 로
칸을 선언하고, 여기서 칸이 비었는지 검사한 뒤 안쪽 포트끼리 곧바로 배선한다. 펼친 그래프에
새 노드가 끼지 않으므로 같은 일을 하는 11노드 project.yaml 과 spec_hash 가 같다.

칸은 위에서 아래로 가장 최근에 쓴 스테이션의 포트를 가리킨다. 읽고 같은 칸을 다시 쓰면
(정답 노출 검사가 질문을 통과시키는 것처럼) 아래 스테이션은 다시 쓴 쪽을 읽는다.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from ..core.errors import SpecError
from .loader import _read_yaml


def find_station(ref: str, source_dir: str) -> Dict[str, Any]:
    """`name@semver` 를 `stations/<name>.yaml` 로 해소한다. 찾는 길은 Procedure 와 같다."""
    if "@" not in ref:
        raise SpecError(f"Station 참조는 name@semver로 핀 고정해야 한다: {ref!r}")
    name, version = ref.split("@", 1)
    d, roots = os.path.abspath(source_dir), []
    for _ in range(4):
        roots.append(os.path.join(d, "stations"))
        d = os.path.dirname(d)
    for root in roots:
        cand = os.path.join(root, f"{name}.yaml")
        if os.path.exists(cand):
            data = _read_yaml(cand)
            if data.get("kind") != "Station":
                raise SpecError(f"{cand}: kind는 Station이어야 한다")
            if str(data.get("version")) != version:
                raise SpecError(
                    f"Station {ref}: 파일 버전은 {data.get('version')}이다 ({cand}). "
                    "참조는 핀 고정이므로 자동 승격하지 않는다."
                )
            return data
    raise SpecError(f"Station {ref}를 찾을 수 없다 (검색 경로: {roots})")


def _targets(v: Any) -> List[str]:
    return [str(x) for x in (v if isinstance(v, list) else [v])]


def _set_param(inst: Dict[str, Dict[str, Any]], st: str, exposed: Dict[str, Any], name: str, value: Any) -> None:
    if name not in exposed:
        raise SpecError(
            f"Station {st}가 노출하지 않은 설정 {name!r}이다 (노출된 것: {sorted(exposed)}). "
            "안쪽 값은 노출한 설정으로만 덮을 수 있다."
        )
    tgt = exposed[name]
    inst[str(tgt["node"])].setdefault("params", {})[str(tgt["param"])] = value


def expand_data(data: Dict[str, Any], source_dir: str, where: str = "line") -> Dict[str, Any]:
    """Line 매핑 -> Project 매핑. 파일을 읽는 것은 Station 정의뿐이다."""
    if data.get("kind") != "Line":
        raise SpecError(f"{where}: kind는 Line이어야 한다 (현재 {data.get('kind')!r})")

    nodes: List[Dict[str, Any]] = []
    procs: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []
    boundary: List[str] = []
    writer: Dict[str, Tuple[str, str]] = {}  # 칸 -> (스테이션 id, "노드:포트")

    stations = data.get("stations") or []
    if not stations:
        raise SpecError(f"{where}: stations가 비었다 — 한 줄에 스테이션이 하나는 있어야 한다")

    seen = set()
    for si, sref in enumerate(stations):
        sid = str(sref.get("id", f"s{si}"))
        if sid in seen:
            raise SpecError(f"{where}: 중복된 스테이션 id {sid!r}")
        seen.add(sid)
        st = find_station(str(sref["ref"]), source_dir)
        label = str(st.get("label") or st.get("name"))
        sn = [dict(n, params=dict(n.get("params") or {})) for n in st.get("nodes") or []]
        sp = [dict(p, params=dict(p.get("params") or {})) for p in st.get("procedures") or []]
        inst = {str(x["id"]): x for x in sn + sp}

        for name, value in (sref.get("params") or {}).items():
            _set_param(inst, str(sref["ref"]), st.get("exposed_params") or {}, str(name), value)

        taken = {n["id"] for n in nodes} | {p["id"] for p in procs}
        dup = sorted(set(inst) & taken)
        if dup:
            raise SpecError(
                f"{where}: 스테이션 {sid}({label})의 안쪽 id {dup}가 앞 스테이션과 겹친다 — "
                "같은 종류의 스테이션을 한 줄에 둘 수 없다."
            )

        # 읽기는 이 스테이션이 쓰기 전에 푼다 — 읽고 다시 쓰는 칸은 위쪽 값을 읽는다.
        for cell, ports in (st.get("reads") or {}).items():
            if cell not in writer:
                raise SpecError(
                    f"{where}: 스테이션 {sid}({label})가 읽는 칸 {cell!r}이 비어 있다.\n"
                    f"  채워진 칸: {sorted(writer) or '없음'}\n"
                    "  안 잡혔다면: 이 스테이션의 입력 포트가 배선되지 않은 채 컴파일에서\n"
                    "  '필수 입력 포트가 연결되지 않았다'로 막히고, 어느 칸이 비었는지는 안 나온다.\n"
                    f"  {cell!r}을 쓰는 스테이션을 {label} 위에 두세요."
                )
            src = writer[cell][1]
            edges += [{"from": src, "to": t} for t in _targets(ports)]
        for cell, port in (st.get("writes") or {}).items():
            writer[str(cell)] = (sid, str(port))

        nodes += sn
        procs += sp
        edges += [{"from": str(e["from"]), "to": str(e["to"])} for e in st.get("edges") or []]
        boundary += [b for b in st.get("boundary") or [] if b not in boundary]

    mat = dict(data.get("materialize") or {})
    mat["boundary"] = list(mat.get("boundary") or []) + [b for b in boundary if b not in (mat.get("boundary") or [])]

    out: Dict[str, Any] = {
        "spec_version": data.get("spec_version", 1),
        "kind": "Project",
        "id": data.get("id", ""),
        "name": data.get("name", ""),
        "sample_space": data.get("sample_space") or {},
        "nodes": nodes,
        "edges": edges,
        "materialize": mat,
    }
    for k in ("procedures",):
        if procs:
            out[k] = procs
    for k in ("runtime_profile", "defaults", "node_modules", "debug"):
        if k in data:
            out[k] = data[k]
    return out


def is_line(path: str) -> bool:
    return _read_yaml(path).get("kind") == "Line"


def expand_file(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    return expand_data(_read_yaml(path), os.path.dirname(path), os.path.basename(path))
