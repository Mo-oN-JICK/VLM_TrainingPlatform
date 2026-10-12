"""Line(5노드 한 줄) -> Project 스펙. 실행 전에 기존 그래프로 펼친다.

**팔레트(칸)는 이 층의 약속일 뿐 실행 때 흐르는 값이 아니다.** 스테이션이 `reads`/`writes` 로
칸을 선언하고, 여기서 칸이 비었는지 검사한 뒤 안쪽 포트끼리 곧바로 배선한다. 펼친 그래프에
새 노드가 끼지 않으므로 같은 일을 하는 11노드 project.yaml 과 spec_hash 가 같다.

칸은 위에서 아래로 가장 최근에 쓴 스테이션의 포트를 가리킨다. 읽고 같은 칸을 다시 쓰면
(정답 노출 검사가 질문을 통과시키는 것처럼) 아래 스테이션은 다시 쓴 쪽을 읽는다.
"""

from __future__ import annotations

import os

import yaml
from typing import Any, Dict, List, Optional, Tuple

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


# 학습량 단계와 반복 횟수의 배율. 보통은 설정 파일의 값을 그대로 쓴다(아무것도 덮지 않는다).
AMOUNTS = {"적게": 0.5, "보통": 1.0, "많게": 2.0}


def amount_of(sref: Dict[str, Any]) -> str:
    """이 스테이션이 지금 어느 학습량인가. 직접 입력한 횟수가 있으면 단계가 아니다(빈 문자열)."""
    p = sref.get("params") or {}
    if p.get("epochs"):
        return ""
    return str(p.get("amount") or "보통")


def _apply_amount(inst: Dict[str, Dict[str, Any]], st: Dict[str, Any], amount: str,
                  given: Dict[str, Any], source_dir: str, where: str, sid: str, label: str) -> None:
    spec = st.get("amount")
    if not spec:
        raise SpecError(f"{where}: 스테이션 {sid}({label})에는 학습량이 없다.\n"
                        "  안 잡혔다면: 적은 값이 아무 데도 쓰이지 않는다.\n  amount 를 빼세요.")
    if amount not in AMOUNTS:
        raise SpecError(f"{where}: 학습량 {amount!r}는 없는 값이다 ({' | '.join(AMOUNTS)}).\n"
                        "  안 잡혔다면: 몇 배로 학습하는지 짐작해야 한다.\n  셋 중 하나를 적으세요.")
    target = str(spec["node"])
    direct = inst[target].get("params", {}).get(str(spec["param"]))
    if direct:
        raise SpecError(f"{where}: 스테이션 {sid}({label})에 학습량 {amount!r}과 직접 입력한 횟수 {direct}가 같이 있다.\n"
                        "  안 잡혔다면: 어느 쪽이 적용되는지 화면과 실제가 갈라진다.\n"
                        "  둘 중 하나만 남기세요(직접 입력을 지우거나 학습량을 빼세요).")
    if amount == "보통":
        return
    cfg_param = inst[target].get("params", {}).get(str(spec["config"]))
    path = str(cfg_param or "")
    full = path if os.path.isabs(path) else os.path.normpath(os.path.join(source_dir, path))
    try:
        with open(full, "r", encoding="utf-8") as fh:
            stages = (yaml.safe_load(fh) or {}).get("stages") or []
    except OSError as exc:
        raise SpecError(f"{where}: 학습량을 셈하려면 학습 설정 파일이 필요한데 읽지 못했다: {full} ({exc})\n"
                        "  안 잡혔다면: 몇 번이 기본인지 모른 채 학습이 시작된다.\n"
                        "  학습 설정 파일 경로를 확인하세요.") from None
    factor = AMOUNTS[amount]
    inst[target].setdefault("params", {})[str(spec["param"])] = {
        str(s.get("name", "default")): max(1, int(int(s.get("epochs", 1)) * factor + 0.5)) for s in stages}


def _apply_overrides(data: Dict[str, Any], nodes: List[Dict[str, Any]], procs: List[Dict[str, Any]],
                     where: str) -> None:
    """안쪽 상자 값 수정. 스테이션이 노출하지 않은 값도 고칠 수 있는 길이고, 전문가 모드의 몫이다."""
    by_id = {str(x["id"]): x for x in nodes + procs}
    for i, o in enumerate(data.get("overrides") or []):
        if not isinstance(o, dict) or not {"node", "param"} <= set(o) or "value" not in o:
            raise SpecError(f"{where}: overrides[{i}]는 {{node, param, value}} 이어야 한다 — {o!r}")
        tgt = by_id.get(str(o["node"]))
        if tgt is None:
            raise SpecError(f"{where}: overrides[{i}]가 없는 상자를 가리킨다: {o['node']!r} (있는 상자: {sorted(by_id)}).\n"
                            "  안 잡혔다면: 고친 값이 어디에도 적용되지 않는데 적용된 줄 안다.\n"
                            "  상자 id 를 확인하세요.")
        tgt.setdefault("params", {})[str(o["param"])] = o["value"]


def expand_data(data: Dict[str, Any], source_dir: str, where: str = "line", *, check: bool = True,
                info: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Line 매핑 -> Project 매핑. 읽는 파일은 Station 정의와, `check` 일 때 표(첫 줄과 split)다."""
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

        given = dict(sref.get("params") or {})
        amount = given.pop("amount", None)
        for name, value in given.items():
            _set_param(inst, str(sref["ref"]), st.get("exposed_params") or {}, str(name), value)
        if amount is not None:
            _apply_amount(inst, st, str(amount), given, source_dir, where, sid, label)

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

        if info is not None:
            shown = {}
            for name, tgt in (st.get("exposed_params") or {}).items():
                shown[str(name)] = {
                    "label": str(tgt.get("label") or name),
                    "value": inst[str(tgt["node"])].get("params", {}).get(str(tgt["param"]), None),
                    "level": str(tgt.get("level") or "expert"),
                    "choices": list(tgt.get("choices") or []),
                    "target": [str(tgt["node"]), str(tgt["param"])],
                }
            am = st.get("amount")
            info.append({
                "id": sid, "number": si + 1, "ref": str(sref["ref"]), "name": str(st.get("name")),
                "label": label, "hint": str(st.get("hint") or ""), "members": list(inst),
                "settings": shown,
                "amount": ({"label": str(am.get("label") or "학습량"), "level": str(am.get("level") or "basic"),
                            "value": amount_of(sref), "choices": list(AMOUNTS)} if am else None),
            })

        nodes += sn
        procs += sp
        edges += [{"from": str(e["from"]), "to": str(e["to"])} for e in st.get("edges") or []]
        boundary += [b for b in st.get("boundary") or [] if b not in boundary]

    _apply_overrides(data, nodes, procs, where)

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
    if check:
        from .table_check import check_table

        check_table(out, source_dir, where)
    return out


def is_line(path: str) -> bool:
    return _read_yaml(path).get("kind") == "Line"


def expand_file(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    return expand_data(_read_yaml(path), os.path.dirname(path), os.path.basename(path))


def describe_file(path: str) -> Dict[str, Any]:
    """화면이 그리는 Line 의 요약: 스테이션 목록(안쪽 id, 설정)과 표 설정. 표는 읽지 않는다."""
    path = os.path.abspath(path)
    data = _read_yaml(path)
    stations: List[Dict[str, Any]] = []
    expand_data(data, os.path.dirname(path), os.path.basename(path), check=False, info=stations)
    ss = data.get("sample_space") or {}
    return {"id": str(data.get("id", "")), "name": str(data.get("name", "")), "stations": stations,
            "table": {"index": str(ss.get("index", "")), "key": str(ss.get("key", "")),
                      "splits": dict(ss.get("splits") or {})}}
