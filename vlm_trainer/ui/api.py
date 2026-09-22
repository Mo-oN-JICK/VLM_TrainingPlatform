"""편집기 API — UI가 코어를 호출하는 유일한 통로.

**UI 전용 실행 경로를 만들지 않는다.** 모든 변경은 GraphModel을 고치고 곧바로 컴파일해
G1/G2를 통과해야만 받아들여진다. 통과하지 못하면 변경 자체가 거부된다.
스펙 파일은 저장할 때만 쓰인다 — 편집 중에는 디스크의 진실이 흔들리지 않는다.
설계 문서 12, 그리고 "UI는 스펙을 편집하는 뷰일 뿐"이라는 원칙.
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.compiler import CompiledGraph, compile_graph, override_target
from ..core.errors import VlmtError
from ..core.graph import Edge, GraphModel, NodeInstance, ProcedureInstance
from ..core.registry import all_defs, resolve as resolve_node
from ..core.unify import unify_ports
from ..engine import runner as runner_mod
from ..engine import samples as samples_mod
from ..spec import recipe as recipe_mod
import yaml

from ..spec.loader import load_project
from .editor_common import HistoryEntry, _msg, _short
# Editor 의 관심사별 믹스인. 한 클래스가 아홉 가지를 담고 있어 갈랐다 —
# 메서드 이름과 동작은 그대로이고, 어디 적혀 있는지만 바뀐다.
from .editor_history import HistoryMixin
from .editor_project import ProjectMixin
from .editor_recipe import RecipeMixin
from .editor_runs import RunsMixin




def compat_matrix(cg: CompiledGraph) -> Dict[str, Dict[str, str]]:
    """출력 포트마다 어떤 입력 포트에 꽂을 수 있는지. 값이 빈 문자열이면 가능.

    이 표가 있으면 드래그 중에 서버를 다시 부르지 않고도 호환되는 포트만 밝힐 수 있다.
    실제 연결은 그래도 컴파일로 확정한다 — 표는 미리 거르는 장치일 뿐이다.
    """
    out: Dict[str, Dict[str, str]] = {}

    for snid in cg.order:
        sn = cg.nodes[snid]
        for sport, stype in sn.output_types.items():
            row: Dict[str, str] = {}
            for dnid in cg.order:
                if dnid == snid:
                    continue
                dd = resolve_node(cg.nodes[dnid].ref)
                for dport, port in dd.inputs.items():
                    ref = f"{dnid}:{dport}"
                    # 이미 배선이 있는 입력에도 놓을 수 있다 — 그 자리를 **교체**한다.
                    # 팬인은 여전히 금지되지만, 갈아끼우기를 두 단계로 만들면 편집기가 쓸모없어진다.
                    if cg.nodes[dnid].lane <= sn.lane:
                        row[ref] = "위로 향하는 배선은 만들 수 없다"
                        continue
                    res = unify_ports(stype, port.type)
                    row[ref] = "" if res.ok else ", ".join(str(m) for m in res.mismatches[:3])
            out[f"{snid}:{sport}"] = row
    return out


def param_meta(node_ref: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """파라미터마다 위젯을 고를 수 있게 값과 성질을 함께 낸다.

    타입에 영향 주는 파라미터는 표식이 필요하다 — 바꾸면 배선이 다시 검사되기 때문이다.
    """
    d = resolve_node(node_ref)
    out: List[Dict[str, Any]] = []
    for name, value in params.items():
        if isinstance(value, bool):
            kind = "bool"
        elif isinstance(value, (int, float)):
            kind = "number"
        elif isinstance(value, (list, tuple, dict)):
            kind = "json"
        else:
            kind = "text"
        out.append(
            {
                "name": name,
                "value": value,
                "kind": kind,
                "overridable": name in d.recipe_overridable,
                "type_affecting": name in d.type_affecting,
            }
        )
    return out


def occupied_inputs(cg: CompiledGraph) -> List[str]:
    """이미 배선이 있는 입력 포트. 놓으면 교체된다는 것을 UI가 알려주기 위한 것."""
    return sorted({e.dst for e in cg.edges})









@dataclass
class Editor(HistoryMixin, ProjectMixin, RecipeMixin, RunsMixin):
    """열려 있는 프로젝트 하나. 편집은 메모리에서, 저장은 명시적으로.

    관심사별 믹스인 넷을 상속한다(이력·프로젝트 설정·레시피·실행). 믹스인은 여기 있는
    `_try` · `_recompile` · `graph` 를 쓰므로 홀로 서는 클래스가 아니다 — 갈라 둔 것은
    1175줄 한 파일을 사람이 붙들 수 없었기 때문이다.
    """

    path: str
    graph: GraphModel = field(default_factory=GraphModel)
    compiled: Optional[CompiledGraph] = None
    error: str = ""
    dirty: bool = False
    valid: bool = True
    # 오버레이가 걸려 있어도 **스펙은 오버레이 전의 것**이다. base가 저장과 History의 대상이고,
    # compiled는 화면에 보이는 것이다. 이 둘을 섞으면 레시피 값이 프로젝트에 스며든다.
    base: Optional[CompiledGraph] = None
    book: Optional[Any] = None
    book_dirty: bool = False
    recipe_id: Optional[int] = None
    overlay: Dict[str, Any] = field(default_factory=dict)
    history: List[HistoryEntry] = field(default_factory=list)
    cursor: int = -1
    # 실행은 하위 프로세스다. 편집기는 그 진행 파일을 읽기만 한다
    extra_modules: Tuple[str, ...] = ()
    proc: Optional[Any] = None
    run_id: str = ""
    progress_path: str = ""
    train_progress_path: str = ""
    console_path: str = ""
    phase: str = ""
    stopped: bool = False
    launched_at: float = 0.0   # 작업을 띄운 시각. 경과 시간의 기준이다
    finished_at: float = 0.0   # 끝난 시각. 끝난 뒤에도 얼마 걸렸는지는 남아야 한다
    expanded: set = field(default_factory=set)  # 펼쳐 둔 Procedure. 화면 상태일 뿐이다
    layout: Dict[str, Any] = field(default_factory=dict)  # 상자 id -> (x, y)

    @staticmethod
    def open(path: str) -> "Editor":
        e = Editor(path=os.path.abspath(path))
        e.graph = load_project(e.path)
        e.book = recipe_mod.load(e.path)
        e._recompile()
        e._load_history()
        e._load_layout()
        if not (e.history and e.compiled is not None
                and e.history[-1].spec_hash == e.compiled.spec_hash):
            e._record("열기")
        return e

    # ── 내부 ────────────────────────────────────────────────────────────
    def _compile(self, overlay: Optional[Dict[str, Any]]):
        """strict로 먼저, 실패하면 draft로. (그래프, strict 오류, valid) 또는 (None, 오류, False).

        타입 오류는 draft에서도 잡히므로 편집 자체가 거부된다.
        구조가 덜 된 상태(배선을 잇는 중)만 통과하고, 그때 valid=False가 된다 —
        저장과 실행은 valid=True를 요구한다.
        """
        try:
            return compile_graph(self.graph, recipe_overrides=overlay or None), "", True
        except Exception as exc:
            strict_error = _msg(exc)

        try:
            return compile_graph(self.graph, recipe_overrides=overlay or None, draft=True), strict_error, False
        except Exception as exc:
            return None, _msg(exc), False

    def _recompile(self) -> bool:
        base, base_error, base_valid = self._compile(None)
        if base is None:
            self.error, self.valid = base_error, False
            return False
        self.base = base

        if not self.overlay:
            self.compiled, self.error, self.valid = base, base_error, base_valid
            return True

        cg, over_error, over_valid = self._compile(self.overlay)
        if cg is None:
            # 레시피가 없는 노드나 화이트리스트 밖 파라미터를 가리킨다 — 편집을 거부한다
            self.error, self.valid = over_error, False
            return False
        self.compiled, self.error, self.valid = cg, over_error, over_valid and base_valid
        return True

    def _try(self, mutate, label: str = "편집", check=None, undo=None) -> Dict[str, Any]:
        """변경을 적용해 보고, 게이트를 통과하지 못하면 되돌린다.

        `check`는 컴파일 뒤·기록 전에 도는 추가 검사다. 이 자리여야 하는 이유가 있다 —
        기록한 뒤에 되돌리면 저널에는 이미 줄이 들어간 뒤라 거부된 편집이 History에 남는다.
        """
        before_nodes = [NodeInstance(n.id, n.type, dict(n.params)) for n in self.graph.nodes]
        before_edges = list(self.graph.edges)
        # Procedure 인스턴스도 되돌린다. 빠져 있으면 거부된 노출 파라미터 편집이
        # 그래프에 그대로 남아, 화면은 "거부됨"인데 값은 바뀌어 있는 상태가 된다.
        before_procs = [
            ProcedureInstance(p.id, p.ref, dict(p.params)) for p in self.graph.procedures
        ]
        before_compiled, before_error, before_valid = self.compiled, self.error, self.valid

        def rollback() -> None:
            self.graph.nodes, self.graph.edges = before_nodes, before_edges
            self.graph.procedures = before_procs
            self.compiled, self.error, self.valid = before_compiled, before_error, before_valid
            if undo is not None:
                undo()

        try:
            mutate()
        except Exception as exc:
            if undo is not None:
                undo()
            return {"ok": False, "reason": _short(exc), "detail": str(exc)}

        if not self._recompile():
            reason, detail = _short(self.error), self.error
            rollback()
            return {"ok": False, "reason": reason, "detail": detail}

        problem = check() if check is not None else ""
        if problem:
            rollback()
            self._recompile()
            return {"ok": False, "reason": _short(problem), "detail": problem}

        self.dirty = True
        self._record(label)
        return {"ok": True}

    # ── 변경 ────────────────────────────────────────────────────────────
    def connect(self, src: str, dst: str) -> Dict[str, Any]:
        """입력 포트 하나에 배선은 하나. 이미 있으면 원자적으로 교체한다.

        거부되면 원래 배선이 그대로 남는다 — 갈아끼우다 실패해서 그래프가 깨지지 않는다.
        """
        replaced = next((e.src for e in self.graph.edges if e.dst == dst), "")

        def go() -> None:
            self.graph.edges = [e for e in self.graph.edges if e.dst != dst]
            self.graph.edges.append(Edge.parse(src, dst))

        res = self._try(go, f"연결 {src} -> {dst}")
        if res["ok"] and replaced:
            res["replaced"] = replaced
        return res

    def disconnect(self, dst: str) -> Dict[str, Any]:
        def go() -> None:
            self.graph.edges = [e for e in self.graph.edges if e.dst != dst]

        return self._try(go, f"배선 제거 {dst}")

    def set_param(self, node_id: str, param: str, value: Any) -> Dict[str, Any]:
        '''레시피가 덮고 있는 파라미터면 오버레이를 고치고, 아니면 스펙을 고친다.

        어느 쪽인지는 패널이 표식으로 보여준다. 덮인 값을 스펙에 써 봐야 화면에서는
        오버레이에 가려 보이지 않는다 — 그래서 조용히 스펙을 고치지 않는다.
        '''
        # 오버레이 표는 **해소된 안쪽 주소**로 키를 잡는다(override_target). 패널이 보내는
        # 것은 상자 주소(`p_prep.size`)라 그대로 조회하면 레시피가 덮고 있는데도 못 찾고
        # 스펙을 고치러 간다 — 그러면 화면에는 오버레이에 가려 바뀐 것이 안 보인다.
        key: Tuple[str, str] = (node_id, param)
        if self.compiled is not None:
            try:
                key = override_target(self.compiled, f"{node_id}.{param}")
            except ValueError:
                pass
        path = self._overlay_paths().get(key)
        if path is not None:
            return self.recipe_set(path, value)

        def go() -> None:
            try:
                n = self.graph.instance(node_id)
            except KeyError:
                raise VlmtError(
                    f"'{node_id}' 라는 상자가 이 프로젝트에 없다.\n"
                    f"  안 잡혔다면: 스펙에 없는 곳에 값을 써 두고 화면에만 반영돼,\n"
                    f"    다음에 여는 사람은 다른 값으로 돌리게 된다.\n"
                    f"  추정 낭비: 없음(편집이 거부됐다).\n"
                    f"  있는 상자: {', '.join(self.graph.ids)}"
                ) from None
            n.params[param] = value

        return self._try(go, f"{node_id}.{param} = {value!r}")

    def add_node(self, node_type: str, node_id: str = "") -> Dict[str, Any]:
        d = resolve_node(node_type)
        base = node_id or "n_" + d.type.split(".")[-1]
        nid, i = base, 1
        existing = set(self.graph.ids)
        while nid in existing:
            i += 1
            nid = f"{base}_{i}"

        def go() -> None:
            self.graph.nodes.append(NodeInstance(nid, d.ref, dict(d.default_params())))

        res = self._try(go, f"노드 추가 {nid}")
        res["id"] = nid
        return res

    def remove_node(self, node_id: str) -> Dict[str, Any]:
        def go() -> None:
            self.graph.nodes = [n for n in self.graph.nodes if n.id != node_id]
            self.graph.edges = [
                e for e in self.graph.edges if e.src_node != node_id and e.dst_node != node_id
            ]

        return self._try(go, f"노드 삭제 {node_id}")

    def toggle_expand(self, pid: str) -> Dict[str, Any]:
        """Procedure 상자를 펼치거나 접는다.

        **스펙은 바뀌지 않는다** — 보는 방식일 뿐이라 History에도 남지 않고 dirty로도 치지 않는다.
        게이트는 언제나 펼쳐진 그래프를 본다.
        """
        known = {p["id"] for p in (self.compiled.procedures if self.compiled else [])}
        if pid not in known:
            return {"ok": False, "reason": f"그런 Procedure가 없다: {pid} (있는 것: {sorted(known)})"}
        self.expanded.discard(pid) if pid in self.expanded else self.expanded.add(pid)
        return {"ok": True, "expanded": sorted(self.expanded)}

    # ── 조회 ────────────────────────────────────────────────────────────
    def library(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": d.ref,
                "category": d.category,
                "kind": d.kind.value,
                "summary": d.doc.summary,
                "inputs": list(d.inputs),
                "outputs": list(d.outputs),
            }
            for d in all_defs()
        ]

    def state(self) -> Dict[str, Any]:
        cg = self.compiled
        if cg is None:
            return {"ok": False, "error": self.error, "nodes": [], "edges": []}
        return {
            "ok": True,
            "id": cg.id,
            "name": cg.name,
            "spec_hash": cg.spec_hash,
            "dirty": self.dirty,
            "valid": self.valid,
            "error": self.error,
            "nodes": [
                {
                    "id": nid,
                    "type": cg.nodes[nid].ref,
                    "kind": cg.nodes[nid].kind.value,
                    "category": cg.nodes[nid].category,
                    "lane": cg.nodes[nid].lane,
                    "params": cg.nodes[nid].params,
                    "param_meta": param_meta(cg.nodes[nid].ref, cg.nodes[nid].params),
                    "inputs": {p: str(t) for p, t in cg.nodes[nid].input_types.items()},
                    "outputs": {p: str(t) for p, t in cg.nodes[nid].output_types.items()},
                    "wired": cg.nodes[nid].inputs,
                }
                for nid in cg.order
            ],
            "edges": [{"from": e.src, "to": e.dst} for e in cg.edges],
            "compat": compat_matrix(cg),
            "occupied": occupied_inputs(cg),
            "recipe": self.recipe_view(),
            "sample_space": self.sample_space_view(),
            "boundary": list(self.graph.materialize.boundary),
            "expanded": sorted(self.expanded),
            "layout": {k: list(v) for k, v in self.layout.items()},
            "procedures": [p["id"] for p in cg.procedures],
            "profile": self.graph.runtime_profile,
            "profiles": self.profiles(),
            "overlaid": [f"{n}:{p}" for (n, p) in self._overlay_paths()],
            "history": self.history_view(),
            "cursor": self.cursor,
        }
