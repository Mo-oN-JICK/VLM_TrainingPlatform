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

from ..core.compiler import CompiledGraph, compile_graph, current_value, override_target
from ..core.errors import VlmtError
from ..core.graph import Edge, GraphModel, NodeInstance
from ..core.node import NodeKind
from ..core.registry import all_defs, resolve as resolve_node
from ..core.unify import unify_ports
from ..engine import runner as runner_mod
from ..engine import samples as samples_mod
from ..spec import recipe as recipe_mod
from ..spec.decompile import decompile, dump_yaml
import yaml

from ..spec.loader import build_project, load_project


def _short(errors: Any) -> str:
    lines = [ln.strip() for ln in str(errors).splitlines() if ln.strip()]
    if lines and lines[0].endswith("게이트 위반:"):
        lines = lines[1:]  # 개수 머리말이 아니라 첫 번째 이유를 보여준다
    return lines[0] if lines else str(errors)


def _msg(exc: Exception) -> str:
    return str(exc) if isinstance(exc, VlmtError) else f"{type(exc).__name__}: {exc}"


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


HISTORY_FILE = "edit_history.jsonl"
HISTORY_STORE = "edit_history"   # 시점별 스펙 본문. 내용 주소라 같은 상태로 돌아와도 파일이 늘지 않는다
HISTORY_WINDOW = 200             # 열 때 되살릴 시점의 최대 개수


@dataclass
class HistoryEntry:
    """한 번의 편집. 스펙이 텍스트라 시점 복원이 스냅샷 하나로 끝난다."""

    label: str
    spec_hash: str
    spec: str = ""  # canonical YAML
    at: str = ""
    diff: List[str] = field(default_factory=list)
    past: bool = False  # 이전 세션에서 남은 시점


@dataclass
class Editor:
    """열려 있는 프로젝트 하나. 편집은 메모리에서, 저장은 명시적으로."""

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
    expanded: set = field(default_factory=set)  # 펼쳐 둔 Procedure. 화면 상태일 뿐이다

    @staticmethod
    def open(path: str) -> "Editor":
        e = Editor(path=os.path.abspath(path))
        e.graph = load_project(e.path)
        e.book = recipe_mod.load(e.path)
        e._recompile()
        e._load_history()
        if not (e.history and e.compiled is not None
                and e.history[-1].spec_hash == e.compiled.spec_hash):
            e._record("열기")
        return e

    # ── History ─────────────────────────────────────────────────────────
    @property
    def history_path(self) -> str:
        return os.path.join(os.path.dirname(self.path), HISTORY_FILE)

    def _spec_text(self) -> str:
        return dump_yaml(decompile(self.base, keep_procedures=True)) if self.base else ""

    @property
    def history_store(self) -> str:
        return os.path.join(os.path.dirname(self.path), HISTORY_STORE)

    def _snapshot_path(self, spec_hash: str) -> str:
        return os.path.join(self.history_store, spec_hash.replace(":", "_") + ".yaml")

    def _keep_snapshot(self, spec_hash: str, text: str) -> None:
        """시점의 스펙 본문을 내용 주소로 남긴다.

        저널은 사람이 읽는 diff고, 되감기에는 본문이 필요하다. 둘을 한 파일에 섞으면
        저널이 읽을 수 없게 된다. 실패해도 편집을 막지 않는다.
        """
        path = self._snapshot_path(spec_hash)
        if os.path.exists(path):
            return  # 같은 상태로 돌아온 것이다
        try:
            os.makedirs(self.history_store, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except OSError:
            pass

    def _load_snapshot(self, spec_hash: str) -> str:
        try:
            with open(self._snapshot_path(spec_hash), "r", encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            return ""

    def _load_history(self) -> None:
        """저널을 되짚어 지난 세션의 시점을 되살린다.

        본문이 남아 있는 시점만 되살린다 — 되감을 수 없는 항목을 목록에 두면
        누를 수 있는 것과 없는 것이 섞여 History가 신뢰를 잃는다.
        """
        lines: List[Dict[str, Any]] = []
        try:
            with open(self.history_path, "r", encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        lines.append(json.loads(ln))
                    except ValueError:
                        continue
        except OSError:
            return

        for rec in lines[-HISTORY_WINDOW:]:
            spec_hash = str(rec.get("spec_hash", ""))
            text = self._load_snapshot(spec_hash) if spec_hash else ""
            if not text:
                continue
            if self.history and self.history[-1].spec_hash == spec_hash:
                continue  # 같은 상태가 이어지면 시점 하나다
            self.history.append(
                HistoryEntry(
                    label=str(rec.get("label", "")),
                    spec_hash=spec_hash,
                    spec=text,
                    at=str(rec.get("at", "")),
                    diff=list(rec.get("diff") or [])[:12],
                    past=True,
                )
            )
        self.cursor = len(self.history) - 1
        self._prune_snapshots({str(r.get("spec_hash", "")) for r in lines})

    def _prune_snapshots(self, referenced: Any) -> None:
        """저널이 더 이상 가리키지 않는 본문을 지운다. 저장소는 저널을 따라간다."""
        names = {h.replace(":", "_") + ".yaml" for h in referenced if h}
        try:
            for name in os.listdir(self.history_store):
                if name.endswith(".yaml") and name not in names:
                    os.remove(os.path.join(self.history_store, name))
        except OSError:
            pass

    def _record(self, label: str) -> None:
        """편집 한 번을 시점으로 남긴다. redo 가지가 있으면 잘라낸다."""
        if self.compiled is None:
            return
        text = self._spec_text()
        prev = self.history[self.cursor].spec.splitlines() if self.cursor >= 0 else []
        diff = [
            ln
            for ln in difflib.unified_diff(prev, text.splitlines(), lineterm="", n=0)
            if ln and ln[0] in "+-" and not ln.startswith(("+++", "---"))
        ]
        del self.history[self.cursor + 1 :]
        entry = HistoryEntry(
            label=label,
            spec_hash=self.compiled.spec_hash,
            spec=text,
            at=time.strftime("%Y-%m-%d %H:%M:%S"),
            diff=diff[:12],
        )
        self.history.append(entry)
        self.cursor = len(self.history) - 1
        self._keep_snapshot(entry.spec_hash, text)
        try:  # 저널은 사람이 읽는 기록이다. 실패해도 편집을 막지 않는다
            payload = {"at": entry.at, "label": label, "spec_hash": entry.spec_hash, "diff": diff}
            with open(self.history_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _restore(self, index: int) -> Dict[str, Any]:
        if not (0 <= index < len(self.history)):
            return {"ok": False, "reason": f"그 시점이 없다: {index}"}
        entry = self.history[index]
        try:
            data = yaml.safe_load(entry.spec) or {}
            self.graph = build_project(data, os.path.dirname(self.path), "history")
        except Exception as exc:
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        if not self._recompile():
            return {"ok": False, "reason": _short(self.error), "detail": self.error}
        self.cursor = index
        self.dirty = True
        return {"ok": True, "cursor": index, "label": entry.label}

    def undo(self) -> Dict[str, Any]:
        if self.cursor <= 0:
            return {"ok": False, "reason": "되돌릴 편집이 없다"}
        return self._restore(self.cursor - 1)

    def redo(self) -> Dict[str, Any]:
        if self.cursor >= len(self.history) - 1:
            return {"ok": False, "reason": "다시 할 편집이 없다"}
        return self._restore(self.cursor + 1)

    def rewind(self, index: int) -> Dict[str, Any]:
        """History 항목을 클릭하면 그 시점으로 돌아간다. Undo/Redo는 이 커서의 이동일 뿐이다."""
        return self._restore(int(index))

    def history_view(self) -> List[Dict[str, Any]]:
        return [
            {
                "index": i,
                "label": h.label,
                "at": h.at[-8:],  # 시:분:초
                "when": h.at,
                "past": h.past,
                "spec_hash": h.spec_hash,
                "current": i == self.cursor,
                "diff": h.diff,
            }
            for i, h in enumerate(self.history)
        ]

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
        before_compiled, before_error, before_valid = self.compiled, self.error, self.valid

        def rollback() -> None:
            self.graph.nodes, self.graph.edges = before_nodes, before_edges
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
        path = self._overlay_paths().get((node_id, param))
        if path is not None:
            return self.recipe_set(path, value)

        def go() -> None:
            n = self.graph.node(node_id)
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

    # ── 물질화 경계와 실행 프로파일 ──────────────────────────────────────
    #
    # 둘 다 그래프 밖의 한 줄이지만 게이트가 그 한 줄을 보고 판단한다.
    # 경계가 비어 있으면 외부 모델이 학습 루프 안에서 돌고, 프로파일이 틀리면
    # 이 기계에서 돌지 않는 설정으로 몇 시간을 태운다.

    def set_boundary(self, node_id: str, on: bool) -> Dict[str, Any]:
        '''노드 하나를 물질화 경계에 넣거나 뺀다. 경계까지가 미리 굽는 구간이다.'''
        if node_id not in (self.compiled.nodes if self.compiled else {}):
            return {"ok": False, "reason": f"그런 노드가 없다: {node_id}"}
        before = list(self.graph.materialize.boundary)
        if on and node_id in before:
            return {"ok": False, "reason": f"{node_id}는 이미 경계에 있다"}
        if not on and node_id not in before:
            return {"ok": False, "reason": f"{node_id}는 경계에 없다"}

        def go() -> None:
            b = list(self.graph.materialize.boundary)
            self.graph.materialize.boundary = (
                b + [node_id] if on else [x for x in b if x != node_id]
            )

        def back() -> None:
            self.graph.materialize.boundary = before

        verb = "경계에 추가" if on else "경계에서 제거"
        return self._try(go, f"물질화 {verb} {node_id}", undo=back)

    def set_profile(self, profile: str) -> Dict[str, Any]:
        before = self.graph.runtime_profile

        def go() -> None:
            self.graph.runtime_profile = str(profile)

        def back() -> None:
            self.graph.runtime_profile = before

        return self._try(go, f"runtime_profile = {profile}", undo=back)

    def profiles(self) -> List[str]:
        from ..train.config import PROFILE_UNSUPPORTED

        known = sorted(PROFILE_UNSUPPORTED)
        cur = self.graph.runtime_profile
        return known if cur in known else [cur] + known

    # ── Sample Space ────────────────────────────────────────────────────
    #
    # 그래프 밖의 선언이지만 그래프만큼 자주 틀린다. key 하나가 어긋나면 컴파일은 통과하고
    # 실행이 첫 샘플에서 죽는다. 그래서 고칠 때마다 **실제로 읽어 본다.**

    SAMPLE_FIELDS = ("index", "key", "filter", "splits")

    def sample_space_view(self) -> Dict[str, Any]:
        ss = self.graph.sample_space
        out: Dict[str, Any] = {
            "index": ss.index,
            "key": ss.key,
            "filter": ss.filter,
            "splits": dict(ss.splits or {}),
            "rows": 0,
            "columns": [],
            "split_counts": {},
            "error": "",
        }
        if self.compiled is None:
            return out
        try:
            space = samples_mod.load(self.compiled.sample_space, os.path.dirname(self.path))
        except Exception as exc:
            out["error"] = _msg(exc)
            return out

        out["rows"] = len(space.rows)
        cols: List[str] = []
        for r in space.rows[:50]:
            for c in r:
                if c not in cols:
                    cols.append(c)
        out["columns"] = cols
        counts: Dict[str, int] = {}
        for r in space.rows:
            counts[str(r.get("_split", ""))] = counts.get(str(r.get("_split", "")), 0) + 1
        out["split_counts"] = counts
        return out

    def set_sample_space(self, field_name: str, value: Any) -> Dict[str, Any]:
        """샘플 공간의 한 항목을 고친다. 그래프 편집과 같은 통로를 지난다 —
        컴파일하고, History에 남고, Save 때 디스크에 쓰인다."""
        if field_name not in self.SAMPLE_FIELDS:
            return {
                "ok": False,
                "reason": f"sample_space에 {field_name!r} 항목은 없다 "
                f"(있는 것: {list(self.SAMPLE_FIELDS)})",
            }
        if field_name == "splits" and not isinstance(value, dict):
            return {"ok": False, "reason": "splits는 객체여야 한다"}

        before = getattr(self.graph.sample_space, field_name)

        def go() -> None:
            setattr(self.graph.sample_space, field_name, value)

        def back() -> None:
            setattr(self.graph.sample_space, field_name, before)

        # 컴파일은 통과해도 인덱스가 읽히지 않으면 거부한다. 읽히지 않는 선언은
        # 실행 첫 샘플에서 죽는데, 그때는 이미 편집기를 닫은 뒤다.
        res = self._try(
            go,
            f"sample_space.{field_name} = {value!r}",
            check=lambda: self.sample_space_view()["error"],
            undo=back,
        )
        if res["ok"]:
            res["view"] = self.sample_space_view()
        return res

    # ── Parameter Recipe ────────────────────────────────────────────────
    #
    # 레시피는 **값만 덮는 오버레이**다. CLI의 `--recipe N`과 같은 통로를 쓴다.
    # 그래프에 값을 써 넣지 않는 이유가 있다: Procedure가 노출한 파라미터(`p_crop.max_n`)는
    # 프로젝트 스펙이 아니라 다른 파일 안의 노드를 가리킨다. 그것을 스펙에 써 넣으려면
    # Procedure 파일을 고쳐야 하고, 그러면 그 Procedure를 쓰는 다른 프로젝트가 함께 바뀐다.

    def _overlay_paths(self) -> Dict[Tuple[str, str], str]:
        """(노드id, 파라미터) -> 오버레이 경로. 파라미터 편집이 어디로 갈지 정한다."""
        out: Dict[Tuple[str, str], str] = {}
        if self.compiled is None:
            return out
        for path in self.overlay:
            try:
                out[override_target(self.compiled, path)] = path
            except ValueError:
                pass
        return out

    def _try_overlay(self, overlay: Dict[str, Any]) -> Dict[str, Any]:
        """오버레이를 갈아 끼워 보고, 컴파일되지 않으면 되돌린다."""
        before = dict(self.overlay)
        self.overlay = dict(overlay)
        if self._recompile():
            return {"ok": True}
        reason, detail = _short(self.error), self.error
        self.overlay = before
        self._recompile()
        return {"ok": False, "reason": reason, "detail": detail}

    def recipe_select(self, rid: Optional[int]) -> Dict[str, Any]:
        """레시피를 적용하거나(번호) 벗긴다(None). 프로젝트 스펙은 바뀌지 않는다."""
        if rid is None:
            self.recipe_id = None
            return self._try_overlay({})
        if self.book is None:
            return {"ok": False, "reason": "레시피 파일이 없다"}
        try:
            overrides = self.book.overrides_for(int(rid))
        except Exception as exc:
            return {"ok": False, "reason": _short(exc), "detail": _msg(exc)}
        res = self._try_overlay(overrides)
        if res["ok"]:
            self.recipe_id = int(rid)
        return res

    def recipe_set(self, path: str, value: Any) -> Dict[str, Any]:
        """오버레이의 값 하나를 고친다. 레시피 파일은 '레시피에 담기' 전까지 그대로다."""
        if not self.overlay:
            return {"ok": False, "reason": "레시피가 덮고 있는 값이 없다"}
        return self._try_overlay({**self.overlay, path: value})

    def recipe_add_path(self, path: str) -> Dict[str, Any]:
        """파라미터 하나를 레시피가 다루는 축으로 만든다. 현재 값을 그대로 담는다."""
        try:
            recipe_mod.check_override_path(path)
        except Exception as exc:
            return {"ok": False, "reason": _short(exc), "detail": _msg(exc)}
        if path in self.overlay:
            return {"ok": False, "reason": f"{path}는 이미 레시피가 덮고 있다"}
        value = current_value(self.base, path) if self.base else None
        return self._try_overlay({**self.overlay, path: value})

    def recipe_drop_path(self, path: str) -> Dict[str, Any]:
        if path not in self.overlay:
            return {"ok": False, "reason": f"{path}는 레시피가 덮고 있지 않다"}
        rest = {k: v for k, v in self.overlay.items() if k != path}
        return self._try_overlay(rest)

    def recipe_store(self, rid: Optional[int] = None, name: str = "", note: str = "") -> Dict[str, Any]:
        """지금 오버레이를 레시피로 굳힌다. rid가 없으면 빈 번호를 하나 쓴다.

        파일은 Save 때 함께 쓰인다 — 편집기에서 디스크가 바뀌는 순간은 Save 하나뿐이다.
        """
        if self.book is None:
            return {"ok": False, "reason": "레시피 파일이 없다"}
        if not self.overlay:
            return {"ok": False, "reason": "담을 값이 없다 — 먼저 파라미터를 레시피 축으로 만든다"}
        if rid is None:
            try:
                rid = self.book.next_ids(1, (recipe_mod.MIN_ID, recipe_mod.MAX_ID))[0]
            except Exception as exc:
                return {"ok": False, "reason": _short(exc), "detail": _msg(exc)}
        rid = int(rid)
        old = self.book.recipes.get(rid)
        self.book.recipes[rid] = recipe_mod.Recipe(
            id=rid,
            name=name or (old.name if old else ""),
            note=note or (old.note if old else ""),
            overrides=dict(self.overlay),
        )
        self.book_dirty = True
        self.recipe_id = rid
        return {"ok": True, "id": rid, "label": self.book.recipes[rid].label}

    def recipe_delete(self, rid: int) -> Dict[str, Any]:
        if self.book is None or int(rid) not in self.book.recipes:
            return {"ok": False, "reason": f"레시피 {rid}번이 없다"}
        del self.book.recipes[int(rid)]
        if self.book.active == int(rid):
            self.book.active = None
        self.book_dirty = True
        if self.recipe_id == int(rid):
            self.recipe_select(None)
        return {"ok": True}

    def recipe_set_active(self, rid: Optional[int]) -> Dict[str, Any]:
        """`active:`는 '프로젝트가 지금 이 레시피대로다'라는 표시다(Mech-Vision 규약)."""
        if self.book is None:
            return {"ok": False, "reason": "레시피 파일이 없다"}
        if rid is not None and int(rid) not in self.book.recipes:
            return {"ok": False, "reason": f"레시피 {rid}번이 없다"}
        self.book.active = None if rid is None else int(rid)
        self.book_dirty = True
        return {"ok": True}

    def recipe_view(self) -> Dict[str, Any]:
        """패널이 그릴 것 전부. 상태 판정은 **오버레이 이전의 값**으로 한다."""
        if self.book is None or self.base is None:
            return {"path": "", "status": "", "applied": None, "recipes": [], "overlay": [], "dirty": False}

        paths = {p for r in self.book.recipes.values() for p in r.overrides} | set(self.overlay)
        current: Dict[str, Any] = {}
        for p in sorted(paths):
            try:
                current[p] = current_value(self.base, p)
            except Exception:
                current[p] = None

        def row(p, v):
            return {
                "path": p,
                "display": self.book.display_name(p),
                "value": v,
                "base": current.get(p),
                "differs": current.get(p) != v,
            }

        rows = []
        for rid in sorted(self.book.recipes):
            r = self.book.recipes[rid]
            rows.append(
                {
                    "id": rid,
                    "label": r.label,
                    "name": r.name,
                    "note": r.note,
                    "applied": rid == self.recipe_id,
                    "active": rid == self.book.active,
                    "overrides": [row(p, v) for p, v in sorted(r.overrides.items())],
                }
            )

        return {
            "path": self.book.path,
            "status": recipe_mod.status(self.book, current),
            "applied": self.recipe_id,
            "dirty": self.book_dirty,
            "recipes": rows,
            "overlay": [row(p, v) for p, v in sorted(self.overlay.items())],
        }

    # ── 실행 ────────────────────────────────────────────────────────────
    #
    # 편집기는 실행 경로를 따로 갖지 않는다. `vlmt run`을 그대로 하위 프로세스로 띄우고
    # 그 프로세스가 남기는 진행 스냅샷을 읽을 뿐이다. UI에서만 되는 실행은 존재하지 않는다.

    def run_command(self, limit: int = 8, debug_output: bool = False) -> List[str]:
        """띄울 명령. 사람이 터미널에 그대로 쳐도 같은 결과가 나와야 한다."""
        cmd = [
            sys.executable, "-m", "vlm_trainer.cli.main",
            "run", self.path,
            "--run-id", self.run_id or "ui",
            "--limit", str(int(limit)),
            "--trigger", "ui",
            "--skip-budget",
        ]
        if debug_output:
            cmd.append("--debug-output")
        for mod in self.extra_modules:
            cmd += ["--nodes", mod]
        # 화면에 보이는 값 그대로 돈다. 레시피 오버레이는 CLI의 --set 으로 넘긴다.
        for path, value in sorted(self.overlay.items()):
            cmd += ["--set", f"{path}={json.dumps(value, ensure_ascii=False)}"]
        return cmd

    def _spawn(self, cmd: List[str], phase: str) -> Dict[str, Any]:
        """하위 프로세스 하나를 띄운다. **버튼 하나가 CLI 명령 하나다** —
        편집기가 여러 명령을 엮어 돌리기 시작하면 그것이 UI 전용 실행 경로다."""
        if self.proc is not None and self.proc.poll() is None:
            return {"ok": False, "reason": f"이미 {self.phase or '작업'}이 돌고 있다"}

        run_dir = os.path.join(os.getcwd(), "runs", self.run_id)
        os.makedirs(run_dir, exist_ok=True)
        self.progress_path = os.path.join(run_dir, "progress.json")
        self.train_progress_path = os.path.join(run_dir, "train_progress.json")
        self.console_path = os.path.join(run_dir, f"{phase}.log")
        self.phase = phase
        self.stopped = False

        env = dict(os.environ)
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env["PYTHONPATH"] = os.pathsep.join([p for p in (pkg_root, env.get("PYTHONPATH", "")) if p])
        try:
            fh = open(self.console_path, "w", encoding="utf-8")
            self.proc = subprocess.Popen(
                cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=os.getcwd(), env=env
            )
        except OSError as exc:
            return {"ok": False, "reason": f"실행을 띄우지 못했다: {exc}"}
        return {"ok": True, "run_id": self.run_id, "phase": phase, "command": " ".join(cmd)}

    def _base_command(self, sub: str) -> List[str]:
        cmd = [sys.executable, "-m", "vlm_trainer.cli.main", sub, self.path,
               "--run-id", self.run_id or "ui"]
        for mod in self.extra_modules:
            cmd += ["--nodes", mod]
        for path, value in sorted(self.overlay.items()):
            cmd += ["--set", f"{path}={json.dumps(value, ensure_ascii=False)}"]
        return cmd

    def _ready_to_launch(self) -> Dict[str, Any]:
        if not self.valid:
            return {"ok": False, "reason": _short(self.error), "detail": self.error}
        if self.dirty:
            return {
                "ok": False,
                "reason": "저장하지 않은 변경이 있다 — 실행은 디스크의 스펙을 돈다",
                "detail": (
                    "편집기의 그래프와 project.yaml이 다르다. 지금 실행하면 화면과 다른 그래프가 돈다.\n"
                    "  이 검사가 없었다면: 방금 고친 값이 아니라 예전 값으로 돈 결과를 보고 판단하게 된다.\n"
                    "  Save를 먼저 누른다."
                ),
            }
        return {"ok": True}

    def materialize_start(self, limit: int = 0) -> Dict[str, Any]:
        """`vlmt materialize` — 학습이 읽을 shard를 만든다."""
        ready = self._ready_to_launch()
        if not ready["ok"]:
            return ready
        self.run_id = self.run_id or ("ui_" + time.strftime("%Y%m%dT%H%M%S"))
        cmd = self._base_command("materialize")
        if limit:
            cmd += ["--limit", str(int(limit))]
        return self._spawn(cmd, "materialize")

    def train_start(self) -> Dict[str, Any]:
        """`vlmt train` — 물질화된 shard로 학습한다.

        물질화를 대신 돌려 주지 않는다. 두 단계는 사람이 터미널에서 치는 두 명령이고,
        편집기가 그것을 엮으면 CLI에 없는 경로가 하나 생긴다.
        """
        ready = self._ready_to_launch()
        if not ready["ok"]:
            return ready
        if not self.run_id:
            return {
                "ok": False,
                "reason": "먼저 Materialize를 눌러야 한다",
                "detail": (
                    "학습은 물질화된 shard를 읽는다. 아직 이 세션에서 만든 것이 없다.\n"
                    "  Materialize를 먼저 누르거나, 터미널에서 vlmt materialize를 돌려라."
                ),
            }
        mat = os.path.join(os.getcwd(), "runs", self.run_id, "materialized")
        if not os.path.isdir(mat):
            return {
                "ok": False,
                "reason": "물질화된 shard가 없다",
                "detail": f"{mat}가 없다. Materialize를 먼저 누른다.",
            }
        return self._spawn(self._base_command("train"), "train")

    def run_start(self, limit: int = 8, debug_output: bool = False) -> Dict[str, Any]:
        ready = self._ready_to_launch()
        if not ready["ok"]:
            return ready
        self.run_id = "ui_" + time.strftime("%Y%m%dT%H%M%S")
        return self._spawn(self.run_command(limit, debug_output), "run")

    def run_stop(self) -> Dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            return {"ok": False, "reason": "돌고 있는 것이 없다"}
        self.stopped = True  # 사람이 멈춘 것과 죽은 것은 다르다
        self.proc.terminate()
        return {"ok": True}

    def run_state(self) -> Dict[str, Any]:
        """스냅샷을 읽어 카드에 칠할 상태로. 실행이 없으면 비어 있는 답이다."""
        from .render import state_of

        alive = self.proc is not None and self.proc.poll() is None
        out: Dict[str, Any] = {
            "running": alive,
            "run_id": self.run_id,
            "kind": self.phase,
            "train": self._read_train_progress(),
            "exit": None if (self.proc is None or alive) else self.proc.returncode,
            "stopped": self.stopped and self.proc is not None and not alive,
            "states": {},
            "processed": 0,
            "total": 0,
            "phase": "",
            "aborted": "",
            "quarantine": [],
        }
        # 물질화·학습 중에도 직전 실행이 남긴 카드 상태는 그대로 둔다.
        # 지우면 "아무것도 안 돌았다"로 읽히는데 그것은 사실이 아니다.
        data = self._read_progress()
        if data is None:
            if not alive and self.proc is not None and not self.stopped:
                out["console"] = self._console_tail()
            return out

        rep = runner_mod.report_from_snapshot(data)
        out.update(
            {
                "processed": data.get("processed", 0),
                "total": data.get("total", 0),
                "phase": data.get("phase", ""),
                "aborted": data.get("aborted", ""),
                "quarantine": data.get("quarantine", []),
                "quarantine_total": data.get("quarantine_total", 0),
                "cache": data.get("cache", {}),
                "previews": self._preview_urls(data.get("previews", {})),
            }
        )
        for nid in (self.compiled.order if self.compiled else []):
            state, extra = state_of(rep, nid)
            out["states"][nid] = {"state": state, "extra": extra}
        if not alive and self.proc is not None and self.proc.returncode and not self.stopped:
            out["console"] = self._console_tail()
        return out

    def _preview_urls(self, previews: Dict[str, Any]) -> Dict[str, Any]:
        """디스크 경로를 페이지가 부를 수 있는 주소로. 파일명만 남긴다 —
        경로를 그대로 실어 보내면 서버가 아무 파일이나 내주는 문이 된다."""
        out: Dict[str, Any] = {}
        for nid, p in previews.items():
            row = dict(p)
            path = str(row.pop("image_path", "") or "")
            row["image"] = f"/preview/{os.path.basename(path)}" if path else ""
            out[nid] = row
        return out

    @property
    def preview_dir(self) -> str:
        return os.path.join(os.getcwd(), "runs", self.run_id, "preview") if self.run_id else ""

    def preview_file(self, name: str) -> Optional[bytes]:
        """이번 실행의 미리보기 폴더 안에 있는 파일만 내준다."""
        base = self.preview_dir
        if not base or not name:
            return None
        p = os.path.abspath(os.path.join(base, os.path.basename(name)))
        if os.path.dirname(p) != os.path.abspath(base) or not p.lower().endswith(".png"):
            return None  # 경로를 벗어나려는 시도
        try:
            with open(p, "rb") as fh:
                return fh.read()
        except OSError:
            return None

    def _read_train_progress(self) -> Dict[str, Any]:
        """학습은 샘플이 아니라 step 단위로 움직인다. 진행 파일도 따로 있다."""
        if self.phase != "train" or not self.train_progress_path:
            return {}
        try:
            with open(self.train_progress_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _read_progress(self) -> Optional[Dict[str, Any]]:
        if not self.progress_path or not os.path.exists(self.progress_path):
            return None
        try:
            with open(self.progress_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None  # 원자 교체 사이에 읽었을 뿐이다. 다음 폴링에서 잡힌다

    def _console_tail(self, n: int = 12) -> str:
        try:
            with open(self.console_path, "r", encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-n:])
        except OSError:
            return ""

    def save(self) -> Dict[str, Any]:
        if self.compiled is None:
            return {"ok": False, "reason": "컴파일되지 않은 그래프는 저장하지 않는다"}
        if not self.valid:
            # 편집 중인 미완성 상태는 저장하지 않는다. 스펙 파일은 언제나 돌아가는 그래프여야 한다.
            return {"ok": False, "reason": _short(self.error), "detail": self.error}
        # **오버레이 이전의 그래프**를 쓴다. 레시피 값이 프로젝트 스펙에 스며들면
        # 레시피가 존재할 이유가 사라진다.
        spec = decompile(self.base, keep_procedures=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(dump_yaml(spec))
        os.replace(tmp, self.path)
        self.dirty = False
        written = [self.path]
        if self.book_dirty and self.book is not None:
            written.append(recipe_mod.save_book(self.book))
            self.book_dirty = False
        return {"ok": True, "path": self.path, "written": written}

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
            "procedures": [p["id"] for p in cg.procedures],
            "profile": self.graph.runtime_profile,
            "profiles": self.profiles(),
            "overlaid": [f"{n}:{p}" for (n, p) in self._overlay_paths()],
            "history": self.history_view(),
            "cursor": self.cursor,
        }
