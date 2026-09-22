"""프로젝트 설정 — 배치·경계·프로파일·샘플 공간

`Editor` 가 한 클래스에 아홉 관심사를 담고 있어 믹스인으로 갈랐다.
**메서드 이름과 동작은 하나도 바뀌지 않는다** — `Editor` 가 이것을 상속한다.

여기 있는 메서드는 `self._try` · `self._recompile` · `self.graph` 처럼 Editor 본체가
들고 있는 것을 쓴다. 그것이 믹스인이 독립 클래스가 아닌 이유다.
"""

from __future__ import annotations

from .editor_common import _msg


import yaml

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

from ..engine import runner as runner_mod
from ..engine import samples as samples_mod
from ..spec import recipe as recipe_mod
from .editor_paths import LAYOUT_FILE

class ProjectMixin:
    """프로젝트 설정 — 배치·경계·프로파일·샘플 공간"""

    # ── 캔버스 배치 ──────────────────────────────────────────────────────
    #
    # 좌표는 **스펙에 들어가지 않는다.** 무엇이 실행되는지와 무관한 값이라
    # spec_hash 에 섞이면 상자를 옮긴 것만으로 캐시가 통째로 무효가 된다.
    # 스펙 옆의 `layout.yaml` 에 따로 둔다.

    @property
    def layout_path(self) -> str:
        return os.path.join(os.path.dirname(self.path), LAYOUT_FILE)

    def _load_layout(self) -> None:
        try:
            with open(self.layout_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, ValueError):
            return
        for nid, xy in (data.get("nodes") or {}).items():
            try:
                self.layout[str(nid)] = (int(xy[0]), int(xy[1]))
            except (TypeError, ValueError, IndexError):
                continue

    def _save_layout(self) -> None:
        try:
            if not self.layout:
                if os.path.exists(self.layout_path):
                    os.remove(self.layout_path)
                return
            payload = {"kind": "CanvasLayout", "nodes": {k: list(v) for k, v in sorted(self.layout.items())}}
            tmp = self.layout_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)
            os.replace(tmp, self.layout_path)
        except OSError:
            pass  # 자리는 편의일 뿐이다. 못 적어도 편집을 막지 않는다

    def move_node(self, node_id: str, x: int, y: int) -> Dict[str, Any]:
        """상자를 옮긴다. 스펙도 History도 건드리지 않는다 — 자리는 의미가 없다."""
        known = set(self.compiled.nodes if self.compiled else ()) | {
            p["id"] for p in (self.compiled.procedures if self.compiled else [])
        }
        if node_id not in known and node_id not in self.layout:
            return {"ok": False, "reason": f"그런 상자가 없다: {node_id}"}
        self.layout[node_id] = (max(0, int(x)), max(0, int(y)))
        self._save_layout()
        return {"ok": True, "x": self.layout[node_id][0], "y": self.layout[node_id][1]}

    def reset_layout(self) -> Dict[str, Any]:
        """자동 정렬로 되돌린다."""
        self.layout.clear()
        self._save_layout()
        return {"ok": True}

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

