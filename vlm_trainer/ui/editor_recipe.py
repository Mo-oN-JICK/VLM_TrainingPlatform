"""Parameter Recipe — 값을 덮는 오버레이

`Editor` 가 한 클래스에 아홉 관심사를 담고 있어 믹스인으로 갈랐다.
**메서드 이름과 동작은 하나도 바뀌지 않는다** — `Editor` 가 이것을 상속한다.

여기 있는 메서드는 `self._try` · `self._recompile` · `self.graph` 처럼 Editor 본체가
들고 있는 것을 쓴다. 그것이 믹스인이 독립 클래스가 아닌 이유다.
"""

from __future__ import annotations

from .editor_common import _msg, _short




import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

from ..core.compiler import current_value, override_target
from ..engine import runner as runner_mod
from ..engine import samples as samples_mod
from ..spec import recipe as recipe_mod

class RecipeMixin:
    """Parameter Recipe — 값을 덮는 오버레이"""

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

