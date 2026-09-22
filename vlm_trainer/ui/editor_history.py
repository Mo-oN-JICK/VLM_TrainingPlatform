"""편집 이력 — 시점 기록과 되감기

`Editor` 가 한 클래스에 아홉 관심사를 담고 있어 믹스인으로 갈랐다.
**메서드 이름과 동작은 하나도 바뀌지 않는다** — `Editor` 가 이것을 상속한다.

여기 있는 메서드는 `self._try` · `self._recompile` · `self.graph` 처럼 Editor 본체가
들고 있는 것을 쓴다. 그것이 믹스인이 독립 클래스가 아닌 이유다.
"""

from __future__ import annotations

from ..spec.loader import build_project

from .editor_common import HistoryEntry, _short



import difflib
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
from ..spec.decompile import decompile, dump_yaml
from .editor_paths import HISTORY_FILE, HISTORY_STORE, HISTORY_WINDOW

class HistoryMixin:
    """편집 이력 — 시점 기록과 되감기"""

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

