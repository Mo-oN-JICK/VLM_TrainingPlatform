"""5노드(Line) 편집 — 설정을 고치고 저장하고 11노드로 내보낸다.

Line 파일이 원본이고 11노드 그래프는 그것을 펼친 파생물이다. 그래서 **그래프를 고치지 않고
Line 의 매핑을 고친 뒤 다시 펼쳐 컴파일한다.** 컴파일(strict)을 통과하지 못하는 편집은 그
자리에서 거부되고 원래 값이 남는다 — 화면에만 남은 값이 없게 한다.

고칠 수 있는 것은 설정 값뿐이다. 상자를 더하거나 배선을 바꾸는 길은 없다. 그것은 "11노드로
내보내기"의 몫이고, 내보낸 뒤에는 Line 과 연결이 끊긴다.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List

import string

from ..answer import table as answer_table
from ..core.compiler import compile_graph
from ..core.graph import GraphModel
from ..spec.decompile import dump_yaml
from ..spec.line import expand_data, expand_file
from ..spec.loader import build_project
from . import line_data as LD
from .editor_common import _msg, _short


class LineMixin:
    """`Editor` 가 Line 파일을 열었을 때만 쓰인다(`line_data` 가 있을 때)."""

    # ── 내부 ────────────────────────────────────────────────────────────
    def _line_refuse(self, why: str, detail: str = "") -> Dict[str, Any]:
        return {"ok": False, "reason": why, "detail": detail or why}

    def _line_build(self, data: Dict[str, Any]) -> GraphModel:
        d = os.path.dirname(self.path)
        return build_project(expand_data(data, d, os.path.basename(self.path)), d, os.path.basename(self.path))

    def _line_try(self, mutate: Any) -> Dict[str, Any]:
        """매핑에 편집을 적용해 보고, 펼쳐서 컴파일이 통과해야 받아들인다."""
        if getattr(self, "line_data", None) is None:
            return self._line_refuse("5노드(Line) 파일이 아니다")
        new = copy.deepcopy(self.line_data)
        try:
            mutate(new)
            g = self._line_build(new)
            compile_graph(g)  # strict. 미완성 상태를 저장하지 않는다
        except Exception as exc:
            return self._line_refuse(_short(_msg(exc)), _msg(exc))
        self.line_data, self.graph = new, g
        self._recompile()
        self.dirty = True
        return {"ok": True}

    @staticmethod
    def _station(data: Dict[str, Any], sid: str) -> Dict[str, Any]:
        for s in data.get("stations") or []:
            if str(s.get("id")) == sid:
                s.setdefault("params", {})
                return s
        raise KeyError(f"스테이션 {sid!r}이 없다")

    # ── 편집 ────────────────────────────────────────────────────────────
    def line_set(self, sid: str, name: str, value: Any) -> Dict[str, Any]:
        """스테이션이 노출한 설정 하나. 학습량과 직접 입력 횟수는 서로를 지운다."""
        def mut(d: Dict[str, Any]) -> None:
            p = self._station(d, sid)["params"]
            p[name] = value
            if name == "epochs" and value:
                p.pop("amount", None)

        return self._line_try(mut)

    def line_set_amount(self, sid: str, level: str) -> Dict[str, Any]:
        def mut(d: Dict[str, Any]) -> None:
            p = self._station(d, sid)["params"]
            p["amount"] = level
            p.pop("epochs", None)  # 단계를 고르면 직접 입력은 비운다

        return self._line_try(mut)

    def line_set_override(self, node: str, param: str, value: Any) -> Dict[str, Any]:
        """안쪽 상자 값 하나. 스테이션이 이미 노출한 값이면 그쪽을 고친다 — 같은 값이 두 곳에 적히면
        어느 쪽이 이기는지 화면에서 알 수 없다."""
        def mut(d: Dict[str, Any]) -> None:
            for st in describe_file_data(d, self.path):
                for name, s in st["settings"].items():
                    if s["target"] == [node, param]:
                        self._station(d, st["id"])["params"][name] = value
                        return
            ov = d.setdefault("overrides", [])
            for o in ov:
                if o.get("node") == node and o.get("param") == param:
                    o["value"] = value
                    return
            ov.append({"node": node, "param": param, "value": value})

        return self._line_try(mut)

    def line_clear_override(self, node: str, param: str) -> Dict[str, Any]:
        def mut(d: Dict[str, Any]) -> None:
            d["overrides"] = [o for o in d.get("overrides") or []
                              if not (o.get("node") == node and o.get("param") == param)]
            if not d["overrides"]:
                d.pop("overrides")

        return self._line_try(mut)

    def line_apply_data(self, f: Dict[str, Any], schema: Any = None) -> Dict[str, Any]:
        """데이터 대화상자의 적용. 여러 값을 **한 번에** 고친다 — 표 파일 하나를 바꾸면 열 이름이
        전부 어긋나므로, 하나씩 고치면 중간 상태가 거부된다.

        `schema` 가 있으면 정답 항목 표를 이 프로젝트 전용 파일로 먼저 쓴다. 컴파일이 그 파일을 읽기
        때문이다. 거부되면 파일을 원래대로 되돌린다.
        """
        if getattr(self, "line_data", None) is None:
            return self._line_refuse("5노드(Line) 파일이 아니다")
        d = os.path.dirname(self.path)
        wrote = None
        path = f.get("schema_path", "")
        try:
            if schema is not None:
                built = answer_table.build(
                    schema["rows"], schema_id=schema.get("id") or str(self.line_data.get("id") or "answer"),
                    render=schema.get("render", "tagged"), max_tokens=schema.get("max_tokens", 320),
                    step_order=schema.get("step_order", "strict"), vocabulary=schema.get("vocabulary", "closed"))
                path = LD.owned_schema_path(path, str(self.line_data.get("id") or ""))
                full = LD.resolve(d, path)
                before = None
                if os.path.exists(full):
                    with open(full, "rb") as fh:
                        before = fh.read()
                os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
                with open(full, "w", encoding="utf-8") as fh:
                    fh.write(answer_table.dump(built))
                wrote = (full, before)
        except Exception as exc:
            return self._line_refuse(_short(_msg(exc)), _msg(exc))

        def mut(data: Dict[str, Any]) -> None:
            ss = data.setdefault("sample_space", {})
            ss["index"], ss["key"], ss["splits"] = f["index"], f["key"], f["splits"]
            img, qst, ans = (LD.station(data, n) for n in ("image", "question", "answer"))
            missing = [n for n, s in (("이미지", img), ("질문 형태", qst), ("정답 형태", ans)) if s is None]
            if missing:
                raise ValueError(f"이 Line 에 {', '.join(missing)} 스테이션이 없어 열을 지정할 수 없다")
            for st in (img, qst, ans):
                st.setdefault("params", {})
            if f.get("question_column"):
                qst["params"]["template"] = "{" + f["question_column"] + "}"
            fields = [x for _, x, _, _ in string.Formatter().parse(str(qst["params"].get("template", ""))) if x]
            cols: List[str] = []
            for c in [*fields, *f["answers"].values()]:
                if c and c not in cols:
                    cols.append(c)
            img["params"]["image_column"] = f["image_column"]
            img["params"]["columns"] = cols
            ans["params"]["mapping"] = dict(f["answers"])
            if path:
                ans["params"]["schema_path"] = path

        res = self._line_try(mut)
        if not res.get("ok") and wrote is not None:
            full, before = wrote
            if before is None:
                os.remove(full)
            else:
                with open(full, "wb") as fh:
                    fh.write(before)
        return res

    # ── 저장·내보내기 ───────────────────────────────────────────────────
    def line_save(self) -> Dict[str, Any]:
        """Line 파일을 다시 쓴다. 주석과 따옴표·키 순서는 보존되지 않는다(의미와 해시는 보존)."""
        if getattr(self, "line_data", None) is None:
            return self._line_refuse("5노드(Line) 파일이 아니다")
        if self.compiled is None or not self.valid:
            return self._line_refuse(_short(self.error), self.error)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(dump_yaml(self.line_data))
        os.replace(tmp, self.path)
        self.dirty = False
        return {"ok": True, "path": self.path}

    def line_export_path(self) -> str:
        i = str(self.line_data.get("id") or "line") if getattr(self, "line_data", None) else "line"
        return os.path.join(os.path.dirname(self.path), f"{i}_11nodes.yaml")

    def line_export(self) -> Dict[str, Any]:
        """11노드 Project 로 내보낸다. 이미 있으면 덮지 않는다 — 고친 11노드 파일을 날리는 길이다."""
        if getattr(self, "line_data", None) is None:
            return self._line_refuse("5노드(Line) 파일이 아니다")
        if self.dirty:
            return self._line_refuse("저장하지 않은 변경이 있다", "내보내는 것은 디스크의 Line 이다. 먼저 저장하세요.")
        out = self.line_export_path()
        if os.path.exists(out):
            return self._line_refuse(f"이미 있다: {os.path.basename(out)}",
                                     f"{out} 가 있다. 덮어쓰면 그 파일에서 고친 것이 사라진다. 이름을 바꾸거나 지우고 다시 하세요.")
        text = dump_yaml(expand_file(self.path))
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        return {"ok": True, "path": out}

    # ── 보여 주기 ───────────────────────────────────────────────────────
    def line_describe(self) -> Dict[str, Any]:
        return describe_file_data(self.line_data, self.path, whole=True)


def describe_file_data(data: Dict[str, Any], path: str, whole: bool = False) -> Any:
    """`describe_file` 의 디스크 대신 메모리 매핑 판. 편집 중인 값을 보여 줘야 한다."""
    from ..spec.line import expand_data as _expand

    stations: List[Dict[str, Any]] = []
    _expand(data, os.path.dirname(path), os.path.basename(path), check=False, info=stations)
    if not whole:
        return stations
    ss = data.get("sample_space") or {}
    return {"id": str(data.get("id", "")), "name": str(data.get("name", "")), "stations": stations,
            "table": {"index": str(ss.get("index", "")), "key": str(ss.get("key", "")),
                      "splits": dict(ss.get("splits") or {})},
            "overrides": list(data.get("overrides") or [])}
