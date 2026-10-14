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
from ..core.errors import SpecError
from ..core.graph import GraphModel
from ..engine import folder_table, table_edit, xlsx
from ..engine import samples as samples_mod
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
            return self._line_refuse("5노드(Line) 파일이 아닙니다")
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
        raise KeyError(f"스테이션 {sid!r}이 없습니다")

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
            return self._line_refuse("5노드(Line) 파일이 아닙니다")
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
            # 폴더 입력 방식이면 고른 폴더를 Line 층에 적는다(SampleSpace 가 아니라 spec_hash 는 그대로).
            # 표 방식으로 적용하면 지운다 — 그때부터 표가 원본이다.
            if f.get("folder"):
                data["folder"] = f["folder"]
            else:
                data.pop("folder", None)
            img, qst, ans = (LD.station(data, n) for n in ("image", "question", "answer"))
            missing = [n for n, s in (("이미지", img), ("질문 형태", qst), ("정답 형태", ans)) if s is None]
            if missing:
                raise ValueError(f"이 Line 에 {', '.join(missing)} 스테이션이 없어 열을 지정할 수 없습니다")
            for st in (img, qst, ans):
                st.setdefault("params", {})
            if f.get("question") is not None:
                qst["params"]["template"], qst["params"]["policy"] = LD.parse_question(f["question"])
            elif f.get("question_column"):
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

    # ── 폴더 입력 ───────────────────────────────────────────────────────
    def _folder_full(self) -> str:
        f = str((getattr(self, "line_data", None) or {}).get("folder") or "")
        return LD.resolve(os.path.dirname(self.path), f) if f else ""

    @staticmethod
    def _answer_cols(data: Dict[str, Any]) -> List[str]:
        ans = LD.station(data, "answer") or {}
        return [str(c) for c in ((ans.get("params") or {}).get("mapping") or {}).values()]

    def line_sync_folder(self) -> Dict[str, Any]:
        """폴더 입력 방식이면 폴더에 맞춰 표를 맞춘다(Q67). 열 때와 다시 읽을 때 부른다."""
        full = self._folder_full()
        if not full:
            return {"ok": True}
        try:
            t = table_edit.target(self.line_data, os.path.dirname(self.path))
            return dict(folder_table.sync(full, t["path"], self._answer_cols(self.line_data)), ok=True)
        except SpecError as exc:
            return self._line_refuse(_short(_msg(exc)), _msg(exc))

    def _with_csv_restore(self, csv_path: str, work: Any) -> Dict[str, Any]:
        """표를 먼저 고치고 Line 편집을 시도한다. 편집이 거부되면 표를 원래대로 되돌린다."""
        before = open(csv_path, "rb").read() if os.path.exists(csv_path) else None
        res = work()
        if not res.get("ok"):
            if before is None:
                if os.path.exists(csv_path):
                    os.remove(csv_path)
            else:
                with open(csv_path, "wb") as fh:
                    fh.write(before)
        return res

    def _schema_meta(self, rows: List[Dict[str, str]]) -> Dict[str, Any]:
        s = LD.form(self.line_data, os.path.dirname(self.path))["schema"]
        return {"rows": rows, "id": s["id"], "render": s["render"], "max_tokens": s["max_tokens"],
                "step_order": s["step_order"], "vocabulary": s["vocabulary"]}

    def line_use_folder(self, folder: str, question: str, rows: List[Dict[str, str]],
                        val_pct: int = 20) -> Dict[str, Any]:
        """폴더 입력 방식으로 바꾼다(Q74 a). 폴더·질문 문장·정답 항목을 **한 번에** 받아야 컴파일이 통과한다.

        `line.yaml` 옆 `data.csv` 를 폴더에 맞춰 만들고(정답은 빈 칸), 나누기는 무작위 비율이다.
        """
        if getattr(self, "line_data", None) is None:
            return self._line_refuse("5노드(Line) 파일이 아닙니다")
        d = os.path.dirname(self.path)
        full = os.path.abspath(LD.resolve(d, folder))
        try:
            rel = os.path.relpath(full, d)
        except ValueError:      # 다른 드라이브면 상대 경로가 없다
            rel = full
        rel = rel.replace(os.sep, "/")
        csv_path = os.path.join(d, "data.csv")
        ids = [str(r.get("항목", "")).strip() for r in rows]
        counts: Dict[str, Any] = {}

        def work() -> Dict[str, Any]:
            try:
                counts.update(folder_table.sync(full, csv_path, [i for i in ids if i]))
            except SpecError as exc:
                return self._line_refuse(_short(_msg(exc)), _msg(exc))
            cur = LD.form(self.line_data, d)
            f = {"index": "data.csv", "key": folder_table.KEY_COL, "image_column": folder_table.IMAGE_COL,
                 "splits": LD.make_splits("random", val_pct=int(val_pct), seed=cur["split"]["seed"]),
                 "question_column": None, "question": question, "answers": {i: i for i in ids},
                 "schema_path": cur["schema_path"], "folder": rel}
            return self.line_apply_data(f, schema=self._schema_meta(rows))

        res = self._with_csv_restore(csv_path, work)
        return dict(res, counts=counts) if res.get("ok") else res

    def line_set_question(self, text: str) -> Dict[str, Any]:
        """질문 문장(`{image}` 는 맨 앞이나 맨 뒤). 문장에 표의 열 `{열}` 을 넣으면 그 열도 읽게 한다."""
        def mut(d: Dict[str, Any]) -> None:
            qst, img = LD.station(d, "question"), LD.station(d, "image")
            p = qst.setdefault("params", {})
            p["template"], p["policy"] = LD.parse_question(text)
            fields = [x for _, x, _, _ in string.Formatter().parse(p["template"]) if x]
            cols: List[str] = []
            for c in [*fields, *self._answer_cols(d)]:
                if c and c not in cols:
                    cols.append(c)
            img.setdefault("params", {})["columns"] = cols

        return self._line_try(mut)

    def line_set_val_pct(self, pct: int) -> Dict[str, Any]:
        """폴더 입력 방식의 검증 비율(%). 시드는 그대로 둔다 — 같은 비율이면 같게 나뉜다."""
        def mut(d: Dict[str, Any]) -> None:
            ss = d.setdefault("sample_space", {})
            seed = int((ss.get("splits") or {}).get("seed", LD.DEFAULT_SEED))
            ss["splits"] = LD.make_splits("random", val_pct=int(pct), seed=seed)

        return self._line_try(mut)

    def line_set_schema(self, rows: List[Dict[str, str]], columns: Any = None) -> Dict[str, Any]:
        """③ 의 정답 항목 표를 적용한다(Q64). `columns` 는 표 방식의 항목 → 표의 열(없으면 지금 값, 그것도
        없으면 항목 이름). 폴더 방식은 열 이름이 항목 이름이고, 새 항목의 열을 표에 먼저 덧붙인다."""
        if getattr(self, "line_data", None) is None:
            return self._line_refuse("5노드(Line) 파일이 아닙니다")
        d = os.path.dirname(self.path)
        cur = LD.form(self.line_data, d)
        folder = self._folder_full()
        ids = [str(r.get("항목", "")).strip() for r in rows]
        given = dict(columns or {})
        answers = {i: (i if folder else (given.get(i) or cur["answers"].get(i) or i)) for i in ids}
        f = {"index": cur["index"], "key": cur["key"], "image_column": cur["image_column"],
             "splits": dict((self.line_data.get("sample_space") or {}).get("splits") or {}),
             "question_column": cur["question_column"], "answers": answers, "schema_path": cur["schema_path"],
             "folder": self.line_data.get("folder") or ""}
        schema = self._schema_meta(rows)
        if not folder:
            return self.line_apply_data(f, schema=schema)
        csv_path = table_edit.target(self.line_data, d)["path"]

        def work() -> Dict[str, Any]:
            try:
                folder_table.sync(folder, csv_path, [i for i in ids if i])
            except SpecError as exc:
                return self._line_refuse(_short(_msg(exc)), _msg(exc))
            return self.line_apply_data(f, schema=schema)

        return self._with_csv_restore(csv_path, work)

    # ── ③ 건별 정답 ─────────────────────────────────────────────────────
    def line_answers(self) -> Dict[str, Any]:
        """③ 건별 정답 표. 표에서 제외한 건은 뺀다. 저장 전에 고친 값(`answer_edits`)을 얹어 보인다.

        표가 CSV 면 고칠 수 있고, 엑셀·JSONL 이면 보기 전용이다(이유를 `why` 에 담는다, Q73).
        """
        d = os.path.dirname(self.path)
        try:
            t = table_edit.target(self.line_data, d)
            index = str((self.line_data.get("sample_space") or {}).get("index", ""))
            path, sheet = xlsx.split_sheet(index)
            path = LD.resolve(d, path)
            rows, _n = table_edit.drop_excluded(samples_mod.read_table(path, sheet), path, t["key"])
        except SpecError as exc:
            return self._line_refuse(_short(_msg(exc)), _msg(exc))
        why = ""
        try:
            table_edit._need_csv(path)
        except SpecError as exc:
            why = _msg(exc)
        edits = getattr(self, "answer_edits", None) or {}
        out = []
        for r in rows:
            key = str(r.get(t["key"], ""))
            vals = {str(k): ("" if v is None else str(v)) for k, v in r.items()}
            vals.update(edits.get(key, {}))
            out.append({"key": key, "values": vals})
        return {"ok": True, "key": t["key"], "path": path, "editable": not why, "why": why, "rows": out,
                "cols": self._answer_cols(self.line_data)}

    def line_set_answer(self, sample_key: str, col: str, value: str) -> Dict[str, Any]:
        """건 하나의 정답 칸. 표에 바로 쓰지 않고 `저장` 때 Line 과 함께 쓴다(Q71)."""
        if getattr(self, "answer_edits", None) is None:
            self.answer_edits = {}
        self.answer_edits.setdefault(str(sample_key), {})[str(col)] = str(value)
        self.dirty = True
        return {"ok": True}

    def line_fill_from_folder(self, item: str) -> Dict[str, Any]:
        """빈 정답 칸만 하위 폴더 이름으로 채운다(Q70). 입력해 둔 값은 덮지 않고, 목록 항목이면
        허용값에 없는 폴더 이름은 채우지 않는다. 채운 값은 저장 때 표에 쓴다."""
        if not self._folder_full():
            return self._line_refuse("폴더 입력 방식이 아닙니다",
                                     "하위 폴더 이름으로 채우기는 사진 폴더를 고른 프로젝트에서만 됩니다.")
        got = self.line_answers()
        if not got.get("ok"):
            return got
        rows = LD.form(self.line_data, os.path.dirname(self.path))["schema"]["rows"]
        spec = next((r for r in rows if r["항목"] == item), None)
        if spec is None:
            return self._line_refuse(f"정답 항목 {item!r}가 없습니다",
                                     f"정답 항목 {item!r}가 없습니다 (있는 항목: {[r['항목'] for r in rows]}).\n"
                                     "  안 잡혔다면: 채운 값이 어느 칸에도 쓰이지 않습니다.\n  항목 이름을 확인하세요.")
        allowed = ([v.strip() for v in spec["허용값"].split(answer_table.SEP) if v.strip()]
                   if spec["종류"] == "목록" else [])
        filled = not_allowed = kept = 0
        for row in got["rows"]:
            name = folder_table.folder_name(row["key"])
            if not name:
                continue
            if str(row["values"].get(item, "")).strip():
                kept += 1
            elif allowed and name not in allowed:
                not_allowed += 1
            else:
                self.line_set_answer(row["key"], item, name)
                filled += 1
        return {"ok": True, "filled": filled, "not_allowed": not_allowed, "kept": kept}

    # ── 저장·내보내기 ───────────────────────────────────────────────────
    def line_row(self, sample_key: str) -> Dict[str, Any]:
        """NG 건의 표 한 행. 화면이 고칠 칸을 보여 줄 때 쓴다. CSV 가 아니면 이유만 돌려준다."""
        try:
            t = table_edit.target(self.line_data, os.path.dirname(self.path))
            return {"ok": True, "key": t["key"], "values": table_edit.row_values(t["path"], t["key"], sample_key)}
        except SpecError as exc:
            return self._line_refuse(_short(_msg(exc)), _msg(exc))

    def line_fix_row(self, sample_key: str, edits: Dict[str, str]) -> Dict[str, Any]:
        """표(CSV)의 그 건 칸을 고쳐 쓴다. 다시 읽는 것은 부른 쪽이 한다."""
        try:
            t = table_edit.target(self.line_data, os.path.dirname(self.path))
            return {"ok": True, "changed": table_edit.set_cells(t["path"], t["key"], sample_key, edits)}
        except SpecError as exc:
            return self._line_refuse(_short(_msg(exc)), _msg(exc))

    def line_exclude_row(self, sample_key: str, on: bool = True) -> Dict[str, Any]:
        try:
            t = table_edit.target(self.line_data, os.path.dirname(self.path))
            return {"ok": True, "changed": table_edit.set_excluded(t["path"], t["key"], sample_key, on)}
        except SpecError as exc:
            return self._line_refuse(_short(_msg(exc)), _msg(exc))

    def line_save(self) -> Dict[str, Any]:
        """Line 파일을 다시 쓴다. 주석과 따옴표·키 순서는 보존되지 않는다(의미와 해시는 보존)."""
        if getattr(self, "line_data", None) is None:
            return self._line_refuse("5노드(Line) 파일이 아닙니다")
        if self.compiled is None or not self.valid:
            return self._line_refuse(_short(self.error), self.error)
        # ③ 에서 고친 정답을 먼저 표에 쓴다. 표를 못 쓰면(엑셀에서 열림 등) Line 도 쓰지 않는다.
        edits = getattr(self, "answer_edits", None) or {}
        if edits:
            try:
                t = table_edit.target(self.line_data, os.path.dirname(self.path))
                for key in list(edits):
                    table_edit.set_cells(t["path"], t["key"], key, edits[key])
                    del edits[key]
            except SpecError as exc:
                return self._line_refuse(_short(_msg(exc)), _msg(exc))
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
            return self._line_refuse("5노드(Line) 파일이 아닙니다")
        if self.dirty:
            return self._line_refuse("저장하지 않은 변경이 있습니다", "내보내는 것은 디스크의 Line 입니다. 먼저 저장하세요.")
        out = self.line_export_path()
        if os.path.exists(out):
            return self._line_refuse(f"이미 있습니다: {os.path.basename(out)}",
                                     f"{out} 가 있습니다. 덮어쓰면 그 파일에서 고친 것이 사라집니다. 이름을 바꾸거나 지우고 다시 하세요.")
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
