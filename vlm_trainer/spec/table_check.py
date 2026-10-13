"""Line 의 표 검사 — 컴파일이 통과시키고 첫 샘플에서 죽는 것을 펼치는 자리에서 막는다.

열 이름 하나가 어긋나면 그래프는 멀쩡히 컴파일되고, 실행 첫 샘플(또는 학습 직전)에서야 죽는다.
그때는 사전 준비 시간이 이미 날아간 뒤다. 표의 첫 줄을 읽어 미리 대조한다.
"""

from __future__ import annotations

import os
import string
from typing import Any, Dict, List, Set

from ..answer.schema import AnswerSchema
from ..core.errors import SpecError
from ..engine import samples


def _fail(where: str, what: str, missed: str, fix: str) -> SpecError:
    return SpecError(f"{where}: {what}\n  안 잡혔다면: {missed}\n  {fix}")


def _fields(template: str) -> Set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def check_table(project: Dict[str, Any], source_dir: str, where: str) -> None:
    """펼친 Project 매핑을 표와 대조한다. 어긋나면 SpecError."""
    ss = project.get("sample_space") or {}
    splits = dict(ss.get("splits") or {})
    strategy = str(splits.get("strategy") or "")
    if strategy not in ("column", "random", "group_holdout"):
        raise _fail(
            where,
            f"학습/검증 나누는 방법이 정해지지 않았습니다 (splits.strategy = {strategy!r}).",
            "검증용이 0건이 되어도 실행은 돌고, 검증 단계가 빈 결과로 끝납니다.",
            "column(표의 구분 열 사용) / random(자동, 행마다) / group_holdout(자동, 그룹 열 기준) 중 하나를 적으세요.",
        )

    space = samples.load(ss, source_dir)  # 열·그룹 검사와 split 배정이 여기서 돈다
    if not space.rows:
        raise _fail(where, f"표에 건이 없습니다: {space.index_path}",
                    "빈 데이터로 학습이 시작됩니다.", "표에 행을 채우거나 filter 를 확인하세요.")

    header: Set[str] = set(space.rows[0])
    for r in space.rows[1:]:
        header &= set(r)
    header.discard("_split")
    shown = sorted(header)

    def need(col: str, role: str) -> None:
        if col not in header:
            raise _fail(
                where, f"{role} 열 {col!r}이 표에 없습니다. 표의 열: {shown}",
                "그래프는 컴파일을 통과하고 첫 샘플에서 KeyError 로 멈춥니다. 사전 준비 시간이 먼저 쓰입니다.",
                "열 이름을 표와 똑같이 맞추세요(앞뒤 공백과 대소문자 포함).")

    need(space.key, "건 구분(key)")
    if strategy == "column":
        need(str(splits.get("column", "split")), "학습/검증 구분")
    if strategy == "group_holdout":
        need(str(splits.get("group_by") or space.key), "그룹")

    if strategy != "column":
        names = list(splits.get("ratios") or {"train": 1.0})
        empty = [n for n in names if not any(r["_split"] == n for r in space.rows)]
        if empty:
            raise _fail(
                where, f"{empty}에 배정된 건이 0입니다 (전체 {len(space.rows)}건, 비율 {splits.get('ratios')}).",
                "검증용이 비어 학습만 돌고 검증 단계가 빈 결과로 끝납니다.",
                "건수를 늘리거나 비율을 키우세요. 그룹이 적다면 그룹 열을 빼고 random 으로 나누세요.")

    nodes: List[Dict[str, Any]] = list(project.get("nodes") or [])
    meta_cols: List[str] = []
    for n in nodes:
        t, p = str(n["type"]), n.get("params") or {}
        if t.startswith("source.image"):
            need(str(p.get("column", "image_path")), "사진")
        if t.startswith("source.metadata"):
            meta_cols = [str(c) for c in p.get("columns") or []]
            for c in meta_cols:
                need(c, "메타데이터")

    for pr in project.get("procedures") or []:
        ref, p = str(pr["ref"]), pr.get("params") or {}
        if ref.startswith("prompt_build"):
            for f in _fields(str(p.get("template", ""))):
                if f not in meta_cols:
                    raise _fail(
                        where, f"질문 형태의 {{{f}}}가 메타데이터 열 목록 {meta_cols}에 없습니다.",
                        "자리가 채워지지 않은 채로 모델에게 건네지거나 KeyError 로 멈춥니다.",
                        f"이미지 스테이션의 columns 에 {f!r}를 넣으세요.")

    schema_path = ""
    for n in nodes:
        if str(n["type"]).startswith("schema.define"):
            schema_path = str((n.get("params") or {}).get("path", ""))
    for pr in project.get("procedures") or []:
        if not str(pr["ref"]).startswith("answer_build"):
            continue
        mapping = dict((pr.get("params") or {}).get("mapping") or {})
        steps = [s.id for s in AnswerSchema.load(_asset(schema_path, source_dir, where)).steps] if schema_path else list(mapping)
        for step in steps:
            expr = mapping.get(step, step)  # 매핑이 없으면 열 이름이 항목 이름과 같아야 한다
            cols = [expr] if isinstance(expr, str) and expr.isidentifier() else []
            if isinstance(expr, dict):
                cols = [v for v in expr.values() if isinstance(v, str) and v.isidentifier()]
            for c in cols:
                need(c, f"정답 항목 {step!r}의")
                if c not in meta_cols:
                    raise _fail(
                        where, f"정답 항목 {step!r}의 열 {c!r}이 메타데이터 열 목록 {meta_cols}에 없습니다.",
                        "정답이 비어 쓰이거나 항목 이름이 그대로 정답 글자로 학습됩니다.",
                        f"이미지 스테이션의 columns 에 {c!r}를 넣으세요.")


def _asset(path: str, spec_dir: str, where: str) -> str:
    """스키마 같은 스펙 자산은 스펙 파일이 있는 폴더 기준이다(`RunCtx.asset`). 사진 같은 데이터는 표 기준."""
    full = path if os.path.isabs(path) else os.path.normpath(os.path.join(spec_dir, path))
    if not os.path.exists(full):
        raise _fail(where, f"정답 형태 파일이 없습니다: {full}",
                    "schema.define 이 실행 첫 샘플에서 멈춥니다.", "정답 형태 경로를 확인하세요.")
    return full
