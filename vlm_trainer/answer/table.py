"""정답 형태 표 → AnswerSchema.

사용자는 표 한 장에 행마다 항목 이름 / 종류 / 허용값을 적는다. 화면의 표 편집과 CLI 가 같은 이 함수를
쓴다 — 화면에서만 되는 입력 경로는 없다. 결과는 기존 스키마 파일(YAML)과 같은 모양이다.

종류는 셋이다: **목록**(허용값 중 하나) · **숫자** · **문장**(자유). 목록만 허용값을 가진다.
"""

from __future__ import annotations

import csv
import re
from typing import Any, Dict, List, Sequence

import yaml

from ..core.errors import SpecError

KINDS = {"목록": "enum", "숫자": "float", "문장": "text"}
SEP = "|"
HEADER = ("항목", "이름", "종류", "허용값")
_ID = re.compile(r"^[A-Za-z_가-힣][A-Za-z0-9_가-힣]*$")


def _fail(row: int, what: str, missed: str, fix: str) -> SpecError:
    return SpecError(f"정답 형태 {row}번째 행: {what}\n  안 잡혔다면: {missed}\n  {fix}")


def read_rows(path: str) -> List[Dict[str, str]]:
    """CSV(UTF-8, BOM 가능)에서 행을 읽는다. 첫 줄은 `항목,이름,종류,허용값`."""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        got = [str(n).strip() for n in (rd.fieldnames or [])]
        missing = [h for h in HEADER if h not in got]
        if missing:
            raise SpecError(f"{path}: 첫 줄에 {missing} 열이 없다 (있는 열: {got}).\n"
                            f"  안 잡혔다면: 열이 밀려 읽혀 항목 이름 자리에 허용값이 들어간다.\n"
                            f"  첫 줄을 {','.join(HEADER)} 로 맞추세요.")
        return [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in rd
                if any((v or "").strip() for v in r.values())]


def build(rows: Sequence[Dict[str, str]], *, schema_id: str, render: str = "tagged",
          max_tokens: int = 320) -> Dict[str, Any]:
    """행 목록을 스키마 매핑으로. 어긋나면 SpecError 로 **어느 행**인지 말한다."""
    if not schema_id.strip():
        raise SpecError("정답 형태의 id 가 비었다 — 추론 계약과 캐시에 이 이름이 쓰인다.")
    if not rows:
        raise SpecError("정답 형태에 항목이 하나도 없다.\n  안 잡혔다면: 정답이 빈 문장으로 학습된다.\n"
                        "  항목을 한 줄 이상 적으세요.")
    steps: List[Dict[str, Any]] = []
    seen = set()
    for i, r in enumerate(rows, 1):
        sid = (r.get("항목") or "").strip()
        if not _ID.match(sid):
            raise _fail(i, f"항목 {sid!r}를 태그 이름으로 쓸 수 없다.",
                        "정답이 <항목>값</항목> 꼴로 쓰이고 읽히는데 이름이 깨져 파싱이 어긋난다.",
                        "글자·숫자·밑줄만 쓰고 숫자로 시작하지 마세요(예: part_type).")
        if sid in seen:
            raise _fail(i, f"항목 {sid!r}가 앞에서도 나왔다.",
                        "같은 태그가 두 번 나와 둘째 값이 첫째를 덮는다.", "항목 이름을 서로 다르게 하세요.")
        seen.add(sid)
        kind = (r.get("종류") or "").strip()
        if kind not in KINDS:
            raise _fail(i, f"종류 {kind!r}는 없는 값이다 (목록 | 숫자 | 문장).",
                        "종류를 짐작해 학습하면 형식 검사가 엉뚱한 기준으로 돈다.", "셋 중 하나를 적으세요.")
        vals = [v.strip() for v in (r.get("허용값") or "").split(SEP) if v.strip()]
        step: Dict[str, Any] = {"id": sid, "label": (r.get("이름") or "").strip() or sid, "type": KINDS[kind]}
        if kind == "목록":
            if not vals:
                raise _fail(i, f"항목 {sid!r}는 목록인데 허용값이 없다.",
                            "허용값 밖의 답도 형식 검사를 통과하고, 정답 노출 검사가 볼 낱말이 없다.",
                            f"허용값을 {SEP} 로 구분해 적으세요(예: 가로{SEP}세로{SEP}정사각).")
            dup = sorted({v for v in vals if vals.count(v) > 1})
            if dup:
                raise _fail(i, f"항목 {sid!r}의 허용값이 겹친다 {dup}.", "한 값이 두 번 세어져 분포가 어긋난다.",
                            "겹친 값을 하나로 줄이세요.")
            step["values"] = vals
        elif vals:
            raise _fail(i, f"항목 {sid!r}는 {kind}인데 허용값 {vals}이 적혀 있다.",
                        "적은 허용값이 검사에 안 쓰이는데 쓰이는 줄 알게 된다.",
                        "허용값을 비우거나 종류를 목록으로 바꾸세요.")
        steps.append(step)
    return {
        "kind": "AnswerSchema",
        "id": schema_id.strip(),
        "version": "1.0.0",
        "render": render,
        "locale": "ko",
        "steps": steps,
        "constraints": {"max_tokens": int(max_tokens), "step_order": "strict", "vocabulary": "closed"},
    }


def dump(schema: Dict[str, Any]) -> str:
    return yaml.safe_dump(schema, allow_unicode=True, sort_keys=False, width=100)
