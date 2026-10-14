"""데이터 대화상자가 쓰는 계산 — Qt 를 모른다.

표의 열 목록을 읽고, 지금 Line 이 담은 값을 대화상자의 칸으로 풀고, 고른 나누기 설정으로
학습/검증이 몇 건이 되는지 미리 센다. 화면은 이 결과를 그리기만 한다.
"""

from __future__ import annotations

import os
import posixpath
import re
from typing import Any, Dict, Optional

from ..answer import table as answer_table
from ..core.errors import SpecError
from ..engine import samples as samples_mod
from ..engine import xlsx

_ONE_FIELD = re.compile(r"^\{([^{}]+)\}$")
DEFAULT_SEED = 20260921


def resolve(spec_dir: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(spec_dir, rel))


def station(data: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """종류 이름(image / question / answer / train / verify)으로 스테이션을 찾는다."""
    for s in data.get("stations") or []:
        if str(s.get("ref", "")).split("@")[0] == name:
            return s
    return None


def read_header(spec_dir: str, index: str) -> Dict[str, Any]:
    """표의 열 목록과 건수. 못 읽으면 SpecError 가 그 이유를 말한다."""
    if not index.strip():
        raise SpecError("표 파일이 정해지지 않았습니다.\n  안 잡혔다면: 읽을 건이 없어 학습이 빈 데이터로 시작됩니다.\n  표 파일을 고르세요.")
    index, sheet = xlsx.split_sheet(index)
    path = resolve(spec_dir, index)
    if not os.path.exists(path):
        raise SpecError(f"표 파일이 없습니다: {path}\n  안 잡혔다면: 실행 첫머리에서 멈춥니다.\n  파일 경로를 확인하세요.")
    rows = samples_mod.read_table(path, sheet)
    sheets = xlsx.sheet_names(path) if path.lower().endswith(".xlsx") else []
    cols = list(rows[0]) if rows else []
    for r in rows[1:]:
        cols = [c for c in cols if c in r]
    return {"path": path, "columns": cols, "count": len(rows), "sheets": sheets, "sheet": sheet or (sheets[0] if sheets else "")}


def split_counts(spec_dir: str, index: str, key: str, splits: Dict[str, Any]) -> Dict[str, int]:
    """고른 나누기 설정으로 학습/검증이 각각 몇 건인가. 스펙의 filter 와 별개로 표 전체를 센다."""
    space = samples_mod.load({"index": index, "key": key, "filter": "", "splits": splits}, spec_dir)
    out: Dict[str, int] = {}
    for r in space.rows:
        out[r["_split"]] = out.get(r["_split"], 0) + 1
    return out


def make_splits(mode: str, *, column: str = "", group: str = "", val_pct: int = 10,
                seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """대화상자의 선택 → `sample_space.splits`. 모드는 셋이고 짐작으로 넘어가는 길이 없다."""
    if mode == "column":
        return {"strategy": "column", "column": column}
    ratios = {"train": round(1 - val_pct / 100.0, 6), "val": round(val_pct / 100.0, 6)}
    if mode == "group":
        return {"strategy": "group_holdout", "group_by": group, "ratios": ratios, "seed": int(seed)}
    if mode == "random":
        return {"strategy": "random", "ratios": ratios, "seed": int(seed)}
    raise SpecError(f"나누는 방법 {mode!r}는 없는 값입니다 (column | random | group).\n"
                    "  안 잡혔다면: 검증이 0건이 될 수 있습니다.\n  셋 중 하나를 고르세요.")


def form(data: Dict[str, Any], spec_dir: str) -> Dict[str, Any]:
    """Line 이 지금 담고 있는 값을 대화상자의 칸으로."""
    ss = data.get("sample_space") or {}
    img = station(data, "image") or {"params": {}}
    qst = station(data, "question") or {"params": {}}
    ans = station(data, "answer") or {"params": {}}
    tpl = str((qst.get("params") or {}).get("template", ""))
    m = _ONE_FIELD.match(tpl)

    schema_path = str((ans.get("params") or {}).get("schema_path", ""))
    schema: Dict[str, Any] = {"rows": [], "unsupported": [], "id": "", "render": "tagged", "max_tokens": 320,
                              "step_order": "strict", "vocabulary": "closed"}
    full = resolve(spec_dir, schema_path) if schema_path else ""
    if full and os.path.exists(full):
        schema = answer_table.rows_from_schema(full)

    sp = dict(ss.get("splits") or {})
    strategy = str(sp.get("strategy", ""))
    ratios = sp.get("ratios") or {}
    total = sum(ratios.values()) or 1.0
    mapping = dict((ans.get("params") or {}).get("mapping") or {})
    return {
        "index": str(ss.get("index", "")), "key": str(ss.get("key", "sample_id")),
        "image_column": str((img.get("params") or {}).get("image_column", "")),
        "question_column": m.group(1) if m else None, "template": tpl,
        "answers": {r["항목"]: str(mapping.get(r["항목"], r["항목"])) for r in schema["rows"]},
        "schema_path": schema_path, "schema": schema,
        "split": {"mode": {"column": "column", "random": "random", "group_holdout": "group"}.get(strategy, "random"),
                  "column": str(sp.get("column", "")), "group": str(sp.get("group_by", "")),
                  "val_pct": int(round(100.0 * float(ratios.get("val", 0.1)) / total)) if ratios else 10,
                  "seed": int(sp.get("seed", DEFAULT_SEED))},
    }


def owned_schema_path(current: str, line_id: str) -> str:
    """이 프로젝트 전용 정답 형태 파일 경로. 지금 경로와 같은 폴더에 `<id>_answer.yaml`."""
    base = f"{line_id or 'line'}_answer.yaml"
    d = posixpath.dirname(current.replace("\\", "/"))
    return posixpath.join(d, base) if d else base


# ── 질문 문장의 사진 자리 ─────────────────────────────────────────────────
IMAGE_TOKEN = "{image}"


def parse_question(text: str) -> tuple:
    """화면의 질문 문장 → (template, policy). `{image}` 는 맨 앞이나 맨 뒤에만 둔다(Q61).

    노드는 사진 자리를 문장 앞(prepend)이나 뒤(append)에만 붙인다. 중간에 적힌 것을 앞이나 뒤로
    옮겨 붙이면 화면과 모델이 받는 문장이 갈라지므로 거부한다.
    """
    t = str(text or "").strip()
    low = t.lower()
    n = low.count(IMAGE_TOKEN)
    if n > 1:
        raise SpecError(f"질문 문장에 {IMAGE_TOKEN} 가 {n}번 있습니다: {t!r}\n"
                        "  안 잡혔다면: 사진은 한 장인데 자리가 여럿이라 어디에 들어가는지 화면과 실제가 갈라집니다.\n"
                        f"  {IMAGE_TOKEN} 를 하나만 남기세요.")
    policy = "prepend"
    if n == 1:
        if low.startswith(IMAGE_TOKEN):
            t = t[len(IMAGE_TOKEN):].strip()
        elif low.endswith(IMAGE_TOKEN):
            t, policy = t[:-len(IMAGE_TOKEN)].strip(), "append"
        else:
            raise SpecError(f"질문 문장 가운데에 {IMAGE_TOKEN} 가 있습니다: {t!r}\n"
                            "  안 잡혔다면: 모델은 사진을 문장 앞이나 뒤에만 받아서, 화면에 보이는 문장과 실제로 학습하는 문장이 달라집니다.\n"
                            f"  {IMAGE_TOKEN} 를 문장 맨 앞이나 맨 뒤로 옮기세요(예: {IMAGE_TOKEN} 공의 색상은?).")
    if not t:
        raise SpecError("질문 문장이 비었습니다.\n  안 잡혔다면: 모델이 무엇을 답해야 하는지 모른 채 학습합니다.\n"
                        f"  질문을 적으세요(예: {IMAGE_TOKEN} 공의 색상은?).")
    return t, policy


def question_text(template: str, policy: str) -> str:
    """(template, policy) → 화면에 보이는 질문 문장."""
    return f"{template} {IMAGE_TOKEN}" if policy == "append" else f"{IMAGE_TOKEN} {template}"


def question_for(template: str, policy: str, row: Dict[str, Any]) -> str:
    """한 건에 실제로 들어갈 질문(표의 열 값을 채운 것). 못 채우면 문장 그대로."""
    try:
        filled = str(template).format(**{k: v for k, v in row.items()})
    except (KeyError, IndexError, ValueError):
        filled = str(template)
    return question_text(filled, policy)


def val_pct(splits: Dict[str, Any]) -> int:
    r = splits.get("ratios") or {}
    total = sum(r.values()) or 1.0
    return int(round(100.0 * float(r.get("val", 0.2)) / total)) if r else 20
