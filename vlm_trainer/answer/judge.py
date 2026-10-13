"""검증 결과 읽기 — 모델 답과 정답을 항목별로 나란히 놓고, 사람의 OK/NG 판정을 남긴다.

O/X 는 **보조 표시**다. 점수로 합치지 않는다: 정답 라벨에 잡음이 있으면 합계는 믿을 값이 아니다.
최종 판정은 사람이 건마다 OK/NG 로 내리고, `judgments.jsonl` 에 한 줄씩 덧붙여 남긴다
(같은 건은 마지막 줄이 이긴다). 화면과 CLI(`vlmt judge`)가 같은 이 함수를 쓴다.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

from ..core.errors import SpecError

VERDICTS = ("ok", "ng", "none")  # none = 판정을 지워 미판정으로 되돌린다


def _infer_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "infer")


def _steps(run_dir: str) -> List[str]:
    """실행이 학습 때 쓴 정답 항목 이름들. 추론 계약에 적혀 있다."""
    path = os.path.join(run_dir, "train", "inference_contract.json")
    if not os.path.exists(path):
        raise SpecError(f"{path} 가 없습니다 — 이 실행은 학습까지 끝나지 않았습니다.\n"
                        "  안 잡혔다면: 정답 항목을 몰라 모델 답을 항목별로 나눌 수 없습니다.\n"
                        "  학습이 끝난 실행 폴더를 고르세요.")
    with open(path, "r", encoding="utf-8") as fh:
        return [str(s) for s in (json.load(fh).get("answer_schema") or {}).get("steps") or []]


def _tag(text: str, step: str):
    m = re.search(rf"<{re.escape(step)}>(.*?)</{re.escape(step)}>", text or "", re.S)
    return m.group(1).strip() if m else None


def read_judgments(run_dir: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    path = os.path.join(_infer_dir(run_dir), "judgments.jsonl")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[str(r["id"])] = r
    return out


def judge(run_dir: str, sample_id: str, verdict: str) -> Dict[str, str]:
    """판정 한 줄을 덧붙인다. 답 파일에 없는 건은 거부한다 — 오타가 조용히 기록되면 안 된다."""
    if verdict not in VERDICTS:
        raise SpecError(f"판정 {verdict!r}는 없는 값입니다 ({' | '.join(VERDICTS)}).\n"
                        "  안 잡혔다면: 엉뚱한 글자가 판정으로 남아 요약이 어긋납니다.\n  ok, ng, none 중 하나를 쓰세요.")
    if str(sample_id) not in {r["id"] for r in read_answers(run_dir)}:
        raise SpecError(f"건 {sample_id!r}는 이 실행의 검증 답에 없습니다.\n"
                        "  안 잡혔다면: 없는 건에 판정이 쌓여 어디에도 안 보입니다.\n"
                        "  `vlmt judgments <실행>` 으로 건 id 를 확인하세요.")
    rec = {"id": str(sample_id), "verdict": verdict, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(_infer_dir(run_dir), "judgments.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def read_answers(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(_infer_dir(run_dir), "answers.jsonl")
    if not os.path.exists(path):
        raise SpecError(f"{path} 가 없습니다 — 이 실행은 검증을 하지 않았습니다.\n"
                        "  안 잡혔다면: 빈 목록이 '모두 통과'처럼 보입니다.\n  검증까지 끝난 실행을 고르세요.")
    with open(path, "r", encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    # 같은 실행 id 로 다시 돌리면 답이 이어 붙는다. 건마다 마지막 답이 지금 모델의 답이다
    return list({str(r["id"]): r for r in rows}.values())


def review(run_dir: str) -> Dict[str, Any]:
    """건마다 항목별 (모델 답, 정답, O/X)와 사람 판정. 빈 답은 전부 X 이고 `empty` 로 따로 표시한다."""
    steps = _steps(run_dir)
    judged = read_judgments(run_dir)
    items, count = [], {"ok": 0, "ng": 0, "none": 0}
    for r in read_answers(run_dir):
        ans, exp = str(r.get("answer") or ""), str(r.get("expected") or "")
        fields = []
        for s in steps:
            a, e = _tag(ans, s), _tag(exp, s)
            fields.append({"step": s, "answer": a, "expected": e, "match": a is not None and a == e})
        v = (judged.get(str(r["id"])) or {}).get("verdict", "none")
        count[v if v in count else "none"] += 1
        items.append({"id": str(r["id"]), "split": r.get("split", ""), "empty": not ans.strip(),
                      "broken": [f["step"] for f in fields if f["answer"] is None] if ans.strip() else [],
                      "fields": fields, "verdict": v, "raw": ans, "expected_raw": exp})
    return {"steps": steps, "items": items, "count": count}
