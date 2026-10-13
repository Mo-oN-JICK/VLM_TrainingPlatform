"""검증 결과: 항목별 O/X 와 사람 판정 기록."""

import json
import os

import pytest

from vlm_trainer.answer import judge
from vlm_trainer.core.errors import SpecError


@pytest.fixture
def run(tmp_path):
    d = tmp_path / "r1"
    (d / "train").mkdir(parents=True)
    (d / "infer").mkdir()
    (d / "train" / "inference_contract.json").write_text(json.dumps({"answer_schema": {"steps": ["a", "b"]}}))
    rows = [
        {"id": "s1", "split": "val", "answer": "<a>x</a><b>y</b>", "expected": "<a>x</a><b>y</b>"},
        {"id": "s2", "split": "val", "answer": "<a>x</a>", "expected": "<a>x</a><b>y</b>"},
        {"id": "s3", "split": "val", "answer": "", "expected": "<a>x</a><b>y</b>"},
        {"id": "s1", "split": "val", "answer": "<a>z</a><b>y</b>", "expected": "<a>x</a><b>y</b>"},  # 다시 돌린 답
    ]
    (d / "infer" / "answers.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(d)


def test_ox_empty_broken_and_last_answer_wins(run):
    items = {i["id"]: i for i in judge.review(run)["items"]}
    assert [f["match"] for f in items["s1"]["fields"]] == [False, True]  # 마지막 답이 이긴다
    assert items["s2"]["broken"] == ["b"] and not items["s2"]["empty"]
    assert items["s3"]["empty"] and not any(f["match"] for f in items["s3"]["fields"])


def test_judgments_last_line_wins_and_clear(run):
    judge.judge(run, "s1", "ok")
    judge.judge(run, "s1", "ng")
    judge.judge(run, "s2", "ok")
    assert judge.review(run)["count"] == {"ok": 1, "ng": 1, "none": 1}
    judge.judge(run, "s2", "none")
    assert judge.review(run)["count"] == {"ok": 0, "ng": 1, "none": 2}


def test_unknown_sample_and_verdict_refused(run):
    with pytest.raises(SpecError, match="답에 없습니다"):
        judge.judge(run, "nope", "ok")
    with pytest.raises(SpecError, match="없는 값"):
        judge.judge(run, "s1", "maybe")
    assert not os.path.exists(os.path.join(run, "infer", "judgments.jsonl"))
