"""숫자 항목 범위 — 정답 형태 표의 `0~100` 이 스키마·형식 검사·검증 화면까지 간다."""

import json

import pytest

from vlm_trainer.answer import judge, table
from vlm_trainer.answer.schema import AnswerSchema
from vlm_trainer.core.errors import SpecError


def _rows(rng="0~100"):
    return [{"항목": "part", "이름": "부품", "종류": "목록", "허용값": "기어|나사"},
            {"항목": "tilt", "이름": "기울기", "종류": "숫자", "허용값": rng}]


def _schema(tmp_path, rng="0~100"):
    p = tmp_path / "s.yaml"
    p.write_text(table.dump(table.build(_rows(rng), schema_id="t")), encoding="utf-8")
    return AnswerSchema.load(str(p))


def test_the_range_goes_into_the_schema_and_back_to_the_table(tmp_path):
    d = table.build(_rows("0~100"), schema_id="t")
    assert d["steps"][1]["range"] == [0, 100]
    p = tmp_path / "s.yaml"
    p.write_text(table.dump(d), encoding="utf-8")
    assert table.rows_from_schema(str(p))["rows"][1]["허용값"] == "0~100"
    assert table.rows_from_schema(str(p))["unsupported"] == []


def test_a_number_outside_the_range_is_a_format_violation_and_inside_is_not(tmp_path):
    sc = _schema(tmp_path, "-1.5~2")
    ok = sc.render_answer({"part": "기어", "tilt": 2}, {})
    assert sc.validate(ok) == []
    assert sc.validate(sc.render_answer({"part": "기어", "tilt": -1.5}, {})) == []  # 양 끝 포함
    assert sc.validate(sc.render_answer({"part": "기어", "tilt": 2.5}, {})) == ["range:tilt"]


def test_no_range_means_any_number_passes_as_before(tmp_path):
    sc = _schema(tmp_path, "")
    assert sc.validate(sc.render_answer({"part": "기어", "tilt": 99999}, {})) == []
    assert "range" not in table.build(_rows(""), schema_id="t")["steps"][1]


def test_a_schema_without_a_range_keeps_its_digest():
    sc = AnswerSchema(id="x", steps=[])
    from vlm_trainer.answer.schema import Step

    sc.steps = [Step(id="n", type="float")]
    assert "range" not in sc.digest()["steps"][0]
    sc.steps = [Step(id="n", type="float", range=(0.0, 1.0))]
    assert sc.digest()["steps"][0]["range"] == [0.0, 1.0]


@pytest.mark.parametrize("bad", ["100~0", "0~", "~5", "a~b", "1|2", "0~5~9"])
def test_an_unreadable_range_names_the_row_and_the_fix(bad):
    with pytest.raises(SpecError, match="범위") as e:
        table.build(_rows(bad), schema_id="t")
    assert "2번째 행" in str(e.value) and "최소~최대" in str(e.value)


def test_a_range_on_a_non_number_item_is_refused_when_the_file_is_loaded(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("kind: AnswerSchema\nid: x\nsteps:\n  - {id: a, type: enum, values: [x], range: [0, 1]}\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="range"):
        AnswerSchema.load(str(p))


def test_the_verdict_view_marks_a_model_number_outside_the_range(tmp_path):
    d = tmp_path / "r"
    (d / "train").mkdir(parents=True)
    (d / "infer").mkdir()
    (d / "train" / "inference_contract.json").write_text(json.dumps(
        {"answer_schema": {"steps": ["part", "tilt"], "ranges": {"tilt": [0, 100]}}}))
    row = {"id": "s1", "split": "val", "answer": "<part>기어</part><tilt>150</tilt>",
           "expected": "<part>기어</part><tilt>50</tilt>"}
    (d / "infer" / "answers.jsonl").write_text(json.dumps(row), encoding="utf-8")
    f = judge.review(str(d))["items"][0]["fields"]
    assert [x["out_of_range"] for x in f] == [False, True]
    assert [x["match"] for x in f] == [True, False]
