"""두 번째 도메인 — 이 물건이 플랫폼인지 그래프 하나짜리 도구인지 가르는 검사.

`vlm_parts`는 `dummy_ecg`와 **일부러 다른 모양**이다: 시계열도, 전문가 모델도,
지식 문서도 없고, 이미지 한 장과 라벨 하나뿐이다. 코어가 도메인 무관하다는 주장은
두 번째 도메인이 돌기 전에는 주장일 뿐이다.
"""

from __future__ import annotations

import os

import pytest

from vlm_trainer.core.compiler import compile_project
from vlm_trainer.ui.render import fold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOLUTION = os.path.join(ROOT, "solutions", "vlm_parts")
SPEC = os.path.join(SOLUTION, "projects", "01_parts", "project.yaml")
INDEX = os.path.join(SOLUTION, "data", "parts", "index.jsonl")

pytestmark = pytest.mark.skipif(
    not os.path.exists(INDEX),
    reason="부품 데이터가 없다. python tools/make_parts_dataset.py 를 먼저 실행하라.",
)


@pytest.fixture(scope="module")
def cg():
    return compile_project(SPEC)


def test_it_compiles_without_any_time_series_machinery(cg):
    refs = {n.ref.split("@")[0] for n in cg.nodes.values()}
    assert not any(r.startswith("ts.") for r in refs), "시계열 노드가 남아 있다"
    assert "expert.propose" not in refs, "전문가 모델 없이 돌아야 한다"
    assert "prompt.knowledge_inject" not in refs, "지식 문서 없이 돌아야 한다"


def test_the_canvas_shows_nine_boxes(cg):
    """Flow.png 수준의 밀도. 사람이 한눈에 붙들 수 있어야 한다."""
    shown, edges = fold(cg)

    assert len(shown) == 9, sorted(shown)
    assert len(cg.nodes) == 12, "게이트는 여전히 펼쳐진 그래프를 본다"
    assert {k for k, v in shown.items() if v.inner} == {"p_prep", "p_prompt", "p_answer"}


def test_the_boxes_carry_their_settings(cg):
    """우측 설정창에 뜰 것들. 상자를 접고 설정이 사라지면 접은 값어치가 없다."""
    by_id = {p["id"]: p for p in cg.procedures}

    assert set(by_id["p_prep"]["exposed_params"]) == {"size", "mode", "keep_aspect"}
    assert "template" in by_id["p_prompt"]["exposed_params"]
    assert "mapping" in by_id["p_answer"]["exposed_params"]


def test_one_exposed_input_can_feed_two_inner_nodes(cg):
    """스키마는 정답을 쓰는 쪽과 검사하는 쪽 둘 다에 필요하다.
    입력 하나가 안쪽 여러 포트로 갈라지는 것은 팬인이 아니다."""
    p = next(p for p in cg.procedures if p["id"] == "p_answer")
    assert p["exposed_inputs"]["schema"] == ["n_write:schema", "n_check:schema"]

    fed = {e.dst for e in cg.edges if e.src == "n_schema:schema"}
    assert {"p_answer/n_write:schema", "p_answer/n_check:schema"} <= fed


def test_the_image_prep_box_is_swappable():
    """두 갈래가 같은 포트 이름과 타입을 내야 배선 하나로 갈아 끼운다."""
    from vlm_trainer.spec.loader import find_procedure

    simple = find_procedure("image_prep_resize@1.0.0", os.path.join(SOLUTION, "projects", "01"))
    assert list(simple.exposed_outputs) == ["images"]
    assert list(simple.exposed_inputs) == ["image"]


def test_a_plain_schema_needs_no_evidence():
    """근거 문장을 강제하는 것은 한 도메인의 규약이지 모든 정답의 성질이 아니다."""
    from vlm_trainer.answer.schema import AnswerSchema

    s = AnswerSchema.load(os.path.join(SOLUTION, "schemas", "answer.yaml"))
    assert [st.evidence for st in s.steps] == ["none", "none"]

    text = s.render_answer({"verdict": "정상", "defect": "없음"}, {})
    assert s.validate(text) == [], s.validate(text)


def test_the_ecg_schema_still_demands_evidence():
    """기본값을 바꾼 것이지 기능을 없앤 것이 아니다."""
    from vlm_trainer.answer.schema import AnswerSchema

    s = AnswerSchema.load(os.path.join(ROOT, "solutions", "dummy_ecg", "schemas", "answer.yaml"))
    assert all(st.evidence == "required" for st in s.steps)
