"""Line 의 표 입력 — CSV, 열 지정 검사, 학습/검증 나누기, ③의 두 검사.

표의 한 행이 한 건이다. 열 하나가 어긋난 것을 첫 샘플에서 죽기 전에 펼치는 자리에서 잡고,
정답 노출 검사와 형식 검사가 표로 들어온 데이터에서도 계속 도는지 본다.
"""

from __future__ import annotations

import copy
import os
import shutil

import pytest
import yaml

from vlm_trainer.core import registry
from vlm_trainer.core.compiler import compile_project
from vlm_trainer.core.errors import SpecError
from vlm_trainer.engine import samples as samples_mod
from vlm_trainer.engine.runner import RunOptions, execute
from vlm_trainer.spec.line import expand_data

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOL = os.path.join(ROOT, "solutions")

QUESTION = "아래 사진을 보고 어떤 기계 요소인지 판정하시오."
ROWS = [  # 건, 부품, 방향, 그룹
    ("a1", "기어", "가로", "g1"), ("a2", "나사", "세로", "g1"), ("a3", "너트", "정사각", "g2"),
    ("a4", "체인", "가로", "g2"), ("a5", "풀리", "세로", "g3"), ("a6", "와셔", "가로", "g3"),
    ("a7", "리벳", "세로", "g4"), ("a8", "스프링", "가로", "g4"), ("a9", "베어링", "정사각", "g5"),
    ("a10", "플랜지", "가로", "g5"),
]


@pytest.fixture(autouse=True)
def _nodes():
    registry.load_builtin_nodes()


@pytest.fixture
def tree(tmp_path):
    """스테이션·묶음 상자·스키마를 복사한 작은 과제 하나. 표는 CSV 다."""
    from PIL import Image

    root = tmp_path / "sol"
    for d in ("stations", "procedures"):
        shutil.copytree(os.path.join(SOL, d), root / d)
    (root / "schemas").mkdir()
    shutil.copy(os.path.join(SOL, "vlm_open", "schemas", "answer.yaml"), root / "schemas" / "answer.yaml")
    data = root / "data"
    (data / "img").mkdir(parents=True)
    for r in ROWS:
        Image.new("RGB", (16, 8), (10, 20, 30)).save(data / "img" / f"{r[0]}.png")
    proj = root / "projects" / "p1"
    proj.mkdir(parents=True)
    shutil.copy(os.path.join(SOL, "vlm_open", "projects", "01_open", "trainer.yaml"), proj / "trainer.yaml")

    def write_csv(rows=ROWS, question=QUESTION, header=("sample_id", "image_path", "question", "part_type", "orientation", "group"),
                  split=None, bom=True):
        lines = [",".join(header + (("split",) if split else ()))]
        for i, (sid, pt, ori, g) in enumerate(rows):
            q = question(sid) if callable(question) else question
            row = [sid, f"img/{sid}.png", f'"{q}"', pt, ori, g] + ([split(i)] if split else [])
            lines.append(",".join(row))
        (data / "t.csv").write_bytes((("﻿" if bom else "") + "\r\n".join(lines) + "\r\n").encode("utf-8"))

    write_csv()
    line = {
        "spec_version": 1, "kind": "Line", "id": "p1", "name": "표 시험",
        "sample_space": {"index": "../../data/t.csv", "key": "sample_id", "filter": "",
                         "splits": {"strategy": "random", "ratios": {"train": 0.7, "val": 0.3}, "seed": 7}},
        "stations": [
            {"id": "s1", "ref": "image@1.0.0",
             "params": {"image_column": "image_path", "columns": ["question", "part_type", "orientation"],
                        "size": [32, 32], "keep_aspect": True}},
            {"id": "s2", "ref": "question@1.0.0", "params": {"template": "{question}"}},
            {"id": "s3", "ref": "answer@1.0.0",
             "params": {"schema_path": "../../schemas/answer.yaml",
                        "mapping": {"part_type": "part_type", "orientation": "orientation"},
                        "render": "tagged"}},
            {"id": "s4", "ref": "train@1.0.0", "params": {"config_path": "trainer.yaml"}},
            {"id": "s5", "ref": "verify@1.0.0", "params": {}},
        ],
    }

    class T:
        pass

    t = T()
    t.proj, t.write_csv, t.line = str(proj), write_csv, line
    t.path = lambda: _save(proj, t.line)
    return t


def _save(proj, line):
    p = proj / "line.yaml"
    p.write_text(yaml.safe_dump(line, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(p)


def _mut(t, fn):
    d = copy.deepcopy(t.line)
    fn(d)
    return d


# ── CSV 한 장으로 한 건씩 ────────────────────────────────────────────────
def test_a_csv_with_a_bom_and_quoted_commas_is_read_as_one_row_per_sample(tree):
    tree.write_csv(question=lambda sid: "사진을 보고, 부품을 판정하시오.")
    cg = compile_project(tree.path())
    space = samples_mod.load(cg.sample_space, tree.proj)
    assert len(space) == len(ROWS) and space.rows[0]["question"] == "사진을 보고, 부품을 판정하시오."
    assert "sample_id" in space.rows[0], "BOM 이 첫 열 이름에 붙었다"


def test_csv_values_stay_strings():
    """"3" 을 숫자로 짐작해 바꾸는 것도 자동 캐스팅이다."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.csv")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("sample_id,n\r\nk1,003\r\n")
        assert samples_mod.read_table(p) == [{"sample_id": "k1", "n": "003"}]


# ── 열 지정이 어긋나면 첫 샘플 전에 말한다 ───────────────────────────────
def test_a_wrong_image_column_stops_at_expand_and_lists_the_real_columns(tree):
    tree.line["stations"][0]["params"]["image_column"] = "photo"
    with pytest.raises(SpecError) as e:
        compile_project(tree.path())
    assert "'photo'" in str(e.value) and "image_path" in str(e.value) and "안 잡혔다면" in str(e.value)


def test_an_answer_column_missing_from_the_table_is_refused(tree):
    tree.line["stations"][2]["params"]["mapping"]["orientation"] = "direction"
    with pytest.raises(SpecError, match="direction"):
        compile_project(tree.path())


def test_an_answer_column_the_metadata_list_forgot_is_refused(tree):
    tree.line["stations"][0]["params"]["columns"] = ["question", "part_type"]
    with pytest.raises(SpecError, match="orientation"):
        compile_project(tree.path())


def test_a_question_field_the_metadata_list_forgot_is_refused(tree):
    tree.line["stations"][1]["params"]["template"] = "{question} {note}"
    with pytest.raises(SpecError, match="note"):
        compile_project(tree.path())


# ── 학습/검증 나누기 ─────────────────────────────────────────────────────
def test_a_line_must_say_how_to_split(tree):
    tree.line["sample_space"]["splits"] = {}
    with pytest.raises(SpecError, match="나누는 방법"):
        compile_project(tree.path())


def test_automatic_split_gives_both_sides_and_is_stable_when_rows_are_added(tree):
    cg = compile_project(tree.path())
    a = samples_mod.load(cg.sample_space, tree.proj)
    sp = {r["sample_id"]: r["_split"] for r in a.rows}
    assert {"train", "val"} <= set(sp.values())

    more = ROWS + [("b1", "기어", "가로", "g9"), ("b2", "나사", "세로", "g9")]
    from PIL import Image
    for r in more[len(ROWS):]:
        Image.new("RGB", (16, 8)).save(os.path.join(tree.proj, "..", "..", "data", "img", f"{r[0]}.png"))
    tree.write_csv(rows=more)
    b = samples_mod.load(compile_project(tree.path()).sample_space, tree.proj)
    assert {r["sample_id"]: r["_split"] for r in b.rows if r["sample_id"] in sp} == sp, \
        "행을 더하자 기존 행의 학습/검증이 바뀌었다"


def test_group_split_keeps_a_group_on_one_side(tree):
    tree.line["sample_space"]["splits"]["strategy"] = "group_holdout"
    tree.line["sample_space"]["splits"]["group_by"] = "group"
    cg = compile_project(tree.path())
    rows = samples_mod.load(cg.sample_space, tree.proj).rows
    side = {}
    for r in rows:
        assert side.setdefault(r["group"], r["_split"]) == r["_split"], f"그룹 {r['group']}이 양쪽에 있다"


def test_a_missing_group_column_is_refused_not_silently_split_per_row(tree):
    tree.line["sample_space"]["splits"]["strategy"] = "group_holdout"
    tree.line["sample_space"]["splits"]["group_by"] = "lot"
    with pytest.raises(SpecError, match="lot"):
        compile_project(tree.path())


def test_an_empty_validation_side_is_refused(tree):
    tree.line["sample_space"]["splits"]["ratios"] = {"train": 0.999, "val": 0.001}
    with pytest.raises(SpecError, match="0이다"):
        compile_project(tree.path())


def test_the_column_mode_uses_the_tables_own_column_and_refuses_blanks(tree):
    tree.write_csv(split=lambda i: "val" if i % 5 == 0 else "train")
    tree.line["sample_space"]["splits"] = {"strategy": "column", "column": "split"}
    cg = compile_project(tree.path())
    rows = samples_mod.load(cg.sample_space, tree.proj).rows
    assert sorted({r["_split"] for r in rows}) == ["train", "val"]

    tree.write_csv(split=lambda i: "" if i == 3 else "train")
    with pytest.raises(SpecError, match="비었거나 없다"):
        compile_project(tree.path())


def test_the_column_mode_with_no_such_column_is_refused(tree):
    tree.line["sample_space"]["splits"] = {"strategy": "column", "column": "split"}
    with pytest.raises(SpecError, match="split"):
        compile_project(tree.path())


# ── ③ 의 두 검사는 표로 들어온 데이터에도 돈다 ───────────────────────────
def _run(tree):
    cg = compile_project(tree.path())
    space = samples_mod.load(cg.sample_space, tree.proj)
    opts = RunOptions(run_id="t", cache_dir=os.path.join(tree.proj, "cache"), spec_dir=tree.proj,
                      quarantine_ratio_threshold=1.0, use_cache=False)
    return execute(cg, space, space.rows, opts)


def test_a_question_that_contains_the_answer_is_ejected_at_the_answer_station(tree):
    tree.write_csv(question=lambda sid: "이 기어 사진을 보고 판정하시오." if sid == "a3" else QUESTION)
    rep = _run(tree)
    bad = [q for q in rep.quarantine if q.node_id == "n_guard"]
    assert [q.sample_key for q in bad] == ["a3"], [(q.sample_key, q.node_id) for q in rep.quarantine]
    assert rep.processed == len(ROWS) - 1


def test_an_answer_outside_the_allowed_values_is_ejected_by_the_format_check(tree):
    rows = [(s, "톱니바퀴" if s == "a4" else p, o, g) for s, p, o, g in ROWS]  # 허용값에 없다
    tree.write_csv(rows=rows)
    rep = _run(tree)
    assert [q.sample_key for q in rep.quarantine] == ["a4"]
    assert rep.quarantine[0].node_id == "p_answer/n_check"


def test_clean_rows_all_pass(tree):
    rep = _run(tree)
    assert not rep.quarantine and rep.processed == len(ROWS)
