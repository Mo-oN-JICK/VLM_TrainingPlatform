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
    with pytest.raises(SpecError, match="0입니다"):
        compile_project(tree.path())


def test_the_column_mode_uses_the_tables_own_column_and_refuses_blanks(tree):
    tree.write_csv(split=lambda i: "val" if i % 5 == 0 else "train")
    tree.line["sample_space"]["splits"] = {"strategy": "column", "column": "split"}
    cg = compile_project(tree.path())
    rows = samples_mod.load(cg.sample_space, tree.proj).rows
    assert sorted({r["_split"] for r in rows}) == ["train", "val"]

    tree.write_csv(split=lambda i: "" if i == 3 else "train")
    with pytest.raises(SpecError, match="비었거나 없습니다"):
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


# ── 표 양식과 정답 형태 표 ──────────────────────────────────────────────
def _rows():
    return [
        {"항목": "part_type", "이름": "부품", "종류": "목록", "허용값": "기어|나사|너트"},
        {"항목": "orientation", "이름": "화면", "종류": "목록", "허용값": "가로|세로"},
        {"항목": "count", "이름": "개수", "종류": "숫자", "허용값": ""},
        {"항목": "memo", "이름": "메모", "종류": "문장", "허용값": ""},
    ]


def test_the_answer_table_builds_the_same_shape_as_a_hand_written_schema(tmp_path):
    from vlm_trainer.answer import table
    from vlm_trainer.answer.schema import AnswerSchema

    d = table.build(_rows(), schema_id="t1")
    p = tmp_path / "s.yaml"
    p.write_text(table.dump(d), encoding="utf-8")
    sc = AnswerSchema.load(str(p))
    assert [(s.id, s.type, s.values) for s in sc.steps] == [
        ("part_type", "enum", ("기어", "나사", "너트")), ("orientation", "enum", ("가로", "세로")),
        ("count", "float", ()), ("memo", "text", ())]
    ok = sc.render_answer({"part_type": "기어", "orientation": "가로", "count": 3, "memo": "양호"}, {})
    assert sc.validate(ok) == []
    assert sc.vocabulary_terms() == ["기어", "나사", "너트", "가로", "세로"]


def test_a_number_item_rejects_a_word_and_a_list_item_rejects_an_unlisted_value(tmp_path):
    from vlm_trainer.answer import table
    from vlm_trainer.answer.schema import AnswerSchema

    p = tmp_path / "s.yaml"
    p.write_text(table.dump(table.build(_rows(), schema_id="t1")), encoding="utf-8")
    sc = AnswerSchema.load(str(p))
    bad = sc.render_answer({"part_type": "톱니", "orientation": "가로", "count": "많음", "memo": ""}, {})
    got = sc.validate(bad)
    assert "enum_domain:part_type" in got and "number:count" in got


@pytest.mark.parametrize("edit,match", [
    (lambda r: r[0].update({"허용값": ""}), "허용값이 없습니다"),
    (lambda r: r[0].update({"허용값": "기어|기어"}), "겹칩니다"),
    (lambda r: r[2].update({"허용값": "1|2"}), "허용값"),
    (lambda r: r[1].update({"항목": "part_type"}), "앞에서도"),
    (lambda r: r[1].update({"항목": "1번 항목"}), "태그 이름"),
    (lambda r: r[1].update({"종류": "정수"}), "없는 값"),
])
def test_a_bad_answer_row_names_the_row_and_the_fix(edit, match):
    from vlm_trainer.answer import table

    rows = _rows()
    edit(rows)
    with pytest.raises(SpecError, match=match) as e:
        table.build(rows, schema_id="t1")
    assert "번째 행" in str(e.value) and "안 잡혔다면" in str(e.value)


def test_an_empty_answer_table_is_refused():
    from vlm_trainer.answer import table

    with pytest.raises(SpecError, match="하나도 없습니다"):
        table.build([], schema_id="t1")


def test_the_cli_builds_a_schema_from_a_csv_and_a_blank_table_template(tmp_path):
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
    csv_in = tmp_path / "rows.csv"
    csv_in.write_bytes("﻿항목,이름,종류,허용값\r\npart_type,부품,목록,기어|나사\r\n".encode("utf-8"))
    out = tmp_path / "s.yaml"
    r = subprocess.run([sys.executable, "-m", "vlm_trainer.cli.main", "schema-build", str(csv_in),
                        "--id", "x", "--out", str(out)], env=env, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["steps"][0]["values"] == ["기어", "나사"]

    t = tmp_path / "t.csv"
    r = subprocess.run([sys.executable, "-m", "vlm_trainer.cli.main", "table-template", "--out", str(t),
                        "--answers", "part_type,orientation", "--group", "lot"],
                       env=env, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    raw = t.read_bytes()
    assert raw.startswith("﻿".encode("utf-8")) and raw.endswith(b"\r\n")
    assert raw.decode("utf-8-sig").strip() == "sample_id,image_path,question,part_type,orientation,lot"
    # 양식은 표로 읽힌다(빈 표) — 열 이름이 그대로 들어온다
    assert samples_mod.read_table(str(t)) == []


def test_a_template_with_a_repeated_column_is_refused():
    from vlm_trainer.spec import table

    with pytest.raises(SpecError, match="열 이름"):
        table.header(key="id", image="id", question="q", answers=["a"])


def test_the_same_table_as_csv_and_as_jsonl_gives_the_same_rows_and_splits(tmp_path):
    """실제 예제 표(JSONL)를 CSV 로 옮겨도 행과 학습/검증 배정이 같다 — 형식만 바뀐다."""
    import csv
    import json

    src = os.path.join(SOL, "vlm_open", "data", "open_parts", "index.jsonl")
    if not os.path.exists(src):
        pytest.skip("vlm_open 데이터가 없다")
    rows = [json.loads(x) for x in open(src, encoding="utf-8") if x.strip()]
    out = tmp_path / "index.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    spec = {"index": str(out), "key": "sample_id", "filter": "",
            "splits": {"strategy": "column", "column": "split_hint"}}
    a = samples_mod.load(spec, str(tmp_path)).rows
    b = samples_mod.load(dict(spec, index=src), str(tmp_path)).rows
    assert a == b and {r["_split"] for r in a} == {"train", "val"}
