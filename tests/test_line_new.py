"""새 5노드 프로젝트 — 폴더 만들기부터 데이터 적용·저장까지, YAML 을 손으로 쓰지 않는다."""

import csv
import os

import pytest

from vlm_trainer.core import registry
from vlm_trainer.core.errors import SpecError
from vlm_trainer.spec import scaffold
from vlm_trainer.spec.line import expand_file
from vlm_trainer.ui import line_data as LD
from vlm_trainer.ui.api import Editor


@pytest.fixture(autouse=True)
def _nodes():
    registry.load_builtin_nodes()


def test_a_new_line_opens_without_a_table_and_runs_only_after_data_is_applied(tmp_path):
    path, made = scaffold.new_line(str(tmp_path / "검사1"), name="검사1")
    assert any(m.endswith("stations" + os.sep) for m in made)  # 위쪽에서 안 찾아져 복사했다

    ed = Editor.open_line(path)
    assert ed.compiled is None and "table.csv" in ed.error and not ed.valid

    d = os.path.dirname(path)
    os.makedirs(os.path.join(d, "data"))
    with open(os.path.join(d, "data", "table.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "photo", "q", "kind"])
        for i in range(10):
            w.writerow([f"p{i}", f"img{i}.png", "무엇입니까?", "가" if i % 2 else "나"])
    f = LD.form(ed.line_data, d)
    f.update({"index": "data/table.csv", "key": "id", "image_column": "photo", "question_column": "q",
              "answers": {"kind": "kind"}, "splits": LD.make_splits("random", val_pct=20)})
    schema = {"rows": [{"항목": "kind", "이름": "종류", "종류": "목록", "허용값": "가|나"}], "id": "t1"}
    res = ed.line_apply_data(f, schema)
    assert res["ok"], res
    assert ed.compiled is not None and ed.line_save()["ok"]
    assert expand_file(path)["kind"] == "Project"


def test_an_existing_line_is_not_overwritten(tmp_path):
    scaffold.new_line(str(tmp_path / "a"))
    with pytest.raises(SpecError, match="덮지 않습니다"):
        scaffold.new_line(str(tmp_path / "a"))


def test_stations_found_above_are_not_copied(tmp_path):
    import shutil
    sol = tmp_path / "sol"
    shutil.copytree(os.path.join(scaffold._SAMPLE, "stations"), sol / "stations")
    shutil.copytree(os.path.join(scaffold._SAMPLE, "procedures"), sol / "procedures")
    _, made = scaffold.new_line(str(sol / "projects" / "p1"))
    assert [os.path.basename(m.rstrip(os.sep)) for m in made] == ["line.yaml", "trainer.yaml"]
