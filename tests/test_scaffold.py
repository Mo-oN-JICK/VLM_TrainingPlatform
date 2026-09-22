"""`vlmt new` — 편집기가 열 파일 하나를 만든다.

편집기는 스펙을 **편집하는** 뷰다. 그래서 시작할 파일이 하나는 있어야 하는데,
그 하나를 손으로 쓰게 두면 새 도메인으로 확장하는 첫 관문이 YAML 받아쓰기가 된다.
"""

from __future__ import annotations

import os

import pytest
import yaml

from vlm_trainer.core.errors import SpecError
from vlm_trainer.spec import scaffold
from vlm_trainer.ui.api import Editor


def test_it_makes_a_solution_and_a_project(tmp_path):
    proj, made = scaffold.new_project(str(tmp_path / "ecg2"), name="심전도 2차")

    assert os.path.basename(proj) == "project.yaml"
    assert len(made) == 2 and all(os.path.exists(f) for f in made)

    sol = yaml.safe_load(open(made[0], encoding="utf-8"))
    assert sol["kind"] == "Solution" and sol["id"] == "ecg2"
    assert sol["runtime_profile"] == "windows_single_gpu"

    d = yaml.safe_load(open(proj, encoding="utf-8"))
    assert d["kind"] == "Project" and d["id"] == "01_ecg2" and d["name"] == "심전도 2차"
    assert d["nodes"] == [] and d["edges"] == []


def test_the_index_path_is_relative_to_the_project_file(tmp_path):
    """스펙의 모든 경로는 그 파일 기준이다. 여기서만 예외를 두면 헷갈린다."""
    proj, _ = scaffold.new_project(str(tmp_path / "s"))
    d = yaml.safe_load(open(proj, encoding="utf-8"))

    assert d["sample_space"]["index"] == "../../data/index.jsonl"
    resolved = os.path.normpath(os.path.join(os.path.dirname(proj), d["sample_space"]["index"]))
    assert resolved == os.path.normpath(str(tmp_path / "s" / "data" / "index.jsonl"))


def test_it_refuses_to_overwrite(tmp_path):
    """껍데기를 만드는 명령이 남의 스펙을 지우면 안 된다."""
    scaffold.new_project(str(tmp_path / "s"))
    with pytest.raises(SpecError, match="덮지 않는다"):
        scaffold.new_project(str(tmp_path / "s"))


def test_odd_names_become_usable_ids(tmp_path):
    proj, _ = scaffold.new_project(str(tmp_path / "s"), solution_id="내 과제 / v2!")
    d = yaml.safe_load(open(proj, encoding="utf-8"))
    assert d["id"] == "01_______v2_" or d["id"].startswith("01_")
    assert "/" not in d["id"] and " " not in d["id"]


def test_the_editor_opens_the_empty_project_and_says_what_is_missing(tmp_path):
    """빈 그래프는 컴파일되지 않는다. 그것이 정상이고, 편집기는 이유를 말해야 한다."""
    proj, _ = scaffold.new_project(str(tmp_path / "s"))
    ed = Editor.open(proj)

    assert ed.valid is False
    assert "Output 노드가 없다" in ed.error
    assert not ed.save()["ok"], "미완성인 껍데기를 저장하지 않는다"


def test_the_cli_exposes_it():
    from vlm_trainer.cli.main import build_parser

    a = build_parser().parse_args(["new", "somewhere", "--name", "X", "--profile", "linux_multi_gpu"])
    assert a.dir == "somewhere" and a.name == "X" and a.profile == "linux_multi_gpu"
    assert a.edit is False
