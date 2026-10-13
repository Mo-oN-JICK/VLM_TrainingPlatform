"""5노드(Line) 편집 — 설정을 고치고 저장하고 내보낸다.

원본은 Line 이다. 편집은 Line 의 매핑을 고친 뒤 다시 펼쳐 컴파일하고, 통과해야 받아들인다.
저장 후 다시 열어도 같은 그래프여야 한다(저장이 배선을 지운 사고가 이 왕복에서 났다).
"""

from __future__ import annotations

import os
import shutil

import pytest
import yaml

from vlm_trainer.core import registry
from vlm_trainer.core.compiler import compile_project
from vlm_trainer.core.errors import SpecError
from vlm_trainer.spec.line import expand_data
from vlm_trainer.train.config import TrainerConfig
from vlm_trainer.ui.api import Editor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


@pytest.fixture(autouse=True)
def _nodes():
    registry.load_builtin_nodes()


@pytest.fixture
def work(tmp_path):
    """solutions 전체를 복사한다(묶음 상자·스테이션이 위쪽에 있다). 사진은 안 쓰므로 뺀다."""
    dst = tmp_path / "solutions"
    shutil.copytree(os.path.join(ROOT, "solutions"), dst,
                    ignore=shutil.ignore_patterns("images", "runs", "edit_history*", "layout.yaml", "series"))
    proj = dst / "vlm_open" / "projects" / "01_open"
    return str(proj / "line.yaml"), str(proj / "project.yaml")


def _station(ed, name):
    return next(s for s in ed.line_describe()["stations"] if s["name"] == name)


# ── 설정 하나를 고치면 ──────────────────────────────────────────────────
def test_editing_a_setting_changes_the_expanded_graph_and_marks_it_unsaved(work):
    line, project = work
    ed = Editor.open_line(line)
    before = ed.compiled.spec_hash
    r = ed.line_set("s1", "size", [224, 224])
    assert r["ok"], r
    assert ed.dirty and ed.compiled.spec_hash != before
    assert ed.compiled.nodes["p_prep/n_resize"].params["size"] == [224, 224]
    assert _station(ed, "image")["settings"]["size"]["value"] == [224, 224]


def test_a_rejected_edit_leaves_the_old_value_and_says_why(work):
    line, _ = work
    ed = Editor.open_line(line)
    before = ed.compiled.spec_hash
    for station, name, value, why in [
        ("s1", "columns", ["question", "part_type"], "orientation"),     # 정답 열이 목록에서 빠진다
        ("s2", "template", "{question} {note}", "note"),                  # 없는 자리
        ("s1", "image_column", "photo", "photo"),                         # 표에 없는 열
        ("s1", "nope", 1, "노출하지 않은"),                                # 노출 안 한 설정
    ]:
        r = ed.line_set(station, name, value)
        assert not r["ok"] and why in r["detail"], (name, r)
    assert ed.compiled.spec_hash == before and not ed.dirty


# ── 학습량 ──────────────────────────────────────────────────────────────
def test_amount_scales_the_file_epochs_and_normal_changes_nothing(work):
    line, project = work
    ed = Editor.open_line(line)
    base = ed.compiled.spec_hash
    assert compile_project(project).spec_hash == base
    cfg = TrainerConfig.load(os.path.join(os.path.dirname(line), "trainer.yaml"))
    file_epochs = {s.name: s.epochs for s in cfg.stages}

    assert ed.line_set_amount("s4", "많게")["ok"]
    assert ed.compiled.nodes["n_train"].params["epochs"] == {k: v * 2 for k, v in file_epochs.items()}
    assert ed.line_set_amount("s4", "적게")["ok"]
    assert ed.compiled.nodes["n_train"].params["epochs"] == {k: max(1, int(v * 0.5 + 0.5)) for k, v in file_epochs.items()}
    assert ed.line_set_amount("s4", "보통")["ok"]
    assert ed.compiled.spec_hash == base, "보통은 파일 값 그대로여야 한다"
    assert not ed.line_set_amount("s4", "엄청")["ok"]


def test_a_direct_epochs_entry_replaces_the_level_and_they_never_coexist(work):
    line, _ = work
    ed = Editor.open_line(line)
    assert ed.line_set_amount("s4", "많게")["ok"]
    assert ed.line_set("s4", "epochs", {"projector_align": 1, "lora_ft": 7})["ok"]
    p = ed.line_data["stations"][3]["params"]
    assert "amount" not in p and p["epochs"]["lora_ft"] == 7
    assert _station(ed, "train")["amount"]["value"] == "", "직접 입력 중인데 단계가 적용 중이라고 나온다"
    assert ed.line_set_amount("s4", "적게")["ok"]
    assert "epochs" not in ed.line_data["stations"][3]["params"]

    bad = dict(ed.line_data)
    bad["stations"] = [dict(s, params=dict(s["params"])) for s in bad["stations"]]
    bad["stations"][3]["params"].update({"amount": "적게", "epochs": {"lora_ft": 3}})
    with pytest.raises(SpecError, match="둘 중 하나만"):
        expand_data(bad, os.path.dirname(line))


def test_the_trainer_config_refuses_unknown_stage_names_and_bad_counts():
    path = os.path.join(ROOT, "solutions", "vlm_open", "projects", "01_open", "trainer.yaml")
    ok = TrainerConfig.load(path, epochs={"lora_ft": 9})
    assert [s.epochs for s in ok.stages] == [4, 9]
    with pytest.raises(ValueError, match="없습니다"):
        TrainerConfig.load(path, epochs={"lora": 3})
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="정수"):
            TrainerConfig.load(path, epochs={"lora_ft": bad})


# ── 안쪽 값 ─────────────────────────────────────────────────────────────
def test_an_inner_value_the_station_exposes_goes_to_the_station_not_to_overrides(work):
    line, _ = work
    ed = Editor.open_line(line)
    assert ed.line_set_override("n_infer", "max_new_tokens", 64)["ok"]
    assert "overrides" not in ed.line_data and ed.line_data["stations"][4]["params"]["max_new_tokens"] == 64


def test_an_inner_value_nobody_exposes_is_recorded_as_an_override_and_can_be_cleared(work):
    line, _ = work
    ed = Editor.open_line(line)
    base = ed.compiled.spec_hash
    assert ed.line_set_override("n_img", "on_missing", "skip")["ok"]
    assert ed.line_data["overrides"] == [{"node": "n_img", "param": "on_missing", "value": "skip"}]
    assert ed.compiled.nodes["n_img"].params["on_missing"] == "skip"
    assert ed.line_clear_override("n_img", "on_missing")["ok"]
    assert "overrides" not in ed.line_data and ed.compiled.spec_hash == base
    assert not ed.line_set_override("n_nowhere", "x", 1)["ok"]


# ── 저장은 왕복한다 ─────────────────────────────────────────────────────
def test_saving_and_reopening_gives_the_same_graph_and_reverting_gives_the_original(work):
    line, project = work
    original = compile_project(project).spec_hash
    ed = Editor.open_line(line)
    assert ed.line_set("s1", "size", [320, 320])["ok"] and ed.line_set_amount("s4", "많게")["ok"]
    assert ed.line_set_override("n_img", "on_missing", "skip")["ok"]
    hash_edited = ed.compiled.spec_hash
    assert ed.line_save()["ok"] and not ed.dirty

    again = Editor.open_line(line)
    assert again.compiled.spec_hash == hash_edited
    assert compile_project(line).spec_hash == hash_edited
    with open(line, encoding="utf-8") as fh:
        saved = yaml.safe_load(fh)
    assert saved["kind"] == "Line" and [s["id"] for s in saved["stations"]] == ["s1", "s2", "s3", "s4", "s5"]

    assert again.line_set("s1", "size", [448, 448])["ok"] and again.line_set_amount("s4", "보통")["ok"]
    assert again.line_clear_override("n_img", "on_missing")["ok"] and again.line_save()["ok"]
    assert compile_project(line).spec_hash == original, "되돌렸는데 원래 그래프가 아니다"


def test_an_unsaved_edit_blocks_a_run(work):
    line, _ = work
    ed = Editor.open_line(line)
    assert ed.line_set("s1", "size", [224, 224])["ok"]
    r = ed.pipeline_start(False, 0.0)
    assert not r["ok"] and "저장하지 않은" in r["reason"]


# ── 내보내기 ────────────────────────────────────────────────────────────
def test_export_writes_a_project_with_the_same_graph_and_never_overwrites(work):
    line, _ = work
    ed = Editor.open_line(line)
    assert ed.line_set("s1", "size", [224, 224])["ok"]
    r = ed.line_export()
    assert not r["ok"] and "저장하지 않은" in r["reason"], "저장 전에 내보냈다"

    assert ed.line_save()["ok"]
    r = ed.line_export()
    assert r["ok"] and r["path"].endswith("01_open_11nodes.yaml")
    with open(r["path"], encoding="utf-8") as fh:
        assert yaml.safe_load(fh)["kind"] == "Project"
    assert compile_project(r["path"]).spec_hash == compile_project(line).spec_hash

    again = ed.line_export()
    assert not again["ok"] and "덮어쓰면" in again["detail"]
