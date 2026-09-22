"""Phase 1 완료 조건 — Procedure 인라인과 decompile 왕복."""

from __future__ import annotations

import os

import pytest
import yaml

from vlm_trainer.core.compiler import canonical_view, compile_project
from vlm_trainer.spec.canonical import hash_obj
from vlm_trainer.spec.decompile import decompile, dump_yaml
from vlm_trainer.spec.loader import load_project

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(HERE, "data", "solution", "projects", "01_demo", "project.yaml")


def test_procedure_is_inlined_with_namespaced_ids():
    cg = compile_project(PROJECT)
    assert "p1/n_exp" in cg.nodes and "p1/n_crop" in cg.nodes
    assert cg.nodes["p1/n_exp"].origin == "p1"
    # 노출 파라미터가 내부 노드에 적용된다
    assert cg.nodes["p1/n_exp"].params["topk"] == 2
    assert cg.nodes["p1/n_crop"].params["padding_ratio"] == 0.2
    # 노출되지 않은 내부 파라미터는 기본값 그대로 (캡슐화)
    assert cg.nodes["p1/n_crop"].params["max_n"] == 3


def test_external_edges_are_rewired_into_the_procedure():
    cg = compile_project(PROJECT)
    assert cg.nodes["p1/n_exp"].inputs["subject"] == "n_img:image"
    assert cg.nodes["p1/n_crop"].inputs["image"] == "n_img:image"
    assert cg.nodes["n_out"].inputs["images"] == "p1/n_crop:crops"


def test_roundtrip_preserves_meaning_with_procedures_refolded(tmp_path):
    cg = compile_project(PROJECT)
    spec = decompile(cg, keep_procedures=True)
    assert [p["id"] for p in spec["procedures"]] == ["p1"]
    assert not any(n["id"].startswith("p1/") for n in spec["nodes"])

    out = tmp_path / "restored.yaml"
    out.write_text(dump_yaml(spec), encoding="utf-8")
    # Procedure 파일을 찾을 수 있도록 원본과 같은 깊이에 둔다
    dest = os.path.join(os.path.dirname(PROJECT), "_restored.yaml")
    try:
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(dump_yaml(spec))
        again = compile_project(dest)
        assert canonical_view(again) == canonical_view(cg)
        assert again.spec_hash == cg.spec_hash
    finally:
        if os.path.exists(dest):
            os.remove(dest)


def test_roundtrip_flattened_is_also_stable(tmp_path):
    cg = compile_project(PROJECT)
    spec = decompile(cg, keep_procedures=False)
    assert "procedures" not in spec
    assert any(n["id"] == "p1/n_crop" for n in spec["nodes"])

    dest = tmp_path / "flat.yaml"
    dest.write_text(dump_yaml(spec), encoding="utf-8")
    again = compile_project(str(dest))
    assert again.spec_hash == cg.spec_hash


def test_canonical_records_every_default_explicitly():
    cg = compile_project(PROJECT)
    view = canonical_view(cg)
    crop = next(n for n in view["nodes"] if n["id"] == "p1/n_crop")
    assert set(crop["params"]) == {"padding_ratio", "max_n"}  # 생략 없음
    assert hash_obj(view) == cg.spec_hash


def test_unexposed_procedure_param_is_rejected():
    src = load_project(PROJECT)
    src.procedures[0].params["max_n"] = 9
    from vlm_trainer.core.compiler import compile_graph

    with pytest.raises(Exception, match="노출되지 않은 파라미터"):
        compile_graph(src)


def test_pinned_procedure_version_is_not_auto_upgraded():
    src = load_project(PROJECT)
    src.procedures[0].ref = "expert_crop@1.1.0"
    from vlm_trainer.core.compiler import compile_graph

    with pytest.raises(Exception, match="핀 고정"):
        compile_graph(src)
