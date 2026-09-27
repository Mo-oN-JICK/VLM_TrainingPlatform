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


# ── Procedure 가 둘 이상일 때 ───────────────────────────────────────────
#
# 위 테스트들이 쓰는 그래프에는 Procedure 가 **하나**뿐이라, Procedure 에서 Procedure 로
# 가는 배선이 있을 수 없었다. 그래서 그 배선이 저장할 때마다 사라지는 것을 한 번도
# 잡지 못했다 — 편집기에서 값 하나 고치고 저장하면 그래프가 컴파일되지 않았다.

REAL = os.path.join(os.path.dirname(HERE), "solutions", "vlm_open",
                    "projects", "01_open", "project.yaml")


@pytest.fixture
def nodes():
    from vlm_trainer.core import registry

    registry.load_builtin_nodes()


def _roundtrip(spec_path, tmp_path, *, keep: bool = True):
    """decompile 한 것을 **원본과 같은 깊이**에 두고 다시 컴파일한다.
    Procedure 파일이 solution 위쪽에 있어 경로가 얕아지면 찾지 못한다."""
    import shutil

    root = os.path.dirname(HERE)          # 저장소 루트
    work = tmp_path / "w"
    shutil.copytree(os.path.join(root, "solutions"), work / "solutions")
    dst = str(work / os.path.relpath(spec_path, root))

    cg = compile_project(spec_path)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(dump_yaml(decompile(cg, keep_procedures=keep)))
    return cg, compile_project(dst)


def test_an_edge_between_two_procedures_survives_a_save(nodes, tmp_path):
    """`p_prep:images -> p_prompt:images` 는 **프로젝트가 그은 선**이다. 양쪽 끝이
    Procedure 안에 있다는 이유로 버리면 어디에도 다시 적히지 않는다."""
    cg, again = _roundtrip(REAL, tmp_path)
    assert len(again.edges) == len(cg.edges), "배선이 사라졌다"
    assert canonical_view(again) == canonical_view(cg)
    assert again.spec_hash == cg.spec_hash


def test_the_wiring_inside_one_procedure_is_still_left_to_the_procedure_file(nodes, tmp_path):
    """규칙을 푼 것이지 버린 것이 아니다. 같은 Procedure 안의 배선은 프로젝트가
    다시 적을 것이 아니다 — 적으면 Procedure 를 고쳐도 프로젝트가 옛 배선을 붙든다."""
    cg = compile_project(REAL)
    spec = decompile(cg, keep_procedures=True)
    inner = [e for e in spec["edges"]
             if e["from"].split(":")[0].split("/")[0] == e["to"].split(":")[0].split("/")[0]
             and "/" in e["from"]]
    assert not inner, f"Procedure 내부 배선이 프로젝트에 적혔다: {inner}"


def test_a_graph_with_several_procedures_survives_being_flattened(nodes, tmp_path):
    """Procedure 를 펼쳐 저장하는 길도 같은 배선을 들고 있어야 한다."""
    cg, again = _roundtrip(REAL, tmp_path, keep=False)
    assert again.spec_hash == cg.spec_hash


def test_saving_from_the_editor_leaves_a_graph_that_still_compiles(nodes, tmp_path):
    """이것이 실제로 겪은 길이다 — 앱에서 값 하나 고치고 저장하자 다음 실행이
    "필수 입력 포트 'images' 가 연결되지 않았다" 로 멈췄다."""
    import shutil

    from vlm_trainer.ui.api import Editor

    root = os.path.dirname(HERE)
    work = tmp_path / "w"
    shutil.copytree(os.path.join(root, "solutions"), work / "solutions")
    spec = str(work / "solutions" / "vlm_open" / "projects" / "01_open" / "project.yaml")

    ed = Editor.open(spec)
    assert ed.set_param("n_infer", "max_new_tokens", 32).get("ok")
    ed.save()

    cg = compile_project(spec)          # CLI 가 읽는 길
    assert cg.nodes["n_infer"].params["max_new_tokens"] == 32
    assert len(cg.edges) == 23
