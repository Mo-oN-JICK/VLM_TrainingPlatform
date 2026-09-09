"""Phase 1 완료 조건 — G1/G2가 각각 무엇을 잡는가."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from vlm_trainer.core.compiler import CompileFailed, compile_graph
from vlm_trainer.core.graph import Edge, GraphModel, Materialize, NodeInstance
from vlm_trainer.core.node import NodeKind


def build(nodes: List[tuple], edges: List[tuple], **kw: Any) -> GraphModel:
    g = GraphModel(id="t", name="test")
    g.nodes = [NodeInstance(i, t, dict(p or {})) for i, t, *rest in nodes for p in [rest[0] if rest else {}]]
    g.edges = [Edge.parse(a, b) for a, b in edges]
    if "boundary" in kw:
        g.materialize = Materialize(boundary=list(kw["boundary"]))
    return g


def msg(exc: pytest.ExceptionInfo) -> str:
    return str(exc.value)


# ── 정상 경로 ───────────────────────────────────────────────────────────


def test_valid_graph_compiles_and_orders_topologically():
    g = build(
        [
            ("n_img", "test.image_source@1.0.0"),
            ("n_exp", "test.expert@1.0.0"),
            ("n_crop", "test.crop@1.0.0"),
            ("n_out", "test.image_sink@1.0.0"),
        ],
        [
            ("n_img:image", "n_exp:subject"),
            ("n_img:image", "n_crop:image"),
            ("n_exp:regions", "n_crop:regions"),
            ("n_crop:crops", "n_out:images"),
        ],
        boundary=["n_crop"],
    )
    cg = compile_graph(g)
    assert cg.order.index("n_img") < cg.order.index("n_crop") < cg.order.index("n_out")
    # Input은 최상단 레인, Output은 최하단 레인에 고정된다
    assert cg.lanes["n_img"] == 0
    assert cg.lanes["n_out"] == max(cg.lanes.values())
    # 캐시 키는 노드마다 다르고 재컴파일에도 안정적이다
    keys = {n.cache_key for n in cg.nodes.values()}
    assert len(keys) == len(cg.nodes)
    assert compile_graph(g).spec_hash == cg.spec_hash


def test_generic_is_resolved_from_the_plugged_input():
    g = build(
        [
            ("n_ts", "test.ts_source@1.0.0"),
            ("n_exp", "test.expert@1.0.0"),
            ("n_pr", "test.prompt@1.0.0"),
            ("n_out", "test.sink@1.0.0"),
        ],
        [
            ("n_ts:series", "n_exp:subject"),
            ("n_pr:text", "n_out:text"),
        ],
        boundary=["n_exp"],
    )
    # expert 출력이 어디에도 쓰이지 않으면 도달 불가로 잡힌다 -> 소비자를 붙인다
    g.nodes.append(NodeInstance("n_txt", "test.text_asset@1.0.0", {}))
    g.edges.append(Edge.parse("n_txt:text", "n_pr:text"))
    with pytest.raises(CompileFailed) as e:
        compile_graph(g)
    assert "기여하지 않는 노드" in msg(e)  # n_exp가 죽은 가지임을 정확히 지목


def test_generic_output_type_is_ground_after_compile():
    g = build(
        [
            ("n_img", "test.image_source@1.0.0"),
            ("n_exp", "test.expert@1.0.0"),
            ("n_crop", "test.crop@1.0.0"),
            ("n_out", "test.image_sink@1.0.0"),
        ],
        [
            ("n_img:image", "n_exp:subject"),
            ("n_img:image", "n_crop:image"),
            ("n_exp:regions", "n_crop:regions"),
            ("n_crop:crops", "n_out:images"),
        ],
    )
    cg = compile_graph(g)
    rt = cg.nodes["n_exp"].output_types["regions"]
    assert rt.get_extra("domain") == "image2d" and rt.is_ground()


# ── G1 ──────────────────────────────────────────────────────────────────


def test_g1_rejects_double_normalization_and_says_what_would_have_happened():
    g = build(
        [
            ("n_img", "test.image_source@1.0.0"),
            ("n_norm", "test.norm@1.0.0"),
            ("n_crop", "test.crop@1.0.0"),
            ("n_out", "test.image_sink@1.0.0"),
        ],
        [
            ("n_img:image", "n_norm:image"),
            ("n_norm:image", "n_crop:image"),  # f32/CHW/imagenet -> u8/HWC 기대
            ("n_crop:crops", "n_out:images"),
        ],
    )
    with pytest.raises(CompileFailed) as e:
        compile_graph(g)
    m = msg(e)
    assert "TypeError [G1]" in m
    assert "dtype" in m and "norm" in m
    assert "이 배선이 통과했다면" in m
    assert "adapt.image_denorm" in m  # 구체적 해결 배선


def test_g1_rejects_domain_mismatch_between_image_and_series_regions():
    g = build(
        [
            ("n_ts", "test.ts_source@1.0.0"),
            ("n_img", "test.image_source@1.0.0"),
            ("n_exp", "test.expert@1.0.0"),
            ("n_crop", "test.crop@1.0.0"),
            ("n_out", "test.image_sink@1.0.0"),
        ],
        [
            ("n_ts:series", "n_exp:subject"),  # 시계열 구간 지목
            ("n_img:image", "n_crop:image"),
            ("n_exp:regions", "n_crop:regions"),  # 이미지 crop에 꽂으면 거부
            ("n_crop:crops", "n_out:images"),
        ],
    )
    with pytest.raises(CompileFailed) as e:
        compile_graph(g)
    assert "extra.domain" in msg(e)


# ── G2 ──────────────────────────────────────────────────────────────────


def test_g2_missing_required_input():
    g = build(
        [("n_crop", "test.crop@1.0.0"), ("n_out", "test.image_sink@1.0.0")],
        [("n_crop:crops", "n_out:images")],
    )
    with pytest.raises(CompileFailed) as e:
        compile_graph(g)
    assert "필수 입력 포트" in msg(e)


def test_g2_optional_input_may_stay_unconnected():
    g = build(
        [("n_kb", "test.text_asset@1.0.0"), ("n_pr", "test.prompt@1.0.0"), ("n_out", "test.sink@1.0.0")],
        [("n_kb:text", "n_pr:text"), ("n_pr:text", "n_out:text")],
    )
    compile_graph(g)  # n_out:extra는 optional이므로 미연결이어도 통과


def test_g2_graph_without_output_node():
    g = build(
        [("n_kb", "test.text_asset@1.0.0"), ("n_pr", "test.prompt@1.0.0")],
        [("n_kb:text", "n_pr:text")],
    )
    with pytest.raises(CompileFailed) as e:
        compile_graph(g)
    assert "Output 노드가 없다" in msg(e)


def test_g2_unreachable_node():
    g = build(
        [
            ("n_kb", "test.text_asset@1.0.0"),
            ("n_pr", "test.prompt@1.0.0"),
            ("n_out", "test.sink@1.0.0"),
            ("n_dead", "test.image_source@1.0.0"),
        ],
        [("n_kb:text", "n_pr:text"), ("n_pr:text", "n_out:text")],
    )
    with pytest.raises(CompileFailed) as e:
        compile_graph(g)
    assert "n_dead" in msg(e) and "기여하지 않는" in msg(e)


def test_g2_fan_in_is_forbidden():
    g = build(
        [
            ("n_a", "test.text_asset@1.0.0"),
            ("n_b", "test.text_asset@1.0.0"),
            ("n_pr", "test.prompt@1.0.0"),
            ("n_out", "test.sink@1.0.0"),
        ],
        [
            ("n_a:text", "n_pr:text"),
            ("n_b:text", "n_pr:text"),
            ("n_pr:text", "n_out:text"),
        ],
    )
    with pytest.raises(CompileFailed) as e:
        compile_graph(g)
    assert "팬인" in msg(e)


def test_g2_unknown_port_is_named_with_alternatives():
    g = build(
        [("n_kb", "test.text_asset@1.0.0"), ("n_out", "test.sink@1.0.0")],
        [("n_kb:nope", "n_out:text")],
    )
    with pytest.raises(CompileFailed) as e:
        compile_graph(g)
    assert "출력 포트 'nope'가 없다" in msg(e)


def test_g2_leakage_taint_reaches_prompt():
    """정답/라벨이 프롬프트로 흘러드는 경로는 컴파일이 거부한다."""
    g = build(
        [
            ("n_lbl", "test.label_source@1.0.0"),
            ("n_ans", "test.answer@1.0.0"),
            ("n_pr", "test.prompt@1.0.0"),
            ("n_out", "test.sink@1.0.0"),
        ],
        [
            ("n_lbl:value", "n_ans:value"),
            ("n_ans:answer", "n_pr:text"),
            ("n_pr:text", "n_out:text"),
        ],
    )
    with pytest.raises(CompileFailed) as e:
        compile_graph(g)
    assert "정답 누설" in msg(e)


def test_leakage_guard_clears_the_taint():
    g = build(
        [
            ("n_lbl", "test.label_source@1.0.0"),
            ("n_ans", "test.answer@1.0.0"),
            ("n_txt", "test.text_asset@1.0.0"),
            ("n_pr", "test.prompt@1.0.0"),
            ("n_guard", "test.leakage_guard@1.0.0"),
            ("n_out", "test.sink@1.0.0"),
        ],
        [
            ("n_lbl:value", "n_ans:value"),
            ("n_txt:text", "n_pr:text"),
            ("n_pr:text", "n_guard:prompt"),
            ("n_ans:answer", "n_guard:answer"),
            ("n_guard:prompt", "n_out:text"),
        ],
    )
    cg = compile_graph(g)
    assert cg.nodes["n_guard"].taint == frozenset()


def test_g2_external_call_must_sit_before_the_materialization_boundary():
    g = build(
        [
            ("n_img", "test.image_source@1.0.0"),
            ("n_norm", "test.norm@1.0.0"),
            ("n_exp", "test.expert@1.0.0"),
            ("n_crop", "test.crop@1.0.0"),
            ("n_out", "test.image_sink@1.0.0"),
        ],
        [
            ("n_img:image", "n_norm:image"),
            ("n_img:image", "n_exp:subject"),
            ("n_img:image", "n_crop:image"),
            ("n_exp:regions", "n_crop:regions"),
            ("n_crop:crops", "n_out:images"),
        ],
        boundary=["n_norm"],  # expert가 경계 뒤에 남는다
    )
    with pytest.raises(CompileFailed) as e:
        compile_graph(g)
    assert "물질화 경계 뒤" in msg(e)


def test_hand_written_cycle_is_reported_as_structure_error():
    """캔버스에서는 만들 수 없지만 손으로 쓴 YAML은 가능하다."""
    g = build(
        [
            ("a", "test.norm@1.0.0"),
            ("b", "test.norm@1.0.0"),
            ("n_out", "test.sink@1.0.0"),
        ],
        [("a:image", "b:image"), ("b:image", "a:image")],
    )
    with pytest.raises(CompileFailed) as e:
        compile_graph(g)
    assert "순환" in msg(e)


# ── Recipe 오버레이(문서 13의 기초) ─────────────────────────────────────


def test_recipe_overlay_only_touches_whitelisted_params():
    g = build(
        [
            ("n_img", "test.image_source@1.0.0"),
            ("n_exp", "test.expert@1.0.0"),
            ("n_crop", "test.crop@1.0.0"),
            ("n_out", "test.image_sink@1.0.0"),
        ],
        [
            ("n_img:image", "n_exp:subject"),
            ("n_img:image", "n_crop:image"),
            ("n_exp:regions", "n_crop:regions"),
            ("n_crop:crops", "n_out:images"),
        ],
    )
    cg = compile_graph(g, recipe_overrides={"n_crop.padding_ratio": 0.3})
    assert cg.nodes["n_crop"].params["padding_ratio"] == 0.3
    base = compile_graph(g)
    assert base.nodes["n_crop"].cache_key != cg.nodes["n_crop"].cache_key

    with pytest.raises(Exception, match="덮어쓸 수 없다"):
        compile_graph(g, recipe_overrides={"n_crop.min_side_px": 8})
