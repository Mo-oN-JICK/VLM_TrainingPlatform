"""Phase 0 완료 조건 — 타입 호환성 7규칙."""

from __future__ import annotations

import pytest

from vlm_trainer.core.types import (
    ANY,
    DType,
    DimVar,
    Frame,
    Layout,
    Norm,
    Range,
    image,
    text,
    timeseries,
)
from vlm_trainer.core.unify import apply_subst, unify_ports


def fields_of(res):
    return {m.field for m in res.mismatches}


def test_identical_types_connect():
    assert unify_ports(image(), image()).ok


def test_normalized_image_is_a_different_type():
    """설계의 출발점 — u8/HWC/0-255와 f32/CHW/imagenet은 다른 타입이다."""
    src = image()
    dst = image(
        dtype=DType.F32,
        layout=Layout.CHW,
        value_range=Range.UNIT_0_1,
        norm=Norm.IMAGENET,
        shape=(3, DimVar("H"), DimVar("W")),
    )
    res = unify_ports(src, dst)
    assert not res.ok
    assert {"dtype", "layout", "value_range", "norm"} <= fields_of(res)


def test_base_kind_has_no_subtyping():
    assert not unify_ports(image(), timeseries(hz=500.0)).ok


def test_frame_mismatch_is_rejected():
    res = unify_ports(image(frame=Frame.CROP_PX), image(frame=Frame.ORIG_PX))
    assert not res.ok and "frame" in fields_of(res)


def test_time_base_mismatch_is_rejected():
    res = unify_ports(timeseries(hz=250.0), timeseries(hz=500.0))
    assert not res.ok and any(f.startswith("time_base") for f in fields_of(res))


def test_dst_any_declares_irrelevance():
    dst = image(layout=ANY, dtype=ANY, value_range=ANY, norm=ANY, colorspace=ANY, frame=ANY)
    assert unify_ports(image(), dst).ok
    assert unify_ports(image(dtype=DType.F32, layout=Layout.CHW), dst).ok


def test_src_any_is_conservatively_rejected():
    """소스가 필드를 확정하지 않았다면 통과시키지 않는다."""
    res = unify_ports(image(dtype=ANY), image())
    assert not res.ok and "dtype" in fields_of(res)


def test_semantic_requirement_is_subset_check():
    assert unify_ports(text("prompt", "evidence"), text("prompt")).ok
    res = unify_ports(text(), text("prompt"))
    assert not res.ok and "semantic" in fields_of(res)


def test_optional_source_cannot_feed_required_input():
    res = unify_ports(text("prompt").as_optional(), text("prompt"))
    assert not res.ok and "optional" in fields_of(res)
    assert unify_ports(text("prompt").as_optional(), text("prompt").as_optional()).ok


def test_list_and_scalar_are_different_types():
    res = unify_ports(image().as_list(1, 8), image())
    assert not res.ok and "list" in fields_of(res)
    res2 = unify_ports(image(), image().as_list(1, 8))
    assert not res2.ok and "list" in fields_of(res2)


def test_list_bounds_must_overlap():
    assert unify_ports(image().as_list(1, 4), image().as_list(2, 16)).ok
    res = unify_ports(image().as_list(8, 16), image().as_list(1, 4))
    assert not res.ok and "list.n" in fields_of(res)


def test_symbolic_dims_unify_and_bind():
    src = image(shape=(448, 448, 3))
    dst = image(shape=(DimVar("H"), DimVar("W"), 3))
    res = unify_ports(src, dst)
    assert res.ok
    assert res.subst["H"] == 448 and res.subst["W"] == 448
    ground = apply_subst(dst, res.subst)
    assert ground.shape == (448, 448, 3) and ground.is_ground()


def test_dynamic_dim_cannot_satisfy_a_fixed_size_requirement():
    """가변 크기를 고정 크기 포트에 꽂으면 거부하고 리사이즈를 요구한다."""
    res = unify_ports(image(), image(shape=(448, 448, 3)))
    assert not res.ok
    assert any("adapt.image_resize" in (m.note or "") for m in res.mismatches)
    assert unify_ports(image(shape=(448, 448, 3)), image()).ok  # 고정 -> 가변 허용


def test_conflicting_concrete_dims_are_rejected():
    res = unify_ports(image(shape=(448, 448, 3)), image(shape=(224, 224, 3)))
    assert not res.ok and any(f.startswith("shape[") for f in fields_of(res))


def test_rank_mismatch_is_rejected():
    res = unify_ports(image(shape=(448, 448, 3)), image(shape=(448, 448)))
    assert not res.ok and "shape.rank" in fields_of(res)


def test_same_dim_var_must_bind_consistently():
    subst = {}
    a = unify_ports(image(shape=(448, 448, 3)), image(shape=(DimVar("S"), DimVar("S"), 3)), subst)
    assert a.ok
    b = unify_ports(image(shape=(224, 224, 3)), image(shape=(DimVar("S"), DimVar("S"), 3)), a.subst)
    assert not b.ok  # S는 이미 448로 묶였다


def test_extra_constraints_like_regions_domain():
    from vlm_trainer.core.types import regions

    assert unify_ports(regions(domain="image2d", frame=Frame.ORIG_PX),
                       regions(domain="image2d", frame=Frame.ORIG_PX)).ok
    res = unify_ports(regions(domain="series1d", frame=Frame.TS_SECONDS),
                      regions(domain="image2d", frame=Frame.ORIG_PX))
    assert not res.ok


def test_type_display_is_readable():
    assert str(image()) == "Image{u8, HWC, 0-255, none, RGB, orig_px, [dyn,dyn,3]}"
    assert str(image().as_list(1, 4)).startswith("ImageList{")
