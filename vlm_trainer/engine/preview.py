"""노드 단위 미리보기.

Mech-Vision의 Debug Output 규약을 그대로 따른다.
  - 토글이 꺼져 있으면 미리보기를 생성조차 하지 않는다(계산 절약)
  - 외부에서 트리거된 실행에서는 값과 무관하게 표시하지 않는다
  - Output 노드는 미리보기하지 않는다. 부작용이 있기 때문이다
노드 타입별로 "시각화 출력"이 무엇인지는 설계 문서 08 §8.4의 표를 따른다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Preview:
    node_id: str
    kind: str
    text: str = ""
    image_path: str = ""


def allowed(trigger: str, debug_output: bool) -> bool:
    if trigger == "external":
        return False  # 외부 트리거는 값과 무관하게 표시하지 않는다
    if trigger == "ui":
        return debug_output
    return True  # cli는 명시적으로 요청했을 때만 호출된다


def render(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str = "") -> Preview:
    fn = _RENDERERS.get(kind, _generic)
    return fn(node_id, kind, outputs, out_dir)


def _save_png(arr: np.ndarray, out_dir: str, name: str) -> str:
    if not out_dir:
        return ""
    from PIL import Image as PILImage

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{name}.png")
    PILImage.fromarray(np.asarray(arr, dtype=np.uint8)).save(p)
    return p


def _first(outputs: Dict[str, Any], *types: type) -> Any:
    for v in outputs.values():
        if isinstance(v, types):
            return v
    return None


def _generic(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    from .values import describe

    body = "\n".join(f"  {k}: {describe(v)}" for k, v in outputs.items())
    return Preview(node_id, kind, body)


def _image(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    arr = _first(outputs, np.ndarray)
    if arr is None:
        return _generic(node_id, kind, outputs, out_dir)
    p = _save_png(arr, out_dir, node_id.replace("/", "_"))
    return Preview(node_id, kind, f"  이미지 {list(arr.shape)} {arr.dtype}", p)


def _image_grid(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    lst = _first(outputs, list)
    if not lst:
        return _generic(node_id, kind, outputs, out_dir)
    lines = [f"  이미지 {len(lst)}장"] + [f"    [{i}] {list(np.asarray(a).shape)}" for i, a in enumerate(lst[:6])]
    p = _save_png(lst[0], out_dir, node_id.replace("/", "_"))
    return Preview(node_id, kind, "\n".join(lines), p)


def _table(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    d = _first(outputs, dict) or {}
    return Preview(node_id, kind, "\n".join(f"  {k:<16} {v}" for k, v in d.items()))


def _text(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    v = _first(outputs, str, list)
    if isinstance(v, list):
        return Preview(node_id, kind, "\n".join(f"  {x}" for x in v))
    return Preview(node_id, kind, "\n".join(f"  {ln}" for ln in str(v).splitlines()))


def _prompt_render(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    s = str(outputs.get("text", ""))
    n_slots = s.count("<image>")
    head = f"  렌더된 최종 프롬프트 — {len(s)}자, 이미지 자리표시자 {n_slots}개"
    return Preview(node_id, kind, head + "\n" + "\n".join(f"  | {ln}" for ln in s.splitlines()))


def _answer_render(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    s = str(outputs.get("answer", ""))
    return Preview(node_id, kind, "\n".join(f"  | {ln}" for ln in s.splitlines()))


def _answer_validated(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    rep = outputs.get("report") or {}
    viol = rep.get("violations") or []
    mark = "통과" if not viol else "위반 " + ", ".join(viol)
    body = "\n".join(f"  | {ln}" for ln in str(outputs.get("answer", "")).splitlines())
    return Preview(node_id, kind, f"  스키마 검증: {mark}\n{body}")


def _regions_overlay(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    rs = outputs.get("regions") or []
    lines = [f"  지목 {len(rs)}건"]
    for i, r in enumerate(rs[:8]):
        ext = ", ".join(f"{float(x):.1f}" for x in r["extent"])
        lines.append(f"    [{i}] score={float(r.get('score', 0)):.3f} extent=({ext}) {r.get('label', '')}")
    return Preview(node_id, kind, "\n".join(lines))


def _leak_report(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    rep = outputs.get("report") or {}
    hits = rep.get("leak_terms") or []
    return Preview(node_id, kind, f"  누설 검사: {'통과' if not hits else '의심 어휘 ' + str(hits)}")


def _sample_card(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    s = outputs.get("sample") or {}
    imgs = s.get("images") or []
    lines = [
        f"  id: {s.get('id')}",
        f"  이미지 {len(imgs)}장" + (f" 첫 장 {list(np.asarray(imgs[0]).shape)}" if imgs else ""),
        "  프롬프트:",
    ]
    lines += [f"    | {ln}" for ln in str(s.get("prompt", "")).splitlines()[:12]]
    lines.append("  정답:")
    lines += [f"    | {ln}" for ln in str(s.get("answer", "")).splitlines()]
    p = _save_png(imgs[0], out_dir, node_id.replace("/", "_")) if imgs else ""
    return Preview(node_id, kind, "\n".join(lines), p)


def _timeseries_plot(node_id: str, kind: str, outputs: Dict[str, Any], out_dir: str) -> Preview:
    arr = _first(outputs, np.ndarray)
    if arr is None:
        return _generic(node_id, kind, outputs, out_dir)
    return Preview(
        node_id,
        kind,
        f"  시계열 {list(arr.shape)} {arr.dtype} · 범위 [{float(arr.min()):.3f}, {float(arr.max()):.3f}]",
    )


_RENDERERS = {
    "image": _image,
    "image_grid": _image_grid,
    "table": _table,
    "text": _text,
    "schema": _generic,
    "prompt_render": _prompt_render,
    "answer_render": _answer_render,
    "answer_validated": _answer_validated,
    "regions_overlay": _regions_overlay,
    "leak_report": _leak_report,
    "sample_card": _sample_card,
    "timeseries_plot": _timeseries_plot,
}
