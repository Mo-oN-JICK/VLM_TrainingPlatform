"""Data Acquisition — 전부 Input 노드.

Input 노드는 배선으로 값을 받지 않는다. Project의 sample_space가 정의한 행에서
컬럼을 파라미터로 지정해 값을 주입한다. 파일 시스템에 닿는 유일한 분류다.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any, Dict, List, Optional

import numpy as np

from ..answer.schema import AnswerSchema
from ..core.node import Node, NodeDoc, NodeKind, Port, RunCtx
from ..core.registry import register
from ..core.types import (
    DYN,
    BaseKind,
    Color,
    DType,
    Frame,
    Layout,
    Norm,
    PortType,
    Range,
    image,
    simple,
    text,
    timeseries,
)


def _stat_fingerprint(path: str) -> str:
    try:
        st = os.stat(path)
        h = blake2b(f"{os.path.normcase(os.path.abspath(path))}|{st.st_size}|{st.st_mtime_ns}".encode(), digest_size=12)
        return h.hexdigest()
    except OSError:
        return "missing"


@dataclass
class ImageSourceParams:
    column: str = "image_path"
    color_space: str = "RGB"
    on_missing: str = "fail"  # fail | skip


@register(
    type="source.image",
    version="1.0.0",
    category="Data Acquisition",
    kind=NodeKind.INPUT,
    outputs={"image": Port(image().with_sem("raw_image"), "원본 이미지")},
    params=ImageSourceParams,
    recipe_overridable=["column"],
    preview="image",
    doc=NodeDoc(
        summary="sample_space의 경로 컬럼에서 이미지를 읽어 그래프에 주입한다.",
        scenario="모든 이미지 파이프라인의 시작점.",
    ),
)
class ImageSource(Node):
    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        return _stat_fingerprint(ctx.path(str(ctx.sample.get(params.column, ""))))

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        from PIL import Image as PILImage

        path = ctx.path(str(ctx.sample[params.column]))
        with PILImage.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return {"image": arr}


@dataclass
class TimeSeriesParams:
    column: str = "series_path"
    format: str = "csv"
    hz: float = 500.0
    channels: int = 1
    t0_policy: str = "sample_start"


@register(
    type="source.timeseries",
    version="1.0.0",
    category="Data Acquisition",
    kind=NodeKind.INPUT,
    outputs={"series": Port(timeseries(hz=500.0).with_sem("raw_signal"), "센서 시계열")},
    params=TimeSeriesParams,
    recipe_overridable=["column", "hz", "channels"],
    type_affecting=["hz", "channels"],
    preview="timeseries_plot",
    doc=NodeDoc(summary="CSV/NPY 시계열을 읽어 [T, C] 배열로 주입한다."),
)
class TimeSeriesSource(Node):
    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        return {
            "series": timeseries(hz=float(params.hz), shape=(DYN, int(params.channels))).with_sem(
                "raw_signal"
            )
        }

    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        return _stat_fingerprint(ctx.path(str(ctx.sample.get(params.column, ""))))

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        path = ctx.path(str(ctx.sample[params.column]))
        if params.format == "npy":
            arr = np.load(path).astype(np.float32)
        else:
            rows: List[List[float]] = []
            with open(path, "r", encoding="utf-8", newline="") as fh:
                for row in csv.reader(fh):
                    if not row or row[0].lstrip().startswith("#"):
                        continue
                    try:
                        rows.append([float(x) for x in row])
                    except ValueError:
                        continue  # 헤더 줄
            arr = np.asarray(rows, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[1] != int(params.channels):
            raise ValueError(
                f"채널 수가 선언과 다르다: 선언 {params.channels}, 실제 {arr.shape[1]} ({path})"
            )
        return {"series": arr}


@dataclass
class TextAssetParams:
    path: str = "knowledge/rules.md"
    encoding: str = "utf-8"
    max_chars: int = 4000


@register(
    type="source.text_asset",
    version="1.0.0",
    category="Data Acquisition",
    kind=NodeKind.INPUT,
    outputs={"text": Port(text("domain_knowledge"), "도메인 지식 텍스트")},
    params=TextAssetParams,
    recipe_overridable=["path", "max_chars"],
    preview="text",
    doc=NodeDoc(summary="프롬프트에 넣을 도메인 지식 텍스트 자산을 읽는다."),
)
class TextAssetSource(Node):
    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        return _stat_fingerprint(ctx.path(params.path))

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        with open(ctx.path(params.path), "r", encoding=params.encoding) as fh:
            return {"text": fh.read()[: int(params.max_chars)]}


@dataclass
class FieldParams:
    column: str = "label"
    semantic: str = "label"
    strip: bool = True


@register(
    type="source.field",
    version="1.0.0",
    category="Data Acquisition",
    kind=NodeKind.INPUT,
    outputs={"value": Port(text("label"), "인덱스의 한 컬럼")},
    params=FieldParams,
    recipe_overridable=["column"],
    preview="text",
    doc=NodeDoc(
        summary="sample_space의 컬럼 하나를 텍스트로 주입한다.",
        scenario="정답 라벨이나 메타데이터를 그래프로 들여올 때. semantic 태그가 누설 검사의 기준이 된다.",
    ),
)
class FieldSource(Node):
    def infer_types(self, inputs: Dict[str, PortType], params: Any) -> Dict[str, PortType]:
        return {"value": text(str(params.semantic))}

    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        return f"col:{params.column}"

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        v = str(ctx.sample.get(params.column, ""))
        return {"value": v.strip() if params.strip else v}


@dataclass
class MetadataParams:
    columns: tuple = ()


@register(
    type="source.metadata",
    version="1.0.0",
    category="Data Acquisition",
    kind=NodeKind.INPUT,
    outputs={"meta": Port(simple(BaseKind.TABLE), "메타데이터 표")},
    params=MetadataParams,
    preview="table",
    doc=NodeDoc(summary="인덱스의 여러 컬럼을 한 표로 묶어 주입한다."),
)
class MetadataSource(Node):
    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        return "cols:" + ",".join(params.columns or ())

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        cols = params.columns or tuple(ctx.sample.keys())
        return {"meta": {c: ctx.sample.get(c) for c in cols}}


@dataclass
class SchemaParams:
    path: str = "schemas/answer.yaml"


@register(
    type="schema.define",
    version="1.0.0",
    category="Answer Design",
    kind=NodeKind.INPUT,
    outputs={"schema": Port(simple(BaseKind.SCHEMA), "정답 Text 스키마")},
    params=SchemaParams,
    preview="schema",
    doc=NodeDoc(
        summary="정답 Text 스키마를 읽는다. 렌더러와 파서가 모두 이 정의에서 생성된다.",
        scenario="학습에 쓴 스키마가 추론 계약에도 그대로 실린다.",
    ),
)
class SchemaDefine(Node):
    def fingerprint(self, ctx: RunCtx, params: Any) -> str:
        return _stat_fingerprint(ctx.path(params.path))

    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        return {"schema": AnswerSchema.load(ctx.path(params.path))}
