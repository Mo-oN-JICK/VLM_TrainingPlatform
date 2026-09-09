"""G4 — 자원 예산 검사.

GPU를 잡기 전 마지막 문. 포트 타입 구조체가 실어 나르는 shape 정보와 Trainer 설정만으로
샘플당 비전 토큰 수, 시퀀스 길이, 단계별 VRAM을 정적으로 추정한다.
예산을 넘기면 학습을 시작하지 않고 어느 파라미터가 초과분을 만들었는지 지목한다.

추정은 보수적으로 잡고 headroom을 요구한다. 설계 문서 07 §7.3의 공식을 그대로 쓴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from ..core.compiler import CompiledGraph
from ..core.node import NodeKind
from ..core.types import BaseKind, PortType, _Dyn
from ..plugins.base import BackboneSpec, resolve_backbone
from ..train.config import DTYPE_BYTES, OPTIMIZER_BYTES, Stage, TrainerConfig

GB = float(1 << 30)
CUDA_CONTEXT_GB = 1.0
ACT_BYTES = 2.0  # bf16 활성화
K_BLK = 8.0  # checkpointing on: 한 레이어 재계산 중 살아 있는 중간 텐서 계수
K_ALL = 12.0  # checkpointing off: 전 레이어 활성화 보존 계수
DEFAULT_TEXT_TOKENS = 512  # 실측이 없을 때의 보수적 가정


@dataclass
class Contribution:
    name: str
    gb: float
    cause: str = ""
    fix: str = ""


@dataclass
class StageEstimate:
    name: str
    weights: float = 0.0
    grads: float = 0.0
    optimizer: float = 0.0
    activations: float = 0.0
    logits: float = 0.0
    context: float = CUDA_CONTEXT_GB
    trainable_params: float = 0.0
    contributions: List[Contribution] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.weights + self.grads + self.optimizer + self.activations + self.logits + self.context

    def required(self, headroom: float) -> float:
        return self.total * (1.0 + headroom)


@dataclass
class BudgetResult:
    device: str = ""
    vram_gb: float = 0.0
    reserve_gb: float = 0.0
    headroom_ratio: float = 0.10
    images: int = 0
    tiles: int = 1
    tokens_per_tile: int = 0
    s_vision: int = 0
    s_text: int = 0
    text_estimated: bool = True
    max_len: int = 0
    max_context: int = 0
    stages: List[StageEstimate] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    sensitivity: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def limit(self) -> float:
        return self.vram_gb - self.reserve_gb

    @property
    def s_total(self) -> int:
        return self.s_vision + self.s_text

    @property
    def ok(self) -> bool:
        return not self.errors


# ── 그래프에서 비전 정보 읽기 ────────────────────────────────────────────


def vision_from_graph(cg: CompiledGraph, train_node: str) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Trainer로 들어가는 샘플의 이미지 개수 상한과 해상도를 포트 타입에서 읽는다.

    포트 타입이 shape를 실어 나르기 때문에 학습을 돌려보지 않고도 알 수 있다.
    """
    node = cg.nodes[train_node]
    src = node.inputs.get("sample")
    if not src:
        return 0, None
    producer = cg.nodes[src.split(":")[0]]
    imgs: Optional[PortType] = producer.input_types.get("images")
    if imgs is None or imgs.list_of is None:
        return 0, None
    n = int(imgs.list_of.max_n)
    hw: Optional[Tuple[int, int]] = None
    if imgs.shape and len(imgs.shape) == 3:
        h, w = imgs.shape[0], imgs.shape[1]
        if isinstance(h, int) and isinstance(w, int):
            hw = (h, w)
    return n, hw


# ── 추정 ────────────────────────────────────────────────────────────────


def _quantized_bytes(cfg: TrainerConfig) -> Tuple[float, float]:
    """(llm 그룹 바이트, 그 외 그룹 바이트)."""
    base = DTYPE_BYTES.get(cfg.dtype, 2.0)
    if cfg.quantization.mode in ("nf4", "int8"):
        return DTYPE_BYTES[cfg.quantization.mode] + 0.03, base
    return base, base


def _trainable_params(stage: Stage, spec: BackboneSpec) -> Tuple[float, Dict[str, float]]:
    detail: Dict[str, float] = {}
    total = 0.0
    for group, mode in stage.trainable.items():
        p = float(spec.params_by_group.get(group, 0.0))
        if mode == "true":
            detail[group] = p
        elif mode == "lora":
            detail[group] = spec.lora_params(stage.lora.r, tuple(stage.lora.targets))
        else:
            detail[group] = 0.0
        total += detail[group]
    return total, detail


def estimate(
    cfg: TrainerConfig,
    *,
    images: int = 0,
    image_hw: Optional[Tuple[int, int]] = None,
    measured_text_tokens: int = 0,
    with_sensitivity: bool = True,
) -> BudgetResult:
    spec = resolve_backbone(cfg.backbone).spec()
    vram, reserve = cfg.budget.limits()
    res = BudgetResult(
        device=cfg.budget.device,
        vram_gb=vram,
        reserve_gb=reserve,
        headroom_ratio=cfg.budget.headroom_ratio,
        max_len=cfg.sequence.max_len,
        max_context=spec.max_context,
    )

    # ── 시퀀스 길이 ──────────────────────────────────────────────────
    res.images = int(cfg.vision.max_images_per_sample or images)
    tpt = int(cfg.vision.tokens_per_tile or spec.tokens_per_tile)
    res.tokens_per_tile = tpt
    tiles = 1
    if cfg.vision.tiling:
        tiles = int(cfg.vision.max_tiles)
        if image_hw:
            need = math.ceil(image_hw[0] / cfg.vision.tile_px) * math.ceil(image_hw[1] / cfg.vision.tile_px)
            tiles = min(tiles, max(1, need))
            res.notes.append(
                f"타일 수 {tiles} = min(max_tiles {cfg.vision.max_tiles}, "
                f"{image_hw[0]}x{image_hw[1]} / {cfg.vision.tile_px}px 기준 {need})"
            )
    res.tiles = tiles
    res.s_vision = res.images * tiles * tpt

    if cfg.sequence.text_tokens:
        res.s_text, res.text_estimated = int(cfg.sequence.text_tokens), False
    elif measured_text_tokens:
        res.s_text, res.text_estimated = int(measured_text_tokens), False
        res.notes.append(f"텍스트 토큰 {res.s_text}는 dry-run 실측 문자 수에서 환산한 값이다")
    else:
        res.s_text = DEFAULT_TEXT_TOKENS
        res.notes.append(
            f"텍스트 토큰을 {DEFAULT_TEXT_TOKENS}으로 가정했다. "
            "dryrun을 먼저 돌리면 실측 문자 수로 환산한다."
        )

    if cfg.sequence.text_tokens and measured_text_tokens > cfg.sequence.text_tokens:
        res.errors.append(
            f"선언한 텍스트 토큰 {cfg.sequence.text_tokens}가 실측 {measured_text_tokens}보다 작다.\n"
            "  추정이 실측보다 작으면 예산 계산 전체가 낙관적으로 기운다.\n"
            f"  sequence.text_tokens를 {measured_text_tokens} 이상으로 올려라."
        )

    if res.s_total > cfg.sequence.max_len and cfg.sequence.truncation == "forbid":
        res.errors.append(
            f"시퀀스 길이 초과: 비전 {res.s_vision} + 텍스트 {res.s_text} = {res.s_total} "
            f"> max_len {cfg.sequence.max_len}\n"
            "  truncation: forbid 이므로 잘라내지 않고 거부한다.\n"
            "  이 검사가 없었다면: 이미지 뒷부분이 조용히 잘린 채 학습되고, "
            "손실은 정상적으로 떨어지며 성능만 나빠진다."
        )
    if res.s_total > spec.max_context:
        res.errors.append(
            f"백본 컨텍스트 초과: {res.s_total} > {spec.max_context} ({spec.id})"
        )

    # ── 단계별 메모리 ────────────────────────────────────────────────
    w_llm, w_other = _quantized_bytes(cfg)
    llm_p = float(spec.params_by_group.get("llm", spec.params_total))
    other_p = float(spec.params_total) - llm_p
    weights_gb = (llm_p * w_llm + other_p * w_other) / GB

    for st in cfg.stages:
        p_train, detail = _trainable_params(st, spec)
        e = StageEstimate(name=st.name, weights=weights_gb, trainable_params=p_train)
        e.grads = p_train * 2.0 / GB
        e.optimizer = p_train * OPTIMIZER_BYTES.get(st.optimizer, 8.0) / GB

        b, s_len, h, layers = st.per_device, res.s_total, spec.hidden, spec.n_layers
        act = (layers * b * s_len * h * ACT_BYTES + K_BLK * b * s_len * h * ACT_BYTES) if st.grad_checkpointing \
            else (K_ALL * layers * b * s_len * h * ACT_BYTES)
        e.activations = act / GB

        e.logits = (
            cfg.loss.chunk * spec.vocab * 2.0 * 2.0 if cfg.loss.chunked_ce
            else b * res.s_text * spec.vocab * 2.0 * 3.0
        ) / GB

        e.contributions = _contributions(cfg, st, e, detail, spec)
        res.stages.append(e)

        need = e.required(cfg.budget.headroom_ratio)
        if need > res.limit:
            res.errors.append(_over_budget_message(cfg, st, e, res, need))

    if with_sensitivity:
        res.sensitivity = _sensitivity(cfg, res, images, image_hw, measured_text_tokens)
    return res


def _contributions(
    cfg: TrainerConfig, st: Stage, e: StageEstimate, detail: Dict[str, float], spec: BackboneSpec
) -> List[Contribution]:
    out = [
        Contribution(
            "가중치",
            e.weights,
            f"backbone.quantization.mode = {cfg.quantization.mode}",
            "nf4로 바꾸면 llm 가중치가 1/4로 줄어든다" if cfg.quantization.mode == "none" else "",
        ),
        Contribution(
            "그래디언트",
            e.grads,
            f"stages[{st.name}].trainable = {st.trainable}",
            "학습 그룹을 lora로 바꾸면 거의 사라진다" if e.grads > 0.5 else "",
        ),
        Contribution(
            "옵티마이저",
            e.optimizer,
            f"stages[{st.name}].optimizer = {st.optimizer}",
            "adamw_bnb_8bit로 바꾸면 1/4로 줄어든다" if st.optimizer.startswith("adamw_torch") and e.optimizer > 0.5 else "",
        ),
        Contribution(
            "활성화",
            e.activations,
            f"per_device={st.per_device}, grad_checkpointing={st.grad_checkpointing}, "
            f"S={int(e.activations * GB / max(1.0, st.per_device * spec.hidden * ACT_BYTES * (spec.n_layers + K_BLK)))}",
            "grad_checkpointing을 켜면 크게 줄어든다" if not st.grad_checkpointing else
            ("per_device를 줄이거나 이미지 개수를 줄여라" if e.activations > 2.0 else ""),
        ),
        Contribution(
            "로짓",
            e.logits,
            f"loss.chunked_ce = {cfg.loss.chunked_ce}",
            "chunked_ce를 켜면 크게 줄어든다" if not cfg.loss.chunked_ce else "",
        ),
        Contribution("CUDA 컨텍스트", e.context, "고정값", ""),
    ]
    return sorted(out, key=lambda c: -c.gb)


def _over_budget_message(
    cfg: TrainerConfig, st: Stage, e: StageEstimate, res: BudgetResult, need: float
) -> str:
    lines = [
        f'예산 초과 — stage "{st.name}": 추정 {e.total:.1f} GB '
        f"(여유 {int(res.headroom_ratio * 100)}% 포함 {need:.1f} GB) > 예산 {res.limit:.1f} GB "
        f"[{res.device}, VRAM {res.vram_gb:.0f} - reserve {res.reserve_gb:.1f}]",
        "",
        "  초과 기여 (큰 순):",
    ]
    for i, c in enumerate(e.contributions[:3], 1):
        fix = f"  ({c.fix})" if c.fix else ""
        lines.append(f"    {i}) {c.name} {c.gb:.1f} GB — {c.cause}{fix}")
    lines += [
        "",
        "  이 검사가 없었다면:",
        "    물질화를 마치고 백본을 올린 뒤 첫 optimizer.step()에서 OOM.",
        "    체크포인트 없음. 한 장뿐인 GPU가 그동안 점유된다.",
    ]
    return "\n".join(lines)


def _sensitivity(
    cfg: TrainerConfig,
    base: BudgetResult,
    images: int,
    image_hw: Optional[Tuple[int, int]],
    measured: int,
) -> List[Tuple[str, float]]:
    """다른 값을 고정한 채 하나만 바꿨을 때 최대 단계의 총량이 얼마나 변하는지."""
    if not base.stages:
        return []
    base_peak = max(s.total for s in base.stages)
    out: List[Tuple[str, float]] = []

    def peak(c: TrainerConfig) -> float:
        try:
            r = estimate(c, images=images, image_hw=image_hw, measured_text_tokens=measured,
                         with_sensitivity=False)  # 재귀 금지
            return max((s.total for s in r.stages), default=0.0)
        except Exception:
            return base_peak

    variants: List[Tuple[str, TrainerConfig]] = []
    if base.images > 1:
        variants.append(
            (f"이미지 {base.images} -> {base.images - 1}",
             replace(cfg, vision=replace(cfg.vision, max_images_per_sample=base.images - 1)))
        )
    if base.tiles > 1:
        variants.append(
            (f"타일 {base.tiles} -> 1", replace(cfg, vision=replace(cfg.vision, tiling=False)))
        )
    if cfg.quantization.mode == "none":
        variants.append(
            ("양자화 none -> nf4", replace(cfg, quantization=replace(cfg.quantization, mode="nf4")))
        )
    if any(m == "true" for st in cfg.stages for m in st.trainable.values() if m):
        lora_stages = [
            replace(st, trainable={k: ("lora" if v == "true" and k == "llm" else v) for k, v in st.trainable.items()})
            for st in cfg.stages
        ]
        variants.append(("llm true -> lora", replace(cfg, stages=lora_stages)))
    if any(not st.grad_checkpointing for st in cfg.stages):
        variants.append(
            ("grad_checkpointing 켜기",
             replace(cfg, stages=[replace(st, grad_checkpointing=True) for st in cfg.stages]))
        )
    if any(st.per_device > 1 for st in cfg.stages):
        variants.append(
            ("per_device -> 1", replace(cfg, stages=[replace(st, per_device=1) for st in cfg.stages]))
        )

    for label, variant in variants:
        out.append((label, peak(variant) - base_peak))
    return out


# ── 렌더 ────────────────────────────────────────────────────────────────


def render(res: BudgetResult) -> str:
    lines = [
        f"자원 예산 [{res.device}] VRAM {res.vram_gb:.0f} GB - reserve {res.reserve_gb:.1f} "
        f"= 예산 {res.limit:.1f} GB (여유 {int(res.headroom_ratio * 100)}% 요구)",
        f"  시퀀스: 비전 {res.s_vision} (= 이미지 {res.images} x 타일 {res.tiles} x {res.tokens_per_tile}) "
        f"+ 텍스트 {res.s_text}{' (가정)' if res.text_estimated else ' (실측)'} "
        f"= {res.s_total} / max_len {res.max_len} / 백본 컨텍스트 {res.max_context}",
        "",
        f"  {'단계':<20}{'가중치':>9}{'그래디언트':>11}{'옵티마이저':>11}{'활성화':>9}{'로짓':>8}{'합계':>9}  판정",
    ]
    for s in res.stages:
        need = s.required(res.headroom_ratio)
        verdict = "통과" if need <= res.limit else "거부"
        lines.append(
            f"  {s.name:<20}{s.weights:>9.1f}{s.grads:>11.1f}{s.optimizer:>11.1f}"
            f"{s.activations:>9.1f}{s.logits:>8.1f}{s.total:>9.1f}  {verdict} ({need:.1f})"
        )
    for n in res.notes:
        lines.append(f"  · {n}")
    if res.sensitivity:
        lines.append("")
        lines.append("  민감도 (다른 값 고정, 최대 단계 기준):")
        for label, delta in res.sensitivity:
            lines.append(f"    {label:<26} {delta:+.1f} GB")
    if res.errors:
        lines.append("")
        lines.extend(res.errors)
    else:
        lines.append("")
        lines.append("  통과. 네 게이트를 모두 지났다.")
    return "\n".join(lines)


# ── 그래프와 묶어서 쓰는 진입점 ──────────────────────────────────────────


def train_nodes(cg: CompiledGraph) -> List[str]:
    return [i for i in cg.order if cg.nodes[i].kind is NodeKind.OUTPUT and cg.nodes[i].ref.startswith("train.")]


def for_graph(
    cg: CompiledGraph, cfg: TrainerConfig, train_node: str, measured_text_tokens: int = 0
) -> BudgetResult:
    images, hw = vision_from_graph(cg, train_node)
    return estimate(cfg, images=images, image_hw=hw, measured_text_tokens=measured_text_tokens)
