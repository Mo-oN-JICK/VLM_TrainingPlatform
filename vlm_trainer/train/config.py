"""Trainer의 선언형 설정.

다단계 학습·freeze 정책·양자화·오프로딩을 그래프가 아니라 이 스키마로 받는다.
단일 GPU 예산 안에 들어가기 위한 수단이 전부 같은 스키마에 선언되고,
예산 산정(G4)은 이 선언과 포트 타입만으로 학습 전에 계산된다. 설계 문서 07.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ── 장치 프로파일 ────────────────────────────────────────────────────────
# 기본은 지금 이 PC다. 4090은 다른 PC에 있으므로 프로파일로만 열어 둔다.
DEVICES: Dict[str, Dict[str, float]] = {
    "rtx3060_12gb": {"vram_gb": 12.0, "reserve_gb": 1.0},
    "rtx4090_24gb": {"vram_gb": 24.0, "reserve_gb": 1.5},
    "a100_40gb": {"vram_gb": 40.0, "reserve_gb": 2.0},
}

# 파라미터 하나가 차지하는 바이트
DTYPE_BYTES = {"fp32": 4.0, "f32": 4.0, "bf16": 2.0, "fp16": 2.0, "f16": 2.0, "int8": 1.0, "nf4": 0.5}
# 옵티마이저 상태가 학습 파라미터 하나당 차지하는 바이트
OPTIMIZER_BYTES = {
    "adamw_torch": 8.0,
    "adamw_fused": 8.0,
    "adamw_bnb_8bit": 2.0,
    "adafactor": 0.5,
    "sgd": 0.0,
    "sgd_momentum": 4.0,
}


# 기본 실행 프로파일(단일 GPU / Windows)에서 돌지 않는 선택지.
# 여기 걸리면 학습을 시작하지 않는다 — 큐에 넣고 퇴근한 밤이 통째로 날아가는 것을 막는다.
PROFILE_UNSUPPORTED: Dict[str, Dict[str, str]] = {
    "windows_single_gpu": {
        "attn_impl:flash_attn2": "sdpa로 바꿔라. flash-attn은 Windows 휠이 없다",
        "optimizer:deepspeed_adam": "adamw_torch 또는 adamw_bnb_8bit를 써라. DeepSpeed는 Windows를 지원하지 않는다",
        "optimizer:adamw_apex_fused": "adamw_torch를 써라. apex는 Windows 빌드가 없다",
        "offload:disk": "offload를 none 또는 cpu로 두어라",
        "distributed:true": "기본 프로파일은 단일 GPU다. 다중 GPU는 확장 지점이지 기본값이 아니다",
    }
}


@dataclass
class Quantization:
    mode: str = "none"  # none | int8 | nf4
    compute_dtype: str = "bf16"
    double_quant: bool = True


@dataclass
class Vision:
    max_images_per_sample: int = 0  # 0이면 그래프의 ImageList 상한을 쓴다
    tiling: bool = False
    tile_px: int = 448
    max_tiles: int = 1
    tokens_per_tile: int = 0  # 0이면 백본 어댑터가 보고한 값을 쓴다


@dataclass
class Sequence:
    max_len: int = 4096
    truncation: str = "forbid"  # forbid | right | left
    text_tokens: int = 0  # 0이면 chars_per_token으로 추정한다
    chars_per_token: float = 2.5  # 한국어 혼합 텍스트의 보수적 추정치


@dataclass
class Loss:
    chunked_ce: bool = True
    chunk: int = 1024


@dataclass
class Lora:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class Stage:
    name: str = "stage"
    init_from: str = ""
    # 모듈 그룹별 3상태: "false"(동결) | "true"(전체 학습) | "lora"(어댑터만)
    trainable: Dict[str, str] = field(
        default_factory=lambda: {"vision_tower": "false", "projector": "true", "llm": "false"}
    )
    lora: Lora = field(default_factory=Lora)
    optimizer: str = "adamw_torch"
    lr: float = 1e-4
    per_device: int = 1
    grad_accum: int = 16
    grad_checkpointing: bool = True
    amp: str = "bf16"
    epochs: int = 1


@dataclass
class BudgetPolicy:
    device: str = "rtx3060_12gb"
    vram_gb: float = 0.0  # 0이면 프로파일 값
    reserve_gb: float = 0.0
    policy: str = "fail_fast"  # fail_fast | warn
    headroom_ratio: float = 0.10

    def limits(self) -> Tuple[float, float]:
        d = DEVICES.get(self.device) or DEVICES["rtx3060_12gb"]
        return (self.vram_gb or d["vram_gb"], self.reserve_gb or d["reserve_gb"])


@dataclass
class Distributed:
    enabled: bool = False
    strategy: str = "none"


@dataclass
class TrainerConfig:
    backbone: str = "dummy-2b"
    adapter: str = ""
    dtype: str = "bf16"
    attn_impl: str = "sdpa"
    offload: str = "none"  # none | cpu | disk
    quantization: Quantization = field(default_factory=Quantization)
    vision: Vision = field(default_factory=Vision)
    sequence: Sequence = field(default_factory=Sequence)
    loss: Loss = field(default_factory=Loss)
    stages: List[Stage] = field(default_factory=list)
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    distributed: Distributed = field(default_factory=Distributed)
    source_path: str = ""

    def profile_errors(self, profile: str) -> List[str]:
        """실행 프로파일이 지원하지 않는 선택지를 골라낸다."""
        table = PROFILE_UNSUPPORTED.get(profile) or {}
        picked = [f"attn_impl:{self.attn_impl}", f"offload:{self.offload}",
                  f"distributed:{str(self.distributed.enabled).lower()}"]
        picked += [f"optimizer:{s.optimizer}" for s in self.stages]
        out: List[str] = []
        for key in picked:
            if key in table:
                field_name, value = key.split(":", 1)
                out.append(
                    f"실행 프로파일 {profile}가 지원하지 않는 설정: {field_name} = {value}\n"
                    f"  {table[key]}\n"
                    "  이 검사가 없었다면: 물질화를 마치고 백본을 올린 직후 임포트 에러로 죽는다."
                )
        return out

    # ── 로드 ────────────────────────────────────────────────────────────
    @staticmethod
    def from_dict(d: Dict[str, Any], source_path: str = "") -> "TrainerConfig":
        c = TrainerConfig(
            backbone=str(d.get("backbone", "dummy-2b")),
            adapter=str(d.get("adapter", "")),
            dtype=str(d.get("dtype", "bf16")),
            quantization=Quantization(**(d.get("quantization") or {})),
            vision=Vision(**(d.get("vision") or {})),
            sequence=Sequence(**(d.get("sequence") or {})),
            loss=Loss(**(d.get("loss") or {})),
            budget=BudgetPolicy(**(d.get("budget") or {})),
            distributed=Distributed(**(d.get("distributed") or {})),
            attn_impl=str(d.get("attn_impl", "sdpa")),
            offload=str(d.get("offload", "none")),
            source_path=source_path,
        )
        for s in d.get("stages") or []:
            s = dict(s)
            lora = Lora(**{**(s.pop("lora", None) or {}), })
            if isinstance(lora.targets, list):
                lora = replace(lora, targets=tuple(lora.targets))
            c.stages.append(Stage(lora=lora, **s))
        if not c.stages:
            c.stages = [Stage(name="default")]
        return c

    @staticmethod
    def load(path: str) -> "TrainerConfig":
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
        if d.get("kind") not in (None, "TrainerConfig"):
            raise ValueError(f"{path}: kind는 TrainerConfig여야 한다")
        d.pop("kind", None)
        d.pop("version", None)
        return TrainerConfig.from_dict(d, source_path=path)

    def digest(self) -> Dict[str, Any]:
        """캐시 키·추론 계약에 들어가는 정규 표현."""
        return {
            "backbone": self.backbone,
            "dtype": self.dtype,
            "quantization": self.quantization.__dict__,
            "vision": self.vision.__dict__,
            "sequence": self.sequence.__dict__,
            "stages": [
                {
                    "name": s.name,
                    "trainable": s.trainable,
                    "lora": {"r": s.lora.r, "targets": list(s.lora.targets)},
                    "optimizer": s.optimizer,
                    "per_device": s.per_device,
                    "grad_accum": s.grad_accum,
                    "grad_checkpointing": s.grad_checkpointing,
                }
                for s in self.stages
            ],
        }
