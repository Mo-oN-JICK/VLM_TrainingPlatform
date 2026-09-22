"""Trainer의 선언형 설정.

다단계 학습·freeze 정책·양자화·오프로딩을 그래프가 아니라 이 스키마로 받는다.
단일 GPU 예산 안에 들어가기 위한 수단이 전부 같은 스키마에 선언되고,
예산 산정(G4)은 이 선언과 포트 타입만으로 학습 전에 계산된다. 설계 문서 07.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Tuple

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
    # 기본 프로파일. 이 표가 비지 않는 한 다중 GPU는 기본값이 되지 않는다.
    "windows_single_gpu": {
        "attn_impl:flash_attn2": "sdpa로 바꿔라. flash-attn은 Windows 휠이 없다",
        "optimizer:deepspeed_adam": "adamw_torch 또는 adamw_bnb_8bit를 써라. DeepSpeed는 Windows를 지원하지 않는다",
        "optimizer:adamw_apex_fused": "adamw_torch를 써라. apex는 Windows 빌드가 없다",
        "offload:disk": "offload를 none 또는 cpu로 두어라",
        "distributed:true": "기본 프로파일은 단일 GPU다. 다중 GPU는 확장 지점이지 기본값이 아니다",
    },
    # 확장 프로파일. 프로파일을 바꾸는 것은 스펙의 한 줄이고, 그래프는 그대로다.
    # 이 표는 "그 프로파일에서 **돌지 않는** 것"만 담는다. 느린 것은 여기 들어오지 않는다 —
    # 게이트가 취향을 말하기 시작하면 게이트를 믿지 않게 된다.
    "linux_multi_gpu": {},
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
    """다중 GPU는 **확장 지점이지 기본값이 아니다.**

    지금 예산이 답할 수 있는 것은 DDP뿐이다. DDP는 장치마다 모델을 통째로 들고 있어
    장치당 VRAM이 단일 GPU와 같고, 늘어나는 것은 유효 배치다. FSDP나 DeepSpeed는
    가중치와 옵티마이저를 쪼개 장치당 VRAM 자체가 달라지는데, 그 계산은 아직 없다.
    모르는 것을 아는 척 답하면 G4가 존재할 이유가 사라지므로 거부한다.
    """

    enabled: bool = False
    strategy: str = "none"  # none | ddp  (fsdp/deepspeed는 아직 예산을 모른다)
    world_size: int = 1

    KNOWN = ("none", "ddp")
    UNMODELED = ("fsdp", "deepspeed")

    def errors(self) -> List[str]:
        out: List[str] = []
        if not self.enabled:
            if self.world_size != 1:
                out.append(
                    f"distributed.enabled가 false인데 world_size가 {self.world_size}다.\n"
                    "  둘 중 하나가 오타다. 켜지 않은 다중 GPU는 없다.\n"
                    "  이 검사가 없었다면: 한 장으로 돌면서 유효 배치를 네 배로 적어 둔 로그가 남는다."
                )
            if self.strategy not in ("none", ""):
                out.append(
                    f"distributed.enabled가 false인데 strategy가 {self.strategy!r}다.\n"
                    "  enabled를 켜거나 strategy를 none으로 두어라."
                )
            return out

        if self.strategy in self.UNMODELED:
            out.append(
                f"distributed.strategy = {self.strategy}는 아직 예산을 계산하지 못한다.\n"
                "  이 전략은 가중치와 옵티마이저를 장치에 쪼개므로 장치당 VRAM이 달라지는데,\n"
                "  그 모델이 없다. ddp를 쓰거나, 이 전략의 예산 계산을 먼저 구현하라.\n"
                "  이 검사가 없었다면: 단일 GPU 기준 숫자로 통과시킨 뒤 실제로는 다른 곳에서 터진다."
            )
        elif self.strategy not in self.KNOWN:
            out.append(
                f"알 수 없는 distributed.strategy {self.strategy!r} "
                f"(아는 것: {sorted(self.KNOWN + self.UNMODELED)})"
            )
        if self.world_size < 2:
            out.append(
                f"distributed.enabled가 true인데 world_size가 {self.world_size}다.\n"
                "  다중 GPU를 켰으면 장치가 둘 이상이어야 한다."
            )
        return out

    def effective_multiplier(self) -> int:
        """유효 배치가 몇 배가 되는가. DDP는 장치 수만큼 곱해진다."""
        return self.world_size if self.enabled else 1


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
        """실행 프로파일이 지원하지 않는 선택지와, 그 자체로 모순인 설정을 골라낸다."""
        out_self = self.distributed.errors()
        table = PROFILE_UNSUPPORTED.get(profile) or {}
        picked = [f"attn_impl:{self.attn_impl}", f"offload:{self.offload}",
                  f"distributed:{str(self.distributed.enabled).lower()}"]
        picked += [f"optimizer:{s.optimizer}" for s in self.stages]
        out: List[str] = list(out_self)
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
            distributed=Distributed(
                enabled=bool((d.get("distributed") or {}).get("enabled", False)),
                strategy=str((d.get("distributed") or {}).get("strategy", "none")),
                world_size=int((d.get("distributed") or {}).get("world_size", 1)),
            ),
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
            # world_size는 유효 배치를 바꾼다 — 결과가 달라지는 값은 정규 표현에 들어가야 한다.
            # 빠져 있으면 장치 수만 바꾼 다른 학습이 같은 해시를 쓴다.
            "distributed": {
                "enabled": self.distributed.enabled,
                "strategy": self.distributed.strategy,
                "world_size": self.distributed.world_size,
            },
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
