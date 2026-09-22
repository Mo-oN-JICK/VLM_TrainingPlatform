"""플러그인 공통 규약.

ExpertPlugin / BackboneAdapter / StorageBackend가 같은 등록 메커니즘을 공유한다.
새 플러그인 종류를 위해 새 메커니즘을 발명하지 않는다. 설계 문서 05 §5.7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from ..core.errors import RegistrationError
from ..core.types import PortType


@dataclass(frozen=True)
class ResourceCost:
    vram_mb: int = 0
    ms_per_sample: float = 0.0
    out_bytes_per_sample: int = 0


@dataclass(frozen=True)
class ExpertManifest:
    id: str
    version: str
    domain: str  # image2d | series1d
    accepts: PortType
    produces: PortType
    deterministic: bool = True
    external_call: bool = True
    os_support: Tuple[str, ...] = ("windows", "linux")
    vram_mb: int = 0

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"


class ExpertPlugin:
    """의심 영역/구간을 지목하는 외부 모델의 어댑터.

    이미지 영역 지목과 시계열 구간 지목이 같은 인터페이스로 들어온다.
    도메인 차이는 manifest가 선언하는 타입에서만 드러난다.
    """

    manifest: ExpertManifest

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def propose(self, subject: Any, params: Any, ctx: Any) -> List[Dict[str, Any]]:
        """Region 목록. 각 원소는 {extent, score, label}."""
        raise NotImplementedError

    def cost(self, shape: Optional[Tuple], params: Any) -> ResourceCost:
        return ResourceCost(vram_mb=self.manifest.vram_mb)


@dataclass(frozen=True)
class BackboneSpec:
    """백본의 숫자. 모델 가중치를 로드하지 않고 답해야 한다 — G4가 학습 전에 호출한다."""

    id: str
    params_total: float  # 파라미터 개수(단위: 개)
    params_by_group: Dict[str, float]  # vision_tower / projector / llm / lm_head ...
    n_layers: int
    hidden: int
    intermediate: int
    vocab: int
    tokens_per_tile: int
    max_context: int
    tokenizer_id: str = ""
    # 타일당 토큰이 **고정이 아닌** 모델(Qwen2-VL 같은 동적 해상도)을 위한 격자.
    # patch_px가 0이면 tokens_per_tile이 그대로 답이다.
    patch_px: int = 0
    spatial_merge: int = 1
    os_support: Tuple[str, ...] = ("windows", "linux")
    supports_quantization: Tuple[str, ...] = ("none", "int8", "nf4")
    supports_attn: Tuple[str, ...] = ("sdpa", "eager")

    def tokens_for_tile(self, tile_px: int) -> int:
        """타일 하나가 만드는 비전 토큰 수.

        동적 해상도 모델은 타일 크기에 따라 값이 변한다. 고정값 하나를 들고 있으면
        타일을 키운 순간 예산이 조용히 낙관적으로 기운다 — G4가 막아야 할 바로 그 상황이다.
        """
        if not self.patch_px or tile_px <= 0:
            return self.tokens_per_tile
        grid = tile_px // self.patch_px
        return max(1, (grid // max(1, self.spatial_merge)) ** 2)

    def lora_params(self, r: int, targets: Tuple[str, ...]) -> float:
        """LoRA 어댑터 파라미터 개수. 모듈 모양에서 계산한다."""
        h, m = float(self.hidden), float(self.intermediate)
        per_layer = 0.0
        for t in targets:
            if t in ("q_proj", "k_proj", "v_proj", "o_proj"):
                per_layer += r * (h + h)
            elif t in ("gate_proj", "up_proj"):
                per_layer += r * (h + m)
            elif t == "down_proj":
                per_layer += r * (m + h)
        return per_layer * self.n_layers


class BackboneAdapter:
    """백본 교체를 설정 한 줄로 끝내기 위한 추상 경계.

    Phase 3에서는 spec()만 쓴다. build/collate/save는 Phase 5에서 붙는다.
    """

    spec_data: BackboneSpec

    @classmethod
    def spec(cls) -> BackboneSpec:
        return cls.spec_data

    @classmethod
    def build(cls, cfg: Any, stage: Any) -> Any:
        raise NotImplementedError(
            f"{cls.spec().id}: 이 백본 어댑터는 spec()만 제공한다(예산 계산용). "
            "학습하려면 build/module_groups/collate를 구현하라 — tiny_backbone.py가 참조 구현이다."
        )


_BACKBONES: Dict[str, Type[BackboneAdapter]] = {}


def register_backbone(cls: Type[BackboneAdapter]) -> Type[BackboneAdapter]:
    s = cls.spec()
    if "windows" not in s.os_support:
        raise RegistrationError(f"{s.id}: 기본 실행 프로파일(windows_single_gpu)을 지원하지 않는다")
    _BACKBONES[s.id] = cls
    return cls


def resolve_backbone(backbone_id: str) -> Type[BackboneAdapter]:
    if backbone_id not in _BACKBONES and backbone_id.startswith("hf:"):
        # 실물 HF 백본은 처음 참조될 때 config.json만 읽어 등록한다(가중치는 열지 않는다)
        from . import hf_backbone

        return hf_backbone.register(backbone_id)
    if backbone_id not in _BACKBONES:
        raise RegistrationError(
            f"백본 {backbone_id!r}를 찾을 수 없다 (등록된 것: {sorted(_BACKBONES)})"
        )
    return _BACKBONES[backbone_id]


def all_backbones() -> List[BackboneSpec]:
    return sorted((c.spec() for c in _BACKBONES.values()), key=lambda s: s.id)


_EXPERTS: Dict[Tuple[str, str], Type[ExpertPlugin]] = {}


def register_expert(cls: Type[ExpertPlugin]) -> Type[ExpertPlugin]:
    m = cls.manifest
    if m.domain not in ("image2d", "series1d"):
        raise RegistrationError(f"{m.ref}: domain은 image2d 또는 series1d여야 한다")
    if "windows" not in m.os_support:
        raise RegistrationError(
            f"{m.ref}: 기본 실행 프로파일(windows_single_gpu)을 지원하지 않는다"
        )
    _EXPERTS[(m.id, m.version)] = cls
    return cls


def resolve_expert(ref: str) -> Type[ExpertPlugin]:
    if "@" not in ref:
        raise RegistrationError(f"전문가 플러그인 참조는 id@semver로 핀 고정해야 한다: {ref!r}")
    key = tuple(ref.split("@", 1))
    if key not in _EXPERTS:
        have = sorted(f"{i}@{v}" for i, v in _EXPERTS)
        raise RegistrationError(f"전문가 플러그인 {ref}를 찾을 수 없다 (등록된 것: {have})")
    return _EXPERTS[key]  # type: ignore[index]


def all_experts() -> List[ExpertManifest]:
    return sorted((c.manifest for c in _EXPERTS.values()), key=lambda m: m.ref)
