"""플러그인 공통 규약.

ExpertPlugin / BackboneAdapter / StorageBackend가 같은 등록 메커니즘을 공유한다.
새 플러그인 종류를 위해 새 메커니즘을 발명하지 않는다. 설계 문서 05 §5.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

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
