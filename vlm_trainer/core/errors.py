"""게이트 에러.

에러 메시지는 항상 네 가지를 담는다.
  (1) 불일치 필드  (2) 안 잡혔다면 언제 어디서 어떻게 드러났을지
  (3) 추정 낭비 시간  (4) 구체적 해결 배선
검증의 가치를 사용자가 매번 체감해야 게이트가 유지된다. 설계 문서 02 §2.7.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


class VlmtError(Exception):
    """플랫폼 최상위 예외."""


class GateError(VlmtError):
    gate = "G?"

    def __init__(self, message: str, *, node: str = "", port: str = "") -> None:
        super().__init__(message)
        self.node = node
        self.port = port


class RegistrationError(VlmtError):
    """노드 등록 시점 계약 위반. 임포트가 실패해야 한다."""


class SpecError(GateError):
    gate = "G2"


class StructureError(GateError):
    gate = "G2"


class PolicyError(GateError):
    gate = "G2"


class UnresolvedTypeError(GateError):
    gate = "G2"


class BudgetError(GateError):
    gate = "G4"


class DryRunError(GateError):
    gate = "G3"


# 필드별 "이 배선이 통과했다면 어떻게 되었을지".
# (증상, 언제 드러나는지, 추정 낭비, 해결 힌트)
_CONSEQUENCE = {
    "base": (
        "다른 종류의 값이 흘러 첫 샘플에서 AttributeError로 즉사합니다.",
        "실행 직후",
        "세팅 시간",
        None,
    ),
    "dtype": (
        "데이터로더 collate에서 첫 배치가 조립되지 않고 크래시합니다.",
        "학습 시작 직후",
        "물질화 시간 전부",
        "adapt.image_dtype",
    ),
    "layout": (
        "텐서 축이 뒤바뀐 채 백본에 들어가 shape 오류로 크래시하거나, "
        "운이 나쁘면 정사각 입력에서 조용히 통과합니다.",
        "학습 시작 직후 또는 영원히",
        "물질화 시간 + 학습 전체",
        "adapt.image_layout",
    ),
    "value_range": (
        "아무것도 터지지 않습니다. 손실은 정상적으로 감소하고, "
        "학습 완주 후 성능 저하로만 드러납니다. 원인 추적은 사실상 불가능합니다.",
        "드러나지 않음",
        "물질화 + 학습 전체",
        "adapt.image_norm / adapt.image_denorm",
    ),
    "norm": (
        "이미 정규화된 값을 한 번 더 정규화합니다. 손실은 정상적으로 감소하며 "
        "성능만 낮게 나옵니다.",
        "드러나지 않음",
        "물질화 + 학습 전체",
        "adapt.image_denorm",
    ),
    "colorspace": (
        "채널 순서가 뒤바뀐 이미지를 학습합니다. 사람 눈에는 색이 이상하지만 "
        "손실 곡선은 정상으로 보입니다.",
        "드러나지 않음",
        "학습 전체",
        "adapt.colorspace",
    ),
    "frame": (
        "좌표 기준이 다른 박스로 crop합니다. 전 샘플이 엉뚱한 영역을 잘라내고 "
        "학습은 끝까지 완주합니다.",
        "드러나지 않음",
        "물질화 + 학습 전체",
        "adapt.frame",
    ),
    "time_base": (
        "시간축 기준이 어긋난 구간을 잘라냅니다. 전문가가 지목한 위치와 실제 신호가 "
        "밀린 채 라벨이 오염됩니다.",
        "드러나지 않음",
        "물질화 + 학습 전체",
        "adapt.ts_time_base",
    ),
    "semantic": (
        "정답 또는 라벨이 프롬프트 경로로 흘러듭니다. 검증 점수가 비현실적으로 높게 "
        "나오고 실험 전체가 무효가 되지만 아무도 눈치채지 못합니다.",
        "드러나지 않음",
        "실험 전체",
        "answer.leakage_guard",
    ),
    "list": (
        "리스트와 단일 값이 섞입니다. 첫 샘플에서 TypeError가 나거나, "
        "리스트의 첫 원소만 조용히 사용됩니다.",
        "실행 직후 또는 드러나지 않음",
        "물질화 시간",
        "list.wrap / list.map",
    ),
    "optional": (
        "값이 없는 샘플에서 None이 하류로 흘러 NoneType 오류가 납니다.",
        "일부 샘플에서",
        "물질화 시간",
        "optional.unwrap_or",
    ),
    "shape": (
        "배치 조립이 실패하거나, 브로드캐스트가 조용히 성립해 의도하지 않은 "
        "차원으로 학습됩니다.",
        "학습 시작 직후 또는 드러나지 않음",
        "물질화 + 학습 전체",
        None,
    ),
}

_SILENT_FIELDS = {"value_range", "norm", "frame", "time_base", "semantic", "colorspace"}


class TypeMismatchError(GateError):
    """G1 — 편집 시점 정적 검사. 타입이 맞지 않는 배선은 연결 자체가 되지 않는다."""

    gate = "G1"

    def __init__(
        self,
        src_ref: str,
        dst_ref: str,
        src_type,
        dst_type,
        mismatches: Sequence,
        *,
        gate: str = "G1",
    ) -> None:
        self.gate = gate
        self.src_ref, self.dst_ref = src_ref, dst_ref
        self.src_type, self.dst_type = src_type, dst_type
        self.mismatches = list(mismatches)
        super().__init__(self.render(), node=dst_ref.split(":")[0], port=dst_ref)

    # ── 렌더링 ──────────────────────────────────────────────────────────
    def _worst(self):
        """가장 조용한(= 가장 비싼) 불일치를 대표로 고른다."""
        for m in self.mismatches:
            root = m.field.split(".")[0].split("[")[0]
            if root in _SILENT_FIELDS:
                return root
        for m in self.mismatches:
            root = m.field.split(".")[0].split("[")[0]
            if root in _CONSEQUENCE:
                return root
        return None

    def render(self) -> str:
        lines = [
            f"TypeError [{self.gate}] {self.dst_ref}  <-  {self.src_ref}",
            "",
            f"  기대: {self.dst_type}",
            f"  실제: {self.src_type}",
            "  불일치 필드: " + ", ".join(str(m) for m in self.mismatches),
        ]
        key = self._worst()
        if key:
            symptom, when, waste, fix = _CONSEQUENCE[key]
            lines += [
                "",
                "  이 배선이 통과했다면:",
                "    " + symptom,
                f"    드러나는 시점: {when}",
                f"    추정 낭비: {waste}",
            ]
            if fix:
                lines += ["", f"  해결: {fix} 어댑터 노드를 사이에 넣으십시오."]
        return "\n".join(lines)


def render_gate_report(errors: Iterable[GateError]) -> str:
    errs = list(errors)
    if not errs:
        return "게이트 통과."
    head = f"{len(errs)}건의 게이트 위반:"
    return "\n\n".join([head] + [str(e) for e in errs])
