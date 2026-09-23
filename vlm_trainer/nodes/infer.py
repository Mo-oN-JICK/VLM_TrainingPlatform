"""학습된 모델로 답을 만드는 노드.

**`PROCESSING` + `external_call`** 이다. 모델 파일을 디스크에서 읽으므로 순수 함수가 아니고,
그 예외를 선언으로 드러낸다 — `expert.propose` 가 같은 문제를 이미 그렇게 풀었다.

물질화 경계 규칙은 이 노드에 걸리지 않는다. 그 규칙이 막는 위험은 "학습 루프 안에서
외부 모델이 VRAM 을 뺏는 것" 인데, 이 노드는 학습이 **끝난 뒤** 그 산출물을 받아 돈다.
컴파일러가 학습보다 뒤에 있는 노드를 검사에서 빼는 이유다.

모델은 한 번만 올린다. 샘플마다 다시 올리면 10장에 10번 로드한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..core.node import Node, NodeDoc, NodeError, NodeKind, Port, RunCtx
from ..core.registry import register
from ..core.types import ANY, BaseKind, image, simple, text

# (모델 디렉터리, 백본 id) -> 올려 둔 모델. 프로세스가 사는 동안 유지된다.
_LOADED: Dict[Tuple[str, str], Any] = {}


def _load(node_id: str, model_dir: str, backbone: str) -> Any:
    """모델을 한 번만 올린다. 샘플마다 다시 올리면 10장에 10번 로드한다.

    **올리는 일 자체는 `train/checkpoint.py` 가 한다.** Export 도 같은 코드로 올린다 —
    둘이 갈리면 "추론에서는 잘 나왔는데 내보낸 모델은 이상하다" 가 되고, 그때 어느 쪽이
    학습한 모델인지 아무도 모른다.
    """
    key = (os.path.abspath(model_dir), backbone)
    if key in _LOADED:
        return _LOADED[key]

    from ..plugins.base import resolve_backbone
    from ..train import checkpoint as ckpt_mod

    if not hasattr(resolve_backbone(backbone), "generate"):
        raise NodeError(
            node_id,
            cause=f"백본 {backbone!r} 은 답을 생성할 줄 모른다",
            hint=(
                "  안 잡혔다면: 학습은 끝났는데 추론에서 빈 답이 나온다.\n"
                "  추정 낭비: 학습 시간 전부.\n"
                "  어댑터에 generate(model, prompt, images, max_new) 를 구현하라 — "
                "tiny_backbone.py 가 참조 구현이다."
            ),
        )
    try:
        _LOADED[key] = ckpt_mod.load(model_dir, backbone, to_cuda=True)
    except ckpt_mod.CheckpointError as e:
        raise NodeError(node_id, cause=e.cause, hint=e.hint) from None
    return _LOADED[key]


@dataclass
class AnswerReportParams:
    out_dir: str = "runs/{run_id}/infer"


@register(
    type="io.answer_report",
    version="1.0.0",
    category="File",
    kind=NodeKind.OUTPUT,
    inputs={
        "answer": Port(text(), "모델이 내놓은 답"),
        "expected": Port(text().as_optional(), "정답 (있으면 나란히 적는다)"),
    },
    params=AnswerReportParams,
    recipe_overridable=["out_dir"],
    preview="text",
    doc=NodeDoc(
        label="답변 결과 저장",
        hint="모델의 답을 정답과 나란히 파일로 남깁니다.",
        summary="채점하지 않는다. 사람이 읽고 판단하도록 나란히 적을 뿐이다.",
        scenario=(
            "자동 채점 숫자가 나오면 그 숫자를 믿게 되는데, 라벨에 잡음이 있으면 "
            "믿을 값이 아니다. 데이터가 정리된 뒤에 채점을 붙인다."
        ),
    ),
)
class AnswerReport(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        import json

        # **빈 답도 적는다.** 버리면 `n_infer 10건 성공` 인데 파일에는 8줄이 남고,
        # 모델이 아무 말도 안 한 두 건이 세상에서 사라진다. 그 침묵이 결과다.
        out_dir = os.path.abspath(params.out_dir.replace("{run_id}", ctx.run_id))
        os.makedirs(out_dir, exist_ok=True)
        rec = {
            "id": ctx.sample_key,
            "split": str(ctx.sample.get("_split", "")),
            "answer": str(inputs["answer"] or ""),
            "expected": str(inputs.get("expected") or ""),
        }
        with open(os.path.join(out_dir, "answers.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {}


@dataclass
class InferParams:
    max_new_tokens: int = 64
    only_split: str = "val"   # 이 split 의 샘플에만 답한다. 빈 값이면 전부


@register(
    type="infer.vlm",
    version="1.0.0",
    category="Training",
    kind=NodeKind.PROCESSING,
    external_call=True,     # 모델 파일을 읽는다. 순수 함수가 아님을 드러낸다
    deterministic=False,    # 같은 입력이라도 체크포인트가 바뀌면 답이 바뀐다
    inputs={
        "model": Port(simple(BaseKind.MODEL), "학습된 모델"),
        "prompt": Port(text(), "질문"),
        # 학습 때와 **같은 전처리를 거친** 이미지여야 한다. 원본을 바로 물리면
        # 모델이 한 번도 본 적 없는 크기가 들어가고, 답이 나빠진 이유를 모델 탓으로 돌린다.
        "images": Port(image(frame=ANY).as_list(1, 16).as_optional(), "질문에 딸린 이미지"),
    },
    outputs={"answer": Port(text(), "모델이 내놓은 답")},
    params=InferParams,
    recipe_overridable=["max_new_tokens"],
    preview="text",
    doc=NodeDoc(
        label="모델 추론",
        hint="학습된 모델에 질문을 넣어 답을 받습니다.",
        summary="학습이 끝난 뒤 검증 샘플에 실제로 어떤 답이 나오는지 본다.",
        scenario="손실이 내려간 것만으로는 모델이 쓸 만한지 알 수 없다. 답을 읽어야 안다.",
    ),
)
class VlmInfer(Node):
    def run(self, ctx: RunCtx, params: Any, **inputs: Any) -> Dict[str, Any]:
        handle = inputs["model"]
        if not isinstance(handle, dict) or not handle.get("dir"):
            raise NodeError(
                ctx.node_id,
                cause=f"학습된 모델을 가리키는 값이 아니다: {handle!r}",
                hint="  `모델 학습` 상자의 출력을 이 상자의 `model` 에 이어라.",
            )

        # 학습에 쓴 샘플에 답하게 하면 외운 것을 보게 된다. 검증 쪽만 본다.
        want = (params.only_split or "").strip()
        if want and str(ctx.sample.get("_split", "")) != want:
            return {"answer": ""}

        adapter, model = _load(ctx.node_id, handle["dir"], handle["backbone"])
        images = inputs.get("images") or []
        return {
            "answer": adapter.generate(
                model, str(inputs["prompt"]), images, params.max_new_tokens
            )
        }
