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
from ..core.types import BaseKind, simple, text

# (모델 디렉터리, 백본 id) -> 올려 둔 모델. 프로세스가 사는 동안 유지된다.
_LOADED: Dict[Tuple[str, str], Any] = {}


def _load(model_dir: str, backbone: str) -> Any:
    key = (os.path.abspath(model_dir), backbone)
    if key in _LOADED:
        return _LOADED[key]

    import torch

    from ..plugins.base import resolve_backbone

    adapter = resolve_backbone(backbone)
    if not hasattr(adapter, "generate"):
        raise NodeError(
            cause=f"백본 {backbone!r} 은 답을 생성할 줄 모른다",
            hint=(
                "  안 잡혔다면: 학습은 끝났는데 추론에서 빈 답이 나온다.\n"
                "  추정 낭비: 학습 시간 전부.\n"
                "  어댑터에 generate(model, prompt, images, max_new) 를 구현하라 — "
                "tiny_backbone.py 가 참조 구현이다."
            ),
        )

    # 마지막 단계의 체크포인트를 쓴다. 단계 이름을 모르므로 가장 최근 것을 고른다.
    stages = [d for d in sorted(os.listdir(model_dir))
              if os.path.isdir(os.path.join(model_dir, d))]
    ckpts = [os.path.join(model_dir, d, f)
             for d in stages
             for f in sorted(os.listdir(os.path.join(model_dir, d)))
             if f.endswith(".pt")]
    if not ckpts:
        raise NodeError(
            cause=f"{model_dir} 에 체크포인트가 없다",
            hint=(
                "  안 잡혔다면: 학습되지 않은 모델이 답을 내고, 그 답을 보고 판단하게 된다.\n"
                "  추정 낭비: 없음(시작하지 않았다).\n"
                "  먼저 학습을 돌려라."
            ),
        )
    latest = max(ckpts, key=os.path.getmtime)

    model = adapter.build(None, None)
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    if torch.cuda.is_available():
        model = model.cuda()
    _LOADED[key] = (adapter, model)
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

        answer = str(inputs["answer"] or "")
        if not answer:
            return {}      # 추론이 건너뛴 샘플(학습 split)은 남기지 않는다
        out_dir = os.path.abspath(params.out_dir.replace("{run_id}", ctx.run_id))
        os.makedirs(out_dir, exist_ok=True)
        rec = {
            "id": ctx.sample_key,
            "answer": answer,
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
                cause=f"학습된 모델을 가리키는 값이 아니다: {handle!r}",
                hint="  `모델 학습` 상자의 출력을 이 상자의 `model` 에 이어라.",
            )

        # 학습에 쓴 샘플에 답하게 하면 외운 것을 보게 된다. 검증 쪽만 본다.
        want = (params.only_split or "").strip()
        if want and str(ctx.sample.get("_split", "")) != want:
            return {"answer": ""}

        adapter, model = _load(handle["dir"], handle["backbone"])
        images = inputs.get("images") or []
        return {
            "answer": adapter.generate(
                model, str(inputs["prompt"]), images, params.max_new_tokens
            )
        }
