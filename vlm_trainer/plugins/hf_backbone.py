"""HuggingFace 백본 어댑터 — 실물 VLM을 같은 인터페이스로 들여온다.

`backbone: hf:<모델 경로 또는 id>` 한 줄이면 그래프도 스펙도 그대로 둔 채 백본이 바뀐다.

핵심 제약 하나: **`spec()`은 가중치를 로드하지 않는다.** `config.json`만 읽어 파라미터 수와
형상을 답한다. 예산 게이트(G4)가 학습 시작 전에 호출하기 때문이고, 그 시점에 5GB를 올리면
게이트의 의미가 사라진다.

`build`/`collate`는 transformers가 있을 때만 동작한다. 실물 모델로 아직 검증되지 않았다 —
`tiny_backbone.py`가 검증된 참조 구현이고, 이 파일은 같은 계약의 HF 판이다.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from ..core.errors import RegistrationError
from .base import GRAPH_PLACEHOLDER, BackboneAdapter, BackboneSpec, register_backbone

PREFIX = "hf:"

# 모델 계열마다 이름이 다르다. 덩이를 가를 때 쓰는 낱말들.
_VISION_WORDS = ("vision", "visual", "image_encoder", "image_tower")
_PROJ_WORDS = ("proj", "connector", "merger", "mm_", "multi_modal", "adapter")

# config.json의 키 이름은 모델 계열마다 다르다. 흔한 것부터 찾는다.
_HIDDEN = ("hidden_size", "d_model", "n_embd")
_LAYERS = ("num_hidden_layers", "n_layer", "num_layers")
_INTER = ("intermediate_size", "ffn_dim", "n_inner")
_VOCAB = ("vocab_size",)
_CTX = ("max_position_embeddings", "max_sequence_length", "n_positions")
# 비전 타워의 깊이·폭. Qwen2-VL 은 `depth` / `embed_dim` 을 쓰고 CLIP 계열은 흔한 이름을 쓴다.
# `vision_config.hidden_size` 는 **타워의 폭이 아니라 LLM 으로 넘기는 폭**인 경우가 있어
# (Qwen2-VL 이 그렇다) `embed_dim` 을 먼저 본다.
_V_LAYERS = ("depth", "num_hidden_layers", "n_layer", "num_layers")
_V_HIDDEN = ("embed_dim", "hidden_size", "d_model")


def _pick(d: Dict[str, Any], keys: Tuple[str, ...], default: int = 0) -> int:
    for k in keys:
        if isinstance(d.get(k), int):
            return int(d[k])
    return default


def _as_pil(images: Any) -> List[Any]:
    """넘파이 배열이든 PIL 이든 프로세서가 받는 형태로. 빈 것은 빈 목록."""
    from PIL import Image

    out = []
    for a in images or []:
        out.append(a if hasattr(a, "size") and not hasattr(a, "shape")
                   else Image.fromarray(__import__("numpy").asarray(a, dtype="uint8")))
    return out


def fetch(ref: str) -> str:
    """가중치까지 내려받는다. 받은 자리를 돌려준다.

    `find_config`은 캐시를 뒤지기만 하고 **받지는 않는다** — 예산 게이트가 부르는 길이라
    그 자리에서 몇 GB가 떨어지면 안 된다. 받는 것은 사람이 시킬 때만 하는 별도의 일이고,
    그래서 이 함수가 따로 있다.
    """
    from huggingface_hub import snapshot_download

    model_id = ref[len(PREFIX) :] if ref.startswith(PREFIX) else ref
    if os.path.isdir(model_id):
        return model_id
    return snapshot_download(model_id)


def find_config(ref: str) -> str:
    """모델 참조를 config.json 경로로 해소한다. 없으면 무엇을 해야 하는지 말한다."""
    ref = ref[len(PREFIX) :] if ref.startswith(PREFIX) else ref

    if os.path.isdir(ref) and os.path.exists(os.path.join(ref, "config.json")):
        return os.path.join(ref, "config.json")
    if os.path.isfile(ref) and ref.endswith(".json"):
        return ref

    # HF 캐시에서 찾는다 (다운로드는 하지 않는다)
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    cache = os.path.join(home, "hub", "models--" + ref.replace("/", "--"), "snapshots")
    if os.path.isdir(cache):
        for snap in sorted(os.listdir(cache)):
            cand = os.path.join(cache, snap, "config.json")
            if os.path.exists(cand):
                return cand

    raise RegistrationError(
        f"백본 {ref!r}의 config.json을 찾을 수 없다.\n"
        f"  찾아본 곳: 로컬 디렉터리, {cache}\n"
        "  먼저 모델을 내려받아라:\n"
        f'    huggingface-cli download {ref}\n'
        "  또는 로컬 경로를 직접 지정하라: backbone: hf:D:/models/<dir>"
    )


def _vision_grid(cfg: Dict[str, Any]) -> Tuple[int, int]:
    """(패치 픽셀, spatial merge). 없으면 (0, 1) — 타일당 토큰이 고정이라는 뜻이다."""
    v = cfg.get("vision_config") or cfg.get("vision_tower_config") or {}
    patch = v.get("patch_size") or v.get("spatial_patch_size")
    if not isinstance(patch, int) or patch <= 0:
        return 0, 1
    merge = v.get("spatial_merge_size") or cfg.get("spatial_merge_size") or 1
    return int(patch), max(1, int(merge))


def _vision_tokens(cfg: Dict[str, Any]) -> int:
    """타일 하나가 만드는 비전 토큰 수. 모델 계열마다 이름이 다르다.

    config가 타일 크기를 말하지 않는 모델(동적 해상도)이 있다. 그때는 여기서 고정값을
    지어내지 않고 `patch_px`를 함께 실어 보내, 예산이 **선언한 타일 크기로** 계산하게 한다.
    """
    v = cfg.get("vision_config") or cfg.get("vision_tower_config") or {}
    if isinstance(v.get("num_image_tokens"), int):
        return int(v["num_image_tokens"])
    img, patch = v.get("image_size"), v.get("patch_size")
    if isinstance(img, int) and isinstance(patch, int) and patch:
        grid = (img // patch) ** 2
        merge = v.get("spatial_merge_size") or cfg.get("spatial_merge_size") or 1
        return max(1, grid // (int(merge) ** 2))
    return 256  # 격자를 알면 예산이 이 값을 쓰지 않는다


def _params_from_config(cfg: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """config만으로 파라미터 수를 추정한다.

    가중치를 열지 않으므로 근사다. 예산은 보수적으로 잡아야 하므로 올림 쪽으로 센다.
    """
    text = cfg.get("text_config") or cfg
    h = _pick(text, _HIDDEN, 2048)
    n = _pick(text, _LAYERS, 24)
    inter = _pick(text, _INTER, h * 4)
    vocab = _pick(text, _VOCAB, 32000)

    attn = 4 * h * h
    mlp = 3 * h * inter
    # tie_word_embeddings면 lm_head가 임베딩과 같은 행렬이다. 두 번 세면
    # 2B 모델에서 0.23B가 허공에서 생긴다 — 들어갈 학습을 막는 쪽으로 틀린다.
    embed_copies = 1 if cfg.get("tie_word_embeddings") or text.get("tie_word_embeddings") else 2
    llm = n * (attn + mlp) + embed_copies * vocab * h

    v = cfg.get("vision_config") or {}
    vh = _pick(v, _HIDDEN, 1024)
    vn = _pick(v, _LAYERS, 24)
    vision = vn * (4 * vh * vh + 3 * vh * vh * 4) if vh else 0.0
    projector = 2.0 * vh * h if vh else 0.0

    return float(llm + vision + projector), {
        "llm": float(llm),
        "vision_tower": float(vision),
        "projector": float(projector),
    }


def spec_from_config(path: str, model_id: str) -> BackboneSpec:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    text = cfg.get("text_config") or cfg
    total, groups = _params_from_config(cfg)
    return BackboneSpec(
        id=model_id,
        params_total=total,
        params_by_group=groups,
        n_layers=_pick(text, _LAYERS, 24),
        hidden=_pick(text, _HIDDEN, 2048),
        intermediate=_pick(text, _INTER, _pick(text, _HIDDEN, 2048) * 4),
        vocab=_pick(text, _VOCAB, 32000),
        tokens_per_tile=_vision_tokens(cfg),
        max_context=_pick(text, _CTX, 4096),
        # 토크나이저 이름은 AutoTokenizer가 그대로 받을 수 있어야 한다.
        # 우리 접두어(`hf:`)는 백본 id의 것이지 모델 이름의 일부가 아니다.
        tokenizer_id=model_id[len(PREFIX):] if model_id.startswith(PREFIX) else model_id,
        patch_px=_vision_grid(cfg)[0],
        spatial_merge=_vision_grid(cfg)[1],
        vision_layers=_pick(cfg.get("vision_config") or {}, _V_LAYERS),
        vision_hidden=_pick(cfg.get("vision_config") or {}, _V_HIDDEN),
        supports_quantization=("none", "int8", "nf4"),
        supports_attn=("sdpa", "eager"),
    )


class HFBackbone(BackboneAdapter):
    """모델 하나에 대응하는 어댑터. `register(ref)`가 클래스를 만들어 등록한다."""

    model_ref: str = ""
    config_path: str = ""

    @classmethod
    def spec(cls) -> BackboneSpec:
        return cls.spec_data

    # ── 학습 (transformers 필요, 실물 모델로 아직 미검증) ───────────────
    @classmethod
    def build(cls, cfg: Any, stage: Any) -> Any:
        try:
            import torch

            # transformers 5 에서 `AutoModelForVision2Seq` 가 사라지고 이 이름이 됐다.
            # 둘 다 받아 둔다 — 4090 과 이 PC 의 버전이 갈릴 수 있다.
            try:
                from transformers import AutoModelForImageTextToText as AutoVLM
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoVLM
        except ImportError as e:
            raise RegistrationError(
                "실물 백본을 쓰려면 transformers가 필요하다:\n"
                "  python -m pip install transformers accelerate safetensors\n"
                f"  ({e})"
            ) from None

        kw: Dict[str, Any] = {"dtype": getattr(torch, cfg.dtype, torch.bfloat16)}
        if cfg.quantization.mode in ("nf4", "int8"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise RegistrationError(
                    "양자화에는 bitsandbytes가 필요하다: python -m pip install bitsandbytes"
                ) from None
            kw["quantization_config"] = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=cfg.quantization.double_quant,
                    bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.compute_dtype, torch.bfloat16),
                )
                if cfg.quantization.mode == "nf4"
                else BitsAndBytesConfig(load_in_8bit=True)
            )
        model = AutoVLM.from_pretrained(
            os.path.dirname(cls.config_path), attn_implementation=cfg.attn_impl, **kw
        )
        return model

    @classmethod
    def module_groups(cls, model: Any) -> Dict[str, List[Any]]:
        """모델을 학습 정책이 말하는 세 덩이로 가른다.

        **실물 모델은 이름이 평평하지 않다.** Qwen2-VL 은 최상위가 `model` 과 `lm_head`
        뿐이고, 비전 타워와 LLM 은 `model` 안에 있다. 최상위만 훑으면 `vision_tower` 와
        `projector` 가 **빈 목록**이 되고, `projector: true` 라고 선언한 단계가
        "학습할 파라미터가 없다" 로 죽는다. 그나마 죽어 주는 쪽이 다행이다.

        프로젝터가 비전 타워 **안에** 있는 모델도 있다(Qwen2-VL 의 `visual.merger`).
        그것을 비전 타워에 함께 담으면 `vision_tower: false` 와 `projector: true` 가
        서로를 부정하고, freeze 검증이 "false 로 선언했는데 학습 대상이 있다" 를 낸다.
        그래서 갈라 담는다.
        """
        groups: Dict[str, List[Any]] = {"vision_tower": [], "projector": [], "llm": []}
        for name, mod in cls._parts(model):
            low = name.lower().rsplit(".", 1)[-1]
            if any(w in low for w in _VISION_WORDS):
                proj, rest = cls._split_projector(mod)
                groups["projector"] += proj
                groups["vision_tower"] += rest
            elif any(w in low for w in _PROJ_WORDS):
                groups["projector"].append(mod)
            else:
                groups["llm"].append(mod)
        return groups

    @classmethod
    def _parts(cls, model: Any) -> List[Any]:
        """훑을 단위. 비전 타워를 품고만 있는 껍데기는 한 겹 열고 들어간다.

        무턱대고 깊이 내려가지 않는다 — 열어서 비전 타워가 보일 때만 연다. 아니면
        LLM 의 레이어 스물여덟 개가 통째로 최상위 덩이가 되어 쏟아진다.
        """
        out: List[Any] = []
        for name, mod in model.named_children():
            inner = list(getattr(mod, "named_children", list)())
            if inner and any(any(w in n.lower() for w in _VISION_WORDS) for n, _ in inner):
                out += [(f"{name}.{n}", m) for n, m in inner]
            else:
                out.append((name, mod))
        return out

    @classmethod
    def _split_projector(cls, tower: Any) -> Any:
        """비전 타워에서 프로젝터를 떼어 낸다. 없으면 타워가 통째로 남는다."""
        kids = list(getattr(tower, "named_children", list)())
        proj = [m for n, m in kids if any(w in n.lower() for w in _PROJ_WORDS)]
        if not proj:
            return [], [tower]
        return proj, [m for n, m in kids if not any(w in n.lower() for w in _PROJ_WORDS)]

    @classmethod
    def lora_root(cls, model: Any) -> Any:
        groups = cls.module_groups(model)
        return groups["llm"][0] if groups["llm"] else model

    @classmethod
    def processor(cls) -> Any:
        # 배치마다 불린다. 디스크에서 매번 다시 읽으면 학습보다 이쪽이 오래 걸린다.
        proc = cls.__dict__.get("_proc")
        if proc is None:
            from transformers import AutoProcessor

            proc = AutoProcessor.from_pretrained(os.path.dirname(cls.config_path))
            cls._proc = proc
        return proc

    export_format = "huggingface"      # from_pretrained 로 그냥 열린다

    @classmethod
    def save(cls, model: Any, out_dir: str) -> Any:
        """HuggingFace 형식으로 쓴다. 프로세서도 함께 — 토크나이저와 이미지 전처리를
        모르면 가중치만 있어도 첫 입력조차 못 만든다."""
        model.save_pretrained(out_dir)
        cls.processor().save_pretrained(out_dir)
        return sorted(f for f in os.listdir(out_dir) if not f.startswith("."))

    @classmethod
    def _to_model(cls, model: Any, enc: Any) -> Dict[str, Any]:
        """배치를 모델이 있는 장치와 **자료형**으로 옮긴다.

        프로세서는 픽셀을 float32 로 낸다. 모델이 bf16 이면 비전 타워의 첫 행렬곱에서
        `expected mat1 and mat2 to have the same dtype` 로 죽는다. 정수 텐서
        (input_ids · labels · attention_mask)는 건드리지 않는다 — 실수로 바꾸면
        토큰 id 가 반올림된다.
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        out: Dict[str, Any] = {}
        for k, v in enc.items():
            if hasattr(v, "to"):
                v = v.to(device)
                if getattr(v, "is_floating_point", bool)():
                    v = v.to(dtype)
            out[k] = v
        return out

    @classmethod
    def set_grad_checkpointing(cls, model: Any, on: bool) -> bool:
        """HF 는 `use_cache` 를 **함께 꺼야 한다.** 켜 둔 채로 재계산을 켜면
        transformers 가 경고 한 줄을 내고 재계산을 무시한다 — 예산이 깎은 메모리가
        실제로는 안 깎인다."""
        if on and hasattr(model, "config"):
            model.config.use_cache = False
        return super().set_grad_checkpointing(model, on)

    @classmethod
    def forward(cls, model: Any, batch: Any, device: Any) -> Any:
        return model(**cls._to_model(model, batch))

    @classmethod
    def image_placeholder(cls) -> str:
        """이 백본이 "여기에 이미지가 들어간다" 를 적는 방식.

        **프로세서에서 끌어낸다. 손으로 적지 않는다.** 모델마다 다르고, 틀려도 조용히
        지나가기 때문이다 — Qwen2-VL 에 `<image>` 를 주면 예외 없이 통과하지만 vocab 에
        없는 글자라 평범한 바이트로 쪼개지고, 이미지 토큰이 하나도 안 생긴 채 학습이 돈다.
        손실은 내려가는데 모델은 이미지를 보지 않는다. 실측으로 확인한 함정이다.
        """
        ph = cls.__dict__.get("_ph")
        if ph is not None:
            return ph
        proc = cls.processor()
        tok = proc.tokenizer
        img = getattr(proc, "image_token", None)
        if not img:
            cls._ph = GRAPH_PLACEHOLDER
            return cls._ph
        # 비전 경계 토큰이 vocab 에 있으면 감싼다. Qwen2-VL 의 채팅 템플릿이 그렇게 쓴다.
        start, end = "<|vision_start|>", "<|vision_end|>"
        known = tok.convert_tokens_to_ids([start, end])
        cls._ph = (f"{start}{img}{end}"
                   if all(i is not None and i >= 0 for i in known) else str(img))
        return cls._ph

    @classmethod
    def _retarget(cls, text: str) -> str:
        """그래프가 적은 자리표시자를 이 백본의 것으로 바꾼다.

        그래프는 계속 `<image>` 만 안다. 무슨 토큰으로 적을지는 백본의 사정이고,
        그래야 `backbone:` 한 줄로 모델이 바뀐다.
        """
        ph = cls.image_placeholder()
        return text if ph == GRAPH_PLACEHOLDER else text.replace(GRAPH_PLACEHOLDER, ph)

    @classmethod
    def generate(cls, model: Any, prompt: str, images: Any, max_new: int = 64) -> str:
        """탐욕적 디코딩. 답이 형식을 지키는지 보는 것이 목적이라 무작위성을 끈다."""
        import torch

        proc = cls.processor()
        pil = _as_pil(images)
        enc = proc(text=[cls._retarget(prompt)], images=[pil] if pil else None,
                   return_tensors="pt")
        enc = cls._to_model(model, enc)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        # 프롬프트 구간을 잘라내고 답만 돌려준다
        start = enc["input_ids"].shape[1]
        return proc.tokenizer.decode(ids[0][start:], skip_special_tokens=True)

    @classmethod
    def collate(cls, batch: List[Dict[str, Any]], max_len: int) -> Dict[str, Any]:
        """프로세서로 이미지와 텍스트를 함께 묶고, 손실은 **정답 토큰에만** 건다."""
        proc = cls.processor()
        texts = [cls._retarget(f"{r['prompt']}\n{r['answer']}") for r in batch]
        images = [r.get("_images") or [] for r in batch]
        enc = proc(
            text=texts,
            images=images if any(images) else None,
            return_tensors="pt",
            padding=True,
            truncation=False,  # 잘림은 G4가 막는다. 여기서 조용히 자르지 않는다
        )
        enc["labels"] = cls._answer_only_labels(batch, enc)
        return dict(enc)

    @classmethod
    def _answer_only_labels(cls, batch: List[Dict[str, Any]], enc: Any) -> Any:
        """정답 구간만 남기고 전부 -100.

        두 가지를 직접 셈해야 한다. **둘 다 틀려도 학습은 돌고 손실은 내려간다.**

        하나. 프롬프트 길이는 맨 토크나이저로 세면 안 된다. 프로세서가 이미지 자리표시자
        하나를 타일 토큰 수백 개로 부풀리기 때문이다(실측: 31 -> 286). 그 차이만큼
        **이미지 토큰에 손실이 걸린다** — 모델에게 자기가 본 그림을 받아쓰라고 시키는 꼴이다.
        부풀린 양은 실제로 붙은 이미지 토큰 수에서 자리표시자 수를 빼면 정확히 나온다.
        이미지를 한 번 더 전처리할 필요가 없다.

        둘. 이 토크나이저는 **왼쪽으로 패딩한다.** 앞에서부터 가리면 프롬프트가 아니라
        패딩을 가리게 되고, 정답은 그대로인 채 프롬프트 전체에 손실이 걸린다.
        어디서 시작하는지는 `attention_mask` 가 안다 — 패딩 방향을 추측하지 않는다.
        """
        proc = cls.processor()
        ids, mask = enc["input_ids"], enc["attention_mask"]
        labels = ids.new_full(ids.shape, -100)

        img_tok = getattr(proc, "image_token", None)
        img_id = proc.tokenizer.convert_tokens_to_ids(img_tok) if img_tok else -1
        ph = cls.image_placeholder()

        for i, r in enumerate(batch):
            prompt = cls._retarget(r["prompt"])
            n = len(proc.tokenizer(prompt)["input_ids"])
            if img_id is not None and img_id >= 0:
                # 자리표시자 하나가 토큰 여럿으로 부푼 만큼을 더한다
                n += int((ids[i] == img_id).sum()) - prompt.count(ph)
            keep = mask[i].nonzero()
            off = int(keep[0]) if len(keep) else 0      # 패딩이 어디서 끝나는지
            end = off + int(mask[i].sum())
            start = min(off + n, end)
            labels[i, start:end] = ids[i, start:end]
        return labels


def register(ref: str) -> type:
    """`hf:<경로 또는 id>`를 레지스트리에 올린다. config.json만 읽는다."""
    model_id = ref if ref.startswith(PREFIX) else PREFIX + ref
    path = find_config(ref)
    cls = type(
        "HFBackbone_" + model_id.replace("/", "_").replace(":", "_").replace(".", "_"),
        (HFBackbone,),
        {"model_ref": ref, "config_path": path, "spec_data": spec_from_config(path, model_id)},
    )
    return register_backbone(cls)
