# 7. Trainer 노드의 선언형 설정 스키마

Trainer는 **Output 분류**의 노드이고, 그래프의 종결점이다. 다단계 학습·freeze 정책·조건 분기·반복은 그래프로 표현하지 않고 전부 이 선언형 설정으로 받는다.

기준 실행 환경은 **RTX 4090 한 장(24GB) / Windows PC**이며, 이는 부차 조건이 아니라 스키마 설계의 1급 제약이다.

## 7.1 전체 스키마

```yaml
kind: TrainerConfig
version: 1

backbone:
  id: "qwen2.5-vl-7b"            # 백본 어댑터 플러그인이 해석
  adapter: "hf_qwen2vl@1.1.0"
  dtype: bf16                    # 계산 dtype
  quantization:                  # 없으면 null
    mode: nf4                    # none | int8 | nf4 | fp8
    compute_dtype: bf16
    double_quant: true
    skip_modules: [vision_tower.patch_embed, lm_head]
  attn_impl: sdpa                # sdpa(기본) | eager | flash_attn2
  gradient_checkpointing_impl: reentrant_off

vision:
  max_images_per_sample: 4
  tiling: {enabled: true, tile_px: 448, max_tiles: 6, overlap: 0}
  tokens_per_tile: auto          # 백본 어댑터가 보고. 수동 지정도 가능
  min_pixels: 200704
  max_pixels: 1003520

sequence:
  max_len: 4096
  truncation: forbid             # forbid(기본) | right | left
  pack: false                    # 짧은 샘플 패킹

loss:
  type: causal_lm
  mask_prompt: true              # 프롬프트 토큰은 손실에서 제외
  chunked_ce: {enabled: true, chunk: 1024}   # 로짓 메모리 절감 (7.4 참조)

stages:
  - name: projector_align
    trainable:
      vision_tower: false
      projector: true
      llm: false
      lora: none
    epochs: 1
    optimizer: {type: adamw_torch, lr: 1.0e-3, weight_decay: 0.0, betas: [0.9, 0.999], eps: 1.0e-8}
    scheduler: {type: cosine, warmup_ratio: 0.03, min_lr_ratio: 0.1}
    batch: {per_device: 4, grad_accum: 8}      # 유효 배치 32
    precision: {amp: bf16, grad_checkpointing: false}
    clip_grad_norm: 1.0
    eval: {every_steps: 200, metrics: [loss, schema_conformance]}

  - name: full_ft
    init_from: projector_align                 # 단계 연결
    trainable:
      vision_tower: false
      projector: true
      llm: lora
    lora:
      r: 32
      alpha: 64
      dropout: 0.05
      targets: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
      modules_to_save: [projector]
    epochs: 3
    optimizer: {type: adamw_bnb_8bit, lr: 1.0e-4, weight_decay: 0.01}
    scheduler: {type: cosine, warmup_ratio: 0.03}
    batch: {per_device: 1, grad_accum: 16}
    precision: {amp: bf16, grad_checkpointing: true}
    offload: {optimizer: none, params: none}   # none | cpu | disk
    clip_grad_norm: 1.0

budget:
  device: {name: rtx4090, vram_gb: 24, reserve_gb: 1.5}
  policy: fail_fast              # fail_fast(기본) | warn — warn은 CI 전용, 대화형에서는 무시
  headroom_ratio: 0.10           # 추정치 위 10%를 안전 여유로 요구

runtime:
  profile: windows_single_gpu
  dataloader:
    num_workers: 4
    start_method: spawn          # Windows 강제값 (7.6)
    persistent_workers: true
    prefetch_factor: 2
    pin_memory: true
  seed: 20260909
  deterministic: {torch: true, cudnn_benchmark: false}
  distributed: {enabled: false, strategy: none}   # 확장 지점. 기본 프로파일은 의존하지 않는다

checkpoint:
  dir: "runs/{run_id}/ckpt"
  every_steps: 200
  keep_last: 2
  keep_best: {metric: val_loss, mode: min}
  save_optimizer: true           # 재개용
  save_dataloader_state: true    # 재개용 (문서 8.6)
  atomic_write: true             # temp + os.replace (Windows)

logging:
  every_steps: 10
  sinks: [jsonl, tensorboard]
  debug_output: inherit          # 프로젝트 debug 설정을 따름
```

## 7.2 freeze 정책 표현

`trainable`은 모듈 그룹별 3상태를 갖는다.

| 값 | 의미 |
|---|---|
| `false` | 동결. `requires_grad=False` |
| `true` | 전체 학습 |
| `lora` | 어댑터만 학습, 베이스 가중치 동결 |

모듈 그룹 이름(`vision_tower`, `projector`, `llm`, `lm_head`, `embed_tokens`)은 백본 어댑터가 매핑을 제공한다(7.5). 백본이 바뀌어도 설정 문법은 그대로다.

세밀한 제어가 필요하면 정규식 형태를 허용한다.

```yaml
trainable:
  llm: lora
  overrides:
    - {pattern: "model.layers.2[0-7].*", mode: true}     # 상위 8개 레이어만 전체 학습
    - {pattern: "vision_tower.blocks.3[0-1].*", mode: true}
```

`stages[].init_from`이 단계 연결을 만든다. 값은 앞 단계 이름이거나 체크포인트 경로다. 지정하지 않으면 백본 초기 가중치에서 시작한다.

## 7.3 VRAM 추정 모델

포트 타입 구조체가 실어 나르는 shape 정보와 Trainer 설정만으로 **학습 시작 전에** 산정한다. 실측이 아니라 정적 추정이므로 보수적으로 잡고 `headroom_ratio`를 요구한다.

```
S_vision = images_per_sample × tiles_per_image × tokens_per_tile
S_total  = S_text + S_vision                     # sequence.max_len 과 비교 → 초과 시 G4 거부

W  = P_total     × b_w        b_w: fp32 4, bf16 2, int8 1, nf4 0.5(+상수 0.03)
G  = P_trainable × b_g        b_g: bf16 2, fp32 4
O  = P_trainable × b_o        b_o: adamw_torch 8, adamw_bnb_8bit 2, adafactor 0.5, sgd_m 4
A  = 활성화
     checkpointing on :  L × B × S_total × H × 2  +  k_blk × B × S_total × H × 2
     checkpointing off:  k_all × L × B × S_total × H × 2        (k_all ≈ 10~14, 구현 의존)
     k_blk ≈ 8 (한 레이어 재계산 중 살아있는 중간 텐서)
Lg = 로짓
     chunked_ce on : chunk × V × 2 × 2
     chunked_ce off: B × S_text × V × 2  (+ fp32 사본이면 ×3)
C  = CUDA 컨텍스트 + 파편화 + 커널 워크스페이스 ≈ 0.8~1.2 GB

VRAM_peak ≈ W + G + O + A + Lg + C
예산 통과 조건: VRAM_peak × (1 + headroom_ratio) ≤ vram_gb − reserve_gb
```

`P_total`, `P_trainable`, `L`, `H`, `V`, `tokens_per_tile`은 **백본 어댑터의 `spec()`이 보고**한다. 백본이 바뀌어도 추정 코드는 그대로다.

## 7.4 단계별 산출 예시

`qwen2.5-vl-7b`(P_total 8.3B, L=28, H=3584, V=152k) / images 4 / tiles 4 / tokens_per_tile 144 → S_vision=2304, S_text=1024, S_total=3328.

| 항목 | stage1 `projector_align` | stage2 `full_ft`(LoRA r=32) | 참고: stage2를 `llm: true`로 바꾸면 |
|---|---|---|---|
| 가중치 W | nf4 8.3B → **4.7 GB** | 4.7 GB | bf16 필요 → **16.6 GB** |
| 학습 파라미터 P_trainable | 0.11 B (projector) | 0.04 B (LoRA) + 0.11 B(projector) | 8.3 B |
| 그래디언트 G | 0.22 GB | 0.30 GB | **16.6 GB** |
| 옵티마이저 O | adamw fp32 → 0.88 GB | adamw 8bit → 0.30 GB | adamw 8bit라도 **16.6 GB** |
| 활성화 A | ckpt off, B=4 → **9.6 GB** | ckpt on, B=1 → **1.3 GB** | ckpt on, B=1 → 1.3 GB |
| 로짓 Lg | chunked → 0.6 GB | chunked → 0.6 GB | 0.6 GB |
| 컨텍스트 C | 1.0 GB | 1.0 GB | 1.0 GB |
| **합계** | **17.0 GB** | **8.2 GB** | **52.7 GB** |
| 예산(24 − 1.5 = 22.5, 여유 10%) | 18.7 ≤ 22.5 **통과** | 9.0 ≤ 22.5 **통과** | 58.0 > 22.5 **거부** |

거부 메시지 형식:

```
BudgetError [G4] stage "full_ft": 추정 52.7 GB > 예산 22.5 GB (초과 30.2 GB)

  초과 기여 (큰 순):
    1) 가중치 16.6 GB — backbone.quantization.mode = none  (nf4로 바꾸면 -11.9 GB)
    2) 옵티마이저 16.6 GB — stages[1].trainable.llm = true  (lora로 바꾸면 -16.4 GB)
    3) 그래디언트 16.6 GB — 위와 동일 원인

  이 검사가 없었다면:
    물질화 40분 + 백본 로드 3분 후 첫 optimizer.step()에서 OOM.
    체크포인트 없음. GPU 한 장이 43분 점유.

  민감도 (다른 값은 고정):
    images_per_sample 4 → 2 :  S_total 3328 → 2176,  활성화 -0.5 GB   (해결 안 됨)
    max_tiles 4 → 2         :  S_total 3328 → 2176,  활성화 -0.5 GB   (해결 안 됨)
    llm: true → lora(r=32)  :  -33.0 GB                               (해결)
    quantization: nf4       :  -11.9 GB                               (부분)
```

### 검증 케이스의 "전체 미세조정"을 24GB에서 표현하는 법

Triad의 2단계는 "Projector 정렬 후 전체 미세조정"이다. 7B 백본의 진짜 전체 미세조정은 24GB에 **들어가지 않는다**. 플랫폼은 그것을 숨기지 않고 위처럼 거부한 뒤 세 갈래를 제시한다.

| 대안 | 설정 | 추정 |
|---|---|---|
| 전 모듈 LoRA(사실상의 전체 미세조정) | `llm: lora`, `targets: 전 linear`, `r: 64`, `vision_tower: lora` | ≈ 9.5 GB |
| 작은 백본으로 진짜 전체 미세조정 | `backbone.id: 2B급`, `llm: true`, `adamw_bnb_8bit`, ckpt on, B=1 | ≈ 15 GB |
| 상위 레이어만 전체 학습 | `overrides: model.layers.2[0-7]` + 나머지 동결 | ≈ 13 GB |

어느 쪽을 택하든 그래프는 그대로다. 바뀌는 것은 `trainer.yaml`뿐이고, Parameter Recipe로 세 갈래를 번호로 전환할 수 있다(문서 13).

## 7.5 백본 어댑터 인터페이스

```python
class BackboneAdapter(Protocol):
    id: str; version: str
    def spec(self) -> BackboneSpec:
        """모델 로드 없이 답해야 한다 (G4가 호출).
        BackboneSpec(params_total, params_by_group, n_layers, hidden, vocab,
                     tokens_per_tile, max_context, module_map, tokenizer_id,
                     supports={quantization, attn_impls, lora_targets},
                     os_support=[windows, linux])"""
    def build(self, cfg: TrainerConfig, stage: Stage) -> TrainModules: ...
    def apply_freeze(self, modules, policy: TrainablePolicy) -> None: ...
    def collate(self, samples: list[Sample], schema: AnswerSchema) -> dict: ...
    def count_vision_tokens(self, images: list[ImageMeta], vision_cfg) -> int: ...
    def save(self, modules, path: Path, stage: Stage) -> None: ...
    def export_inference_contract(self, cfg, graph) -> dict: ...    # 문서 6.5
```

백본 교체 = `backbone.id` + `adapter` 두 줄 변경. 특정 학습 라이브러리에 종속되지 않도록 어댑터 뒤에 전부 숨긴다.

## 7.6 Windows / 단일 GPU 대응

| 제약 | 설계 대응 |
|---|---|
| 프로세스 생성이 `spawn` (fork 없음) | `dataloader.start_method`는 `spawn` 고정. 워커에 전달되는 것은 **직렬화 가능한 스펙 조각과 경로뿐**이며 람다·클로저·열린 핸들·GPU 텐서를 넘기지 않는다. 노드 구현은 모듈 최상위에서 임포트 가능해야 하고(레지스트리가 등록 시 검사), `persistent_workers: true`로 재생성 비용을 상쇄한다 |
| 경로 길이 상한(260자) | 캐시·물질화 경로는 서술적 이름을 쓰지 않고 **짧은 해시 키**를 쓴다(문서 8.3). 실행 산출물 루트는 `runs/<12자 run_id>/`로 제한 |
| 심볼릭 링크 제한 | 캐시 재사용에 symlink를 쓰지 않는다. 하드링크를 시도하고 실패하면 복사로 폴백. 물질화 shard는 항상 실제 파일 |
| 파일 잠금 동작 차이 | 쓰기는 `temp → os.replace` 원자 교체. 잠금은 별도 `.lock` 파일(`msvcrt.locking` 또는 portalocker). 열린 핸들 때문에 삭제가 실패하면 GC 큐에 넣고 다음 실행에서 정리 |
| Linux 전용 런타임 의존 금지 | `runtime.profile: windows_single_gpu`가 허용 목록을 갖는다. `flash_attn2`, Triton 커널 의존 옵티마이저, DeepSpeed ZeRO 등 미지원 항목을 고르면 **G2에서 컴파일 실패**하고 대체안(`sdpa`, `adamw_bnb_8bit`, `offload.optimizer: cpu`)을 제시한다 |
| GPU가 한 장 | `distributed.enabled: false`가 기본. 다중 GPU·다중 노드는 `strategy`와 어댑터 인터페이스로 열어두되 기본 프로파일이 의존하지 않는다. 스윕은 병렬이 아니라 **순차 큐**로 돈다(문서 13.5) |
| 실패 비용이 큼 | 학습 시작 전 4중 게이트 전부 통과 필수. 게이트 우회 플래그 없음 |
