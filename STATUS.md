# 진행 상황 (handoff)

이 파일은 **세션이 중단되어도 다음 세션이 그대로 이어받을 수 있게** 유지한다.
작업을 끝낼 때마다 "완료"로 옮기고, 새로 알게 된 제약은 "함정"에 적는다.

- 최종 갱신: 2026-09-09
- 마지막 커밋: Phase 7 — 뷰어에 실행 상태와 Debug Output
- 테스트: `.venv\Scripts\python.exe -m pytest tests -q` → **123 passed**
- 실행 환경: **`.venv` (Python 3.12.14 + torch 2.14.0+cu130, CUDA 동작 확인)**
- **4중 게이트가 전부 동작한다.** G1(편집·타입) · G2(compile) · G3(dry-run) · G4(자원 예산)
- 더미 데이터가 없으면 `python tools/make_dummy_dataset.py --n 24`를 먼저 실행한다(엔진 테스트는 없으면 skip)

---

## 0. 이 프로젝트가 무엇인가

Mech-Vision(산업용 3D 비전 노드 편집기)의 규약을 의도적으로 모방한 **범용 VLM 파인튜닝 플랫폼**.
이종 데이터(이미지·센서 시계열·메타데이터)를 정답 Text에 매핑해 학습 데이터를 만들고, 출력 스키마를 정하고, VLM을 파인튜닝한다.
응용 서비스가 아니라 **학습 도구**다. 장기적으로 VLM 외 도메인으로도 확장할 계획이다(코어는 도메인 무관, 노드가 도메인 계층).

설계 문서 13편: [`docs/design/`](docs/design/README.md) · 아티팩트 판본: https://claude.ai/code/artifact/52feaa4b-5290-4c28-8548-9ab8050d5a7b

핵심 원칙 세 줄:
1. Mech-Vision에 있는 개념은 이름과 구조를 그대로 모방한다(4계층, Node Library, 7파티션, Debug Output/History/Data Storage, Parameter Recipe).
2. 딱 한 축에서만 더 엄격하다 — 포트 타입은 구조체이고, 4중 게이트를 다 통과해야 Trainer가 GPU를 잡는다. 근거는 **실패 비용의 비대칭**(비전은 수 초, 우리는 GPU 수십 시간).
3. 그래프는 전처리까지, 학습은 말단 Trainer 노드의 선언형 설정. 양쪽 다 텍스트 스펙으로 compile/decompile되어 CLI로 재현된다.

---

## 1. 확정된 결정 (다시 묻지 말 것)

| 항목 | 결정 | 날짜 |
|---|---|---|
| 데이터셋 | 실데이터 없음 → **합성 더미 데이터로 검증** | 2026-09-09 |
| 백본 | **2B급 + QLoRA**. 7B 전체 미세조정은 24GB에도 안 들어간다 | 2026-09-09 |
| 전문가 모델 | 실물 없음 → **더미 플러그인**(결정적 의사난수 박스/구간)으로 배선만 검증 | 2026-09-09 |
| 진행 방식 | **플랫폼 우선**. 하드코딩 스파이크를 먼저 하지 않는다. 학습 확인이 Phase 5까지 미뤄져도 좋다 | 2026-09-09 |
| 실행 환경 | 현재 PC는 **RTX 3060 12GB**. 4090 24GB는 다른 PC에 있고 나중에 프로파일로 추가 | 2026-09-09 |
| 확장 방향 | 추후 더 큰 서버에서 이어서 진행. 다중 GPU는 인터페이스만 열어두고 기본값으로 두지 않는다 | 2026-09-09 |

---

## 2. 완료

### Phase 0 — 타입 시스템과 레지스트리 ✅

- [x] `vlm_trainer/core/types.py` — `PortType` 12필드 구조체(base/dtype/shape/layout/value_range/norm/colorspace/frame/time_base/semantic/optional/list_of + extra), `Var`·`DimVar`(제네릭), `ANY`(무관 선언), **`DYN`(가변 차원)**, 타입 생성기(`image` `timeseries` `text` `regions`)
- [x] `vlm_trainer/core/unify.py` — 호환성 7규칙, 단일화, 치환 적용, 노드별 변수 이름 분리. 자동 캐스팅 코드 없음
- [x] `vlm_trainer/core/node.py` — `NodeKind`, `Port`, `NodeDef`, `Node`, `RunCtx`(rng/시각만 노출), `NodeError`, `clears_taint`
- [x] `vlm_trainer/core/registry.py` — `@register` 데코레이터. 3분류-포트형상 강제, Input의 `fingerprint()` 요구, `external_call`은 Processing만, recipe 화이트리스트 검증, **spawn 임포트 가능성 검사**(`<locals>`/`__main__` 거부)
- [x] `vlm_trainer/core/errors.py` — 게이트 에러. 필드별 "안 잡혔다면 언제 어디서 터졌을지 + 추정 낭비 + 해결 어댑터" 표

### Phase 1 — 스펙과 컴파일러 ✅

- [x] `vlm_trainer/core/graph.py` — `GraphModel`, `NodeInstance`, `Edge`, `SampleSpace`, `Materialize`
- [x] `vlm_trainer/spec/loader.py` — Project/Procedure YAML 로더, `find_procedure`(핀 고정, 자동 승격 금지)
- [x] `vlm_trainer/spec/canonical.py` — canonical form(기본값 생략 없음), `blake2b` 해시
- [x] `vlm_trainer/spec/decompile.py` — Procedure 되접기/펼치기, 왕복 보장
- [x] `vlm_trainer/core/compiler.py` — resolve → overlay → inline → typecheck(G1) → structure/policy(G2) → order → 캐시 키 → emit
- [x] G2 검사: 미연결 필수 포트 / 미해결 제네릭 / Output 없음 / 도달 불가 노드 / 팬인 / 없는 노드·포트 / **누설 taint 도달성** / **external_call의 물질화 경계 위치** / 손으로 쓴 순환
- [x] `vlm_trainer/cli/main.py` — `compile` `decompile` `nodes` `show`
- [x] `tests/` 51개 — 완료 조건 전부 커버. `tests/fixture_nodes.py`(테스트용 노드), `tests/data/solution/`(Procedure 포함 예제 Solution)

### Phase 2 — 노드 카탈로그와 실행 엔진 ✅

- [x] `tools/make_dummy_dataset.py` — 합성 더미 데이터. 추세·주기·돌출을 심어서 만들고, **이미지 크기를 샘플마다 다르게** 해 가변(dyn) 차원 경로를 실제로 태운다. seed 고정
- [x] `solutions/dummy_ecg/` — 예제 Solution. Procedure(`expert_crop@1.0.0`) 포함, 노드 25개 그래프, 정답 스키마 + 근거 규칙 + 도메인 지식 자산
- [x] `vlm_trainer/answer/schema.py` — AnswerSchema. **렌더러와 파서를 같은 정의에서 생성**하고 9종 위반을 검사한다
- [x] `vlm_trainer/plugins/` — 플러그인 규약 + 더미 전문가 2종. 이미지 영역 지목과 시계열 구간 지목이 **같은 `expert.propose` 노드**로 들어오고 타입만 갈라진다
- [x] 노드 26개 — `source.*`(5) `adapt.*`(3) `image.*` `ts.*`(2) `expert.propose` `text/prompt.*`(3) `answer.*`(4) `list.*`(3) `sample.assemble` `io.dataset_export` `train.vlm_trainer`(스텁) `schema.define`
- [x] `engine/cache.py` — 짧은 해시 경로 `.cache/<2자>/<16자>/`, 원자 교체, Input은 파일 지문을 키에 포함
- [x] `engine/runner.py` — 위상 실행, 3분류별 정책, 노드 상태 6종, (노드, 샘플) 단위 격리, quarantine 비율 임계
- [x] `engine/worker.py` — spawn 워커. `external_call` 노드를 별도 프로세스에서 돌리고 크래시를 `NodeError`로 바꾼다
- [x] `engine/dryrun.py` — G3. 실측 대조 + 결정성 감사 + 정답 위반율 게이트
- [x] `engine/preview.py` — 노드 타입별 시각화 출력(썸네일 / 시계열 / 지목 오버레이 / 렌더된 최종 프롬프트 / 검증 결과가 붙은 정답 / 샘플 카드)
- [x] `engine/samples.py` — sample_space. 그룹 홀드아웃 split(같은 환자가 두 split에 들어가지 않는다), filter, 결정적 표본 추출
- [x] CLI `dryrun` `run` `preview` 추가
- [x] 완료 조건 6개 전부 테스트로 고정 (`tests/test_engine.py`)

**실측 결과**: 22건(필터 통과) 중 8건 실행 시 전 노드 success, 두 번째 실행은 Output을 뺀 전 노드가 `cached`.
정답 Text 예시:

```
<trend>rising — 기저선이 1000표본당 5.991 단위로 올라갑니다.</trend>
<periodicity>present=true, period_n=10 — 자기상관 최대값 0.836에서 주기 10표본이 관찰됩니다.</periodicity>
<spike>found=true, count=2, max_z=4.23 — z>3.000를 넘는 돌출이 2회, 최대 z=4.227입니다.</spike>
<verdict>abnormal — 돌출 2회가 관찰되어 이상 소견으로 판단합니다.</verdict>
```

### Phase 3 — 자원 예산 게이트 ✅

- [x] `plugins/base.py`에 `BackboneSpec` / `BackboneAdapter` / `register_backbone` — `spec()`은 모델을 로드하지 않고 숫자만 답한다
- [x] `plugins/dummy_backbones.py` — `dummy-2b`, `dummy-7b`. 실제 가중치 없이 파라미터 수·레이어·hidden·vocab·컨텍스트만 보고한다
- [x] `train/config.py` — 선언형 TrainerConfig(양자화·비전·시퀀스·손실·단계별 freeze/optimizer/batch/checkpointing·예산). 장치 프로파일 `rtx3060_12gb` / `rtx4090_24gb` / `a100_40gb`
- [x] `engine/budget.py` — 설계 07 §7.3 공식(W·G·O·A·Lg·C), 초과 기여 순위, 민감도 표, 컨텍스트·시퀀스 초과 검사
- [x] **비전 정보를 포트 타입에서 읽는다** — 이미지 개수는 `list.concat.max_n`, 해상도는 ImageList의 shape. 학습을 돌려보지 않고 안다
- [x] dry-run이 실측 문자 수를 보고하고 예산이 그것으로 토큰 수를 환산한다. 선언값이 실측보다 작으면 거부
- [x] CLI `budget` + `--what-if` + `--device`, `run`이 Output 실행 전에 G4를 통과시킨다
- [x] `solutions/dummy_ecg/projects/01_dummy/trainer.yaml` — 2단계(projector 정렬 → LoRA 미세조정), nf4, 3060 프로파일
- [x] 완료 조건 4개 + 검증 11개 (`tests/test_budget.py`)

**실측 결과** (dummy-2b, 이미지 3장, 실측 텍스트 394토큰):

```
자원 예산 [rtx3060_12gb] VRAM 12 GB - reserve 1.0 = 예산 11.0 GB (여유 10% 요구)
  시퀀스: 비전 768 (= 이미지 3 x 타일 1 x 256) + 텍스트 394 (실측) = 1162 / max_len 4096

  단계                        가중치      그래디언트      옵티마이저      활성화      로짓       합계  판정
  projector_align           1.6        0.1        0.3      2.2     0.6      5.7  통과 (6.3)
  lora_ft                   1.6        0.1        0.1      0.1     0.6      3.5  통과 (3.9)
```

7B를 양자화 없이 전체 미세조정하면 23.8 GB로 거부되고, 초과 기여 1위가 가중치(15.5 GB)임을 지목한다.

### Phase 4 — 물질화와 재개 ✅

- [x] `engine/journal.py` — append-only 커밋 저널. `fsync`까지 하고, 크래시로 잘린 마지막 줄은 버린다
- [x] `engine/materialize.py` — 경계 상류만 전 샘플 실행해 shard로 굽는다. **커밋 지점은 매니페스트에 줄이 붙는 순간**이고, 그 전에 죽은 shard는 고아로 보고 지운 뒤 다시 만든다
- [x] shard 원자 커밋: `tmp -> os.replace -> manifest append + fsync`
- [x] `--resume` — 커밋된 키를 인정하고 남은 샘플만 굽는다. 같은 키가 두 번 들어가지 않는다
- [x] 재개 전 canonical spec 해시 비교. 다르면 거부하고 양쪽 해시를 보여준다
- [x] **물질화가 끝나면 워커 풀을 종료**한다 — 전문가 모델이 잡고 있던 VRAM을 완전히 반납해야 학습이 시작될 수 있다
- [x] `train/shards.py` — 산출물 리더. 매니페스트에 없는 파일은 없는 것으로 취급한다
- [x] `runner.execute(on_sample=...)` 콜백 추가 (샘플 단위 커밋 훅)
- [x] CLI `materialize` + `--resume` `--shard-size` `--limit` `--split`
- [x] 완료 조건 4개 + 검증 10개 (`tests/test_materialize.py`)

**실측 결과**: 6건을 shard 3개로 구운 뒤 고아 shard를 심고 `--resume` → 커밋된 6건 재사용, 새로 4건,
고아 2개 삭제. 스펙을 바꾸고 재개하면 양쪽 spec_hash를 보여주며 거부(exit 5).

### Phase 5 — 학습 루프와 추론 계약 ✅ (실물 백본만 남음)

- [x] 환경: `.venv` Python 3.12.14 + torch 2.14.0+cu130. `torch.cuda.is_available() == True`, bf16 matmul 14.9 TFLOP/s
- [x] `plugins/base.py`에 `BackboneAdapter` 계약 확장(build / module_groups / lora_root / collate)
- [x] `plugins/tiny_backbone.py` — **로컬 소형 VLM(2.3M)**. 비전 타워 + 프로젝터 + 4레이어 LLM + UTF-8 바이트 토크나이저. 다운로드 없이 진짜로 학습된다
- [x] `train/freeze.py` — 3상태(false/true/lora) → 실제 `requires_grad`, LoRALinear 주입, 선언과 실제의 일치를 `verify()`가 확인
- [x] `train/loop.py` — 다단계, `init_from` 연결, grad accum, 원자적 체크포인트, **step 단위 재개**, peak VRAM 측정
- [x] `train/contract.py` — `inference_contract.json` + `inference_graph.yaml`. 정답 경로를 잘라내고 프롬프트 경로만 남긴다(누설 차단 노드는 한 단계 위로 거슬러 해소)
- [x] `io.prompt_export` 노드 — 추론 그래프의 종결점이자 바이트 대조 수단
- [x] `NodeDef.per_sample` — Trainer는 샘플 루프가 아니라 데이터셋 전체에 한 번 돈다
- [x] 실행 프로파일 허용 목록 — `flash_attn2` / DeepSpeed / apex / disk offload / distributed를 학습 전에 거부
- [x] CLI `train`, `infer-graph`
- [x] 완료 조건 + 검증 12개 (`tests/test_train.py`)

**실측 결과** (RTX 3060, 12건 물질화 후 학습):

```
학습 [cuda] 샘플 12건
  projector_align  step 30 · loss 5.704 -> 5.716 · 2.1s · peak 0.15 GB
    학습 파라미터: projector 28,928
  lora_ft          step 30 · loss 5.702 -> 5.425 · 2.0s · peak 0.19 GB
    학습 파라미터: projector 28,928, llm 32,768 · LoRA 16개 모듈
추론 그래프 대조: 12건 · 바이트 동일 12건
```

**아직 안 된 것**: 실물 2B 백본. `BackboneAdapter` 계약은 `tiny-vlm`이 참조 구현으로 채워 놓았으므로,
모델 id를 정하고 같은 인터페이스를 구현하면 그래프도 스펙도 그대로 둔 채 `backbone:` 한 줄로 바뀐다.

### Phase 6 — Parameter Recipe와 스윕 ✅

- [x] `spec/recipe.py` — `recipes.yaml` 로더(번호 1~99, 중복·범위 검사), 표시 이름, 구조 변경 금지(`sample_space`/`materialize`/`nodes`/`edges` 접두 거부)
- [x] **Procedure 노출 파라미터 오버라이드** — `p_crop.max_n`이 내부 노드 `p_crop/n_crop.max_n`으로 해소된다. 노출되지 않은 것은 거부
- [x] `status()` — 활성 레시피가 프로젝트를 더 이상 설명하지 못하면 **Customized** (Mech-Vision 규약)
- [x] 스윕 전개 grid / list / random(seed 고정) + `naming` 템플릿 + `recipes.lock.yaml`로 고정
- [x] `engine/sweep.py` — **전개 시점 예산 사전 검사**(큐에 넣기 전에 제외), **물질화 지문 공유**, 단일 GPU 순차 큐
- [x] `materialize_key()` — 경계 상류 캐시 키의 지문. Trainer 설정만 다른 레시피는 같은 bake를 재사용한다
- [x] CLI `recipe list/show/diff/expand/set-active`, `sweep`, 그리고 모든 명령에 `--recipe N`
- [x] 완료 조건 + 검증 14개 (`tests/test_recipe.py`)

**실측 결과** (레시피 3개 순차 실행):

```
스윕: 레시피 3개 · 큐 3개 · 제외 0개
  물질화: 새로 2개 · 재사용 0개 (전처리 지문 2종)
  레시피                   bake                추정    peak   step  결과
  01_baseline           878843d2905a      1.1G   0.19G      6  loss 5.674
  03_single_crop        a38b013c5987      1.1G   0.19G      6  loss 5.661
  04_low_lr             878843d2905a      1.1G   0.19G      6  loss 5.730
```

1번과 4번은 전처리가 같아 **같은 bake를 공유**하고 학습만 따로 돈다. 3번은 crop 개수가 달라 따로 굽는다.

**구현 중 고친 버그**: grad accumulation 카운터가 epoch마다 리셋돼서, 배치 수가 `grad_accum`보다 적은
데이터셋은 optimizer step을 영영 밟지 못했다. 카운터를 epoch 밖으로 옮겼다.

### Phase 7 (착수) — 읽기 전용 그래프 뷰어 ✅

- [x] `ui/tokens.py` — 설계 문서 12의 **실측값이 코드로 사는 곳**. 캡처에서 확인한 것(OBSERVED)과
      같은 대역에서 배정한 것(ASSIGNED)을 주석으로 구분한다. 실측 토큰은 테스트가 지킨다
- [x] `ui/render.py` — 컴파일된 그래프 → 자체 완결 HTML 한 장. 서버도 프레임워크도 외부 스크립트도 없다
- [x] 레이아웃: 위상 레인을 위에서 아래로 쌓고, 레인 안에서는 상류의 무게중심으로 좌우를 정한다
- [x] 캔버스 규약 검증: **Input 최상단 · Output 최하단 · 모든 배선이 아래로 · 카드 겹침 없음**을 테스트로 고정
- [x] 3분류를 색이 아니라 **형태**로 구분(Input은 상단 칩 없음, Output은 하단 칩 없음)
- [x] 포트 칩은 선언이 아니라 **컴파일이 확정한 타입**을 보여준다 — 제네릭이 남으면 눈에 띈다
- [x] 배선 색 = 소스 포트 타입 색을 약간 어둡게(관측된 규칙 그대로)
- [x] 좌측 Node Library(카테고리별 개수) · 우측 Node Quick Info(포트 타입·캐시 키) · 하단 Log
- [x] CLI `view --out --open`
- [x] 검증 12개 (`tests/test_view.py`)

브라우저로 실제 렌더를 확인했다. 25노드·35배선·11레인 그래프가 규약대로 그려진다.

- [x] **실행 상태를 캔버스에 칠한다** — 노드 상태 6종을 좌측 스트라이프 + 점 + 라벨로. 색만으로 구분하지 않는다
- [x] **Debug Output 패널** — 노드 타입별 시각화 출력(설계 08 §8.4의 표 그대로).
      토글이 꺼져 있으면 **미리보기를 생성조차 하지 않고**, 외부 트리거 실행에서는 값과 무관하게 만들지 않는다
- [x] 격리된 샘플 목록과 실행 요약을 Log에 표시
- [x] CLI `run --view --debug-output`

**아직 없는 것**: 편집(드래그·배선·파라미터 수정), 라이브 갱신(실행 중 스트림), History 되감기.
편집기는 이 뷰어 위에 얹는다.

### 설계 문서

- [x] `docs/design/` 13편 + README. Mech-Vision 공개 문서와 화면 캡처 3장 실측이 근거이며 `[문서확인]`/`[이미지확인]`/`[추정]`으로 구분 표기

---

## 3. 다음 — 둘 중 하나를 고른다

### 3a. 실물 2B 백본 (Phase 5의 마지막 조각)
- [ ] 모델 id 확정 (7장의 미결 항목). 2B급 VLM + Windows에서 도는 것
- [ ] `python -m pip install transformers accelerate peft bitsandbytes safetensors sentencepiece`
- [ ] `plugins/hf_backbone.py` — `BackboneAdapter` 구현. `tiny_backbone.py`가 참조 구현이다
      (`spec()`은 가중치 없이 config.json만 읽어 답해야 한다)
- [ ] `collate()`를 실제 프로세서/토크나이저로. `chars_per_token` 추정을 실제 토큰 카운트로 교체
- [ ] QLoRA(nf4) 경로 확인 — 예산 게이트는 이미 nf4를 계산에 넣고 있다
- 완료 조건: 같은 그래프·같은 스펙에서 `backbone:` 한 줄만 바꿔 학습이 완주한다

### 3b. Phase 7 — 편집 가능한 UI (뷰어 위에 얹는다)
- [ ] `ui/server.py` — 코어 API를 감싸는 로컬 서버. **UI 전용 실행 경로를 만들지 않는다**
- [ ] 실행 중 노드 상태 6종을 캔버스에 반영(이벤트 스트림). 색만이 아니라 라벨을 함께
- [ ] Debug Output 패널 — 토글이 꺼져 있으면 preview를 생성조차 하지 않는다
- [ ] 타입이 맞지 않는 배선은 **드롭 자체가 불가**(연결 후 경고가 아니다)
- [ ] History 되감기, Parameter Recipe 편집기(Project Assistant 안)
- 완료 조건: UI로 만든 그래프를 CLI가 같은 결과로 실행한다

## 4. 그 이후 (요약 — 상세는 `docs/design/10-roadmap.md`)

- **Phase 7 UI** — 7파티션, 수직 캔버스, Debug Output, History
- **Phase 8 확장** — 다중 GPU, 원격, 캐시 백엔드

---

## 5. 함정 (실측으로 확인된 것)

| 항목 | 사실 | 대응 |
|---|---|---|
| GPU | RTX 3060 **12GB** (4090 아님) | Phase 3에서 프로파일 분리. stage1 projector 정렬은 `grad_checkpointing: true` + `per_device: 1`이어야 들어간다 |
| torch | **`.venv`에 설치됨** (2.14.0+cu130, CUDA True) | 3.14 host에는 없다. 항상 `.venv\Scripts\python.exe`를 쓴다 |
| `uv venv`는 pip를 넣지 않는다 | 활성화해도 `pip`가 venv 밖으로 샌다 | venv 안에서는 **항상 `python -m pip`**. 이 venv에는 pip를 넣어 두었다 |
| 예산 프로파일 | 기본 `rtx3060_12gb`(이 PC). 4090은 `--device rtx4090_24gb` 또는 trainer.yaml의 `budget.device` | 그래프는 그대로다. 바뀌는 것은 Trainer 설정 한 줄 |
| Python | 3.14.4 (`C:\Users\wnsgu\AppData\Local\Python\pythoncore-3.14-64`) | torch 휠이 3.14를 지원하는지 미확인. 안 되면 3.12 venv를 따로 만든다 |
| 설치된 패키지 | `yaml` `pytest` `numpy` `pydantic` `PIL` 있음 | 코어는 표준 라이브러리 + yaml만 쓴다. 새 의존성은 정말 필요할 때만 |
| CLI 실행 | 내장 노드는 자동 등록된다. `--nodes fixture_nodes`가 필요할 때만 `PYTHONPATH`에 `tests`를 넣는다 | Git Bash의 `$PWD`는 POSIX 경로라 Windows Python이 못 읽는다. **Windows 경로로 지정할 것** |
| 자산 경로 | 스펙 안의 파일 경로(`knowledge/`, `schemas/`)는 **sample_space 인덱스 파일이 있는 디렉터리 기준**이다 | 루트가 하나여야 Input 노드의 지문 계산이 단순해진다. 예제에서 `../../schemas/...`로 쓰는 이유 |
| 더미 데이터 | git에 넣지 않는다(`.gitignore`) | seed 고정이라 `tools/make_dummy_dataset.py`로 언제든 같은 데이터를 다시 만든다 |
| 경로 길이 | Windows MAX_PATH | 캐시·물질화는 서술적 이름 금지, 짧은 해시 경로 |
| 프로세스 | spawn (fork 없음) | 노드 구현은 모듈 최상위에. 레지스트리가 등록 시점에 검사한다 |

---

## 6. 설계 문서와 구현이 다른 지점

`README.md`의 표에 정리되어 있다. 요약: `blake2b`(≠blake3), `list_of` 하나로 리스트 표현(≠별도 base kind), **`DYN`(가변 차원)을 미해결 제네릭과 구분**, 손으로 쓴 YAML에 한해 순환을 구조 오류로 보고.

설계 문서를 고칠 때는 `docs/design/`과 아티팩트 양쪽을 같이 갱신할 것(아티팩트는 같은 URL로 재발행).

---

## 7. 아직 정하지 않은 것

- [ ] 2B급 백본의 구체 모델 id와 어댑터 구현체 (Phase 5 전에 결정)
- [ ] 정답 Text 스키마의 실제 도메인 — 지금은 Triad(ECG) 예시를 그대로 쓰는 중
- [ ] `runtime_profile` 허용 목록의 구체 항목 (Phase 3/5에서 채운다)
- [ ] VLM 외 도메인 확장 시 노드 카테고리를 어떻게 나눌지 (코어는 이미 도메인 무관)
