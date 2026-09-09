# 진행 상황 (handoff)

이 파일은 **세션이 중단되어도 다음 세션이 그대로 이어받을 수 있게** 유지한다.
작업을 끝낼 때마다 "완료"로 옮기고, 새로 알게 된 제약은 "함정"에 적는다.

- 최종 갱신: 2026-09-09
- 마지막 커밋: Phase 3 — 자원 예산 게이트(G4)
- 테스트: `python -m pytest tests -q` → **71 passed**
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

### 설계 문서

- [x] `docs/design/` 13편 + README. Mech-Vision 공개 문서와 화면 캡처 3장 실측이 근거이며 `[문서확인]`/`[이미지확인]`/`[추정]`으로 구분 표기

---

## 3. 다음 — Phase 4 (여기서 시작하면 된다)

목표: **물질화 경계까지 구운 결과를 학습이 그것만 읽고, 중단해도 이어서 재개된다.**

### 3.1 물질화
- [ ] `engine/materialize.py` — `materialize.boundary`(예제에서는 `n_sample`)까지 전 샘플을 실행해 shard로 굽는다
- [ ] shard 원자 커밋: `temp -> os.replace -> manifest.jsonl append`. 부분 tar가 매니페스트에 남으면 안 된다
- [ ] `external_call` 노드(전문가 모델)는 물질화 단계에서만 살고 끝나면 프로세스째 종료 — VRAM을 완전히 반납한다
- [ ] `train/shards.py` — shard 리더. Trainer는 원본 데이터를 다시 읽지 않는다

### 3.2 저널과 재개
- [ ] `engine/journal.py` — append-only `runs/<run_id>/journal.jsonl`. shard 커밋과 체크포인트 사실만 기록
- [ ] `vlmt materialize --resume` — 커밋된 shard를 다시 만들지 않고 남은 샘플 키만 계산
- [ ] 재개 전 canonical spec 해시 비교, 다르면 거부하고 무엇이 바뀌었는지 보여준다

### 3.3 완료 조건
1. 물질화 도중 프로세스를 죽이고 `--resume`하면 커밋된 shard를 다시 만들지 않는다
2. 부분 shard가 매니페스트에 남지 않는다(원자 커밋)
3. `external_call` 노드 종료 후 워커 프로세스가 남지 않는다
4. 스펙이 바뀐 run을 재개하려 하면 거부하고 diff를 보여준다

## 4. 그 이후 (요약 — 상세는 `docs/design/10-roadmap.md`)

- **Phase 5 Trainer·추론 계약** — 2B QLoRA 실학습, `inference_contract.json`, 추론 프롬프트가 학습 프롬프트와 바이트 단위 동일
- **Phase 6 Parameter Recipe·스윕** — 화이트리스트는 이미 컴파일러에 있음. 스펙 파일·전개·순차 큐가 남음
- **Phase 7 UI** — 7파티션, 수직 캔버스, Debug Output, History
- **Phase 8 확장** — 다중 GPU, 원격, 캐시 백엔드

---

## 5. 함정 (실측으로 확인된 것)

| 항목 | 사실 | 대응 |
|---|---|---|
| GPU | RTX 3060 **12GB** (4090 아님) | Phase 3에서 프로파일 분리. stage1 projector 정렬은 `grad_checkpointing: true` + `per_device: 1`이어야 들어간다 |
| torch | **미설치** | Phase 4까지는 필요 없다(numpy/PIL로 충분). Phase 5 전에 설치 |
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
