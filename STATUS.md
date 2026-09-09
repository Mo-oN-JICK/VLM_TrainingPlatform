# 진행 상황 (handoff)

이 파일은 **세션이 중단되어도 다음 세션이 그대로 이어받을 수 있게** 유지한다.
작업을 끝낼 때마다 "완료"로 옮기고, 새로 알게 된 제약은 "함정"에 적는다.

- 최종 갱신: 2026-09-09
- 마지막 커밋: `026639a` — Phase 0–1 코어 구현
- 테스트: `python -m pytest tests -q` → **51 passed**

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

### 설계 문서

- [x] `docs/design/` 13편 + README. Mech-Vision 공개 문서와 화면 캡처 3장 실측이 근거이며 `[문서확인]`/`[이미지확인]`/`[추정]`으로 구분 표기

---

## 3. 다음 — Phase 2 (여기서 시작하면 된다)

목표: **합성 더미 데이터로 Triad 구조가 `sample.assemble`까지 흐르고 dry-run(G3)을 통과한다.**

### 3.1 더미 데이터 생성기
- [ ] `tools/make_dummy_dataset.py` — `data/dummy/`에 생성
  - 이미지 N장(PIL로 합성 파형/도형, 크기 제각각으로 만들어 `DYN` 경로를 실제로 태울 것)
  - 시계열 CSV N개(추세 + 주기 + 스파이크를 파라미터로 심어서, 정답 Text의 정오답을 검증할 수 있게)
  - `index.jsonl` — `{sample_id, patient_id, image_path, ecg_path, label, quality_flag}`
  - seed 고정. 같은 seed면 같은 데이터

### 3.2 노드 (최소 12개부터, 필요할 때 늘린다)
- [ ] `nodes/source/` — `source.image`, `source.timeseries`, `source.text_asset`, `source.field`, `schema.define`
- [ ] `nodes/expert/` — `expert.propose` + 더미 `ExpertPlugin`(이미지용/시계열용 각 1개, `deterministic: true`)
- [ ] `nodes/image/` — `image.crop_by_regions`
- [ ] `nodes/adapt/` — `adapt.image_resize`, `adapt.frame`
- [ ] `nodes/timeseries/` — `ts.stats`, `ts.plot`
- [ ] `nodes/text/` — `text.template`, `prompt.knowledge_inject`, `prompt.image_slots`
- [ ] `nodes/answer/` — `answer.evidence_rules`, `answer.stepwise`, `answer.validate`, `answer.leakage_guard`
- [ ] `nodes/data/` — `sample.assemble`
- [ ] `nodes/io/` — `io.dataset_export`(Output)
- [ ] `nodes/train/` — `train.vlm_trainer`(Phase 5까지는 스텁. `budget_table` 미리보기만)
- [ ] 전부 `vlm_trainer/nodes/__init__.py`에서 임포트되어 `load_builtin_nodes()`로 등록되게 할 것

### 3.3 실행 엔진
- [ ] `engine/cache.py` — 캐시 키(설계 08 §8.3 공식), `.cache/<2자>/<16자>/` 짧은 해시 경로, LRU GC
- [ ] `engine/scheduler.py` — 위상 실행, 노드 상태(pending/queued/running/success/cached/failed/skipped/partial), 3분류별 정책
- [ ] `engine/worker.py` — spawn 워커 풀, GPU 배타 락(`.gpu.lock`), 크래시 감지
- [ ] `engine/isolate.py` — 샘플 quarantine, `quarantine_ratio_threshold`(기본 5%)
- [ ] `engine/dryrun.py` — G3: 샘플 N건 전 노드 통과, 선언 타입 vs 실측 대조, **결정성 감사**(2회 실행 해시 비교), 스키마 위반율 집계
- [ ] `engine/preview.py` — 노드 타입별 렌더러(설계 08 §8.4 표), Debug Output 게이팅(외부 트리거는 항상 off)
- [ ] CLI `dryrun`, `preview` 추가

### 3.4 Phase 2 완료 조건
1. 더미 스펙(샘플 20건)이 `vlmt dryrun`을 통과하고 `sample.assemble`까지 값이 흐른다
2. 같은 명령을 두 번 실행하면 두 번째는 전 노드가 `cached`
3. `vlmt preview --node n_crop`이 상류만 계산하고, 두 번째 호출에서는 대상 노드만 실행
4. 워커 프로세스를 강제 종료해도 엔진은 살아 있고 그 노드만 `failed`
5. 의도적으로 비결정적인 테스트 노드를 결정성 감사가 잡아낸다
6. 정답 Text가 스키마를 위반하도록 규칙을 망가뜨리면 dry-run이 위반율로 중단한다

---

## 4. 그 이후 (요약 — 상세는 `docs/design/10-roadmap.md`)

- **Phase 3 자원 예산 G4** — `engine/budget.py`, `BackboneAdapter.spec()`. 완료 조건: 예산 초과 설정이 학습 시작 전에 거부되고 초과 기여를 큰 순으로 지목. **기준값을 3060 12GB / 4090 24GB 두 프로파일로 분리할 것**
- **Phase 4 물질화·재개** — shard 원자 커밋, 저널, `--resume`
- **Phase 5 Trainer·추론 계약** — 2B QLoRA 실학습, `inference_contract.json`, 추론 프롬프트가 학습 프롬프트와 바이트 단위 동일
- **Phase 6 Parameter Recipe·스윕** — 화이트리스트는 이미 컴파일러에 있음. 스펙 파일·전개·순차 큐가 남음
- **Phase 7 UI** — 7파티션, 수직 캔버스, Debug Output, History
- **Phase 8 확장** — 다중 GPU, 원격, 캐시 백엔드

---

## 5. 함정 (실측으로 확인된 것)

| 항목 | 사실 | 대응 |
|---|---|---|
| GPU | RTX 3060 **12GB** (4090 아님) | Phase 3에서 프로파일 분리. stage1 projector 정렬은 `grad_checkpointing: true` + `per_device: 1`이어야 들어간다 |
| torch | **미설치** | Phase 2까지는 필요 없다(numpy/PIL로 충분). Phase 5 전에 설치 |
| Python | 3.14.4 (`C:\Users\wnsgu\AppData\Local\Python\pythoncore-3.14-64`) | torch 휠이 3.14를 지원하는지 미확인. 안 되면 3.12 venv를 따로 만든다 |
| 설치된 패키지 | `yaml` `pytest` `numpy` `pydantic` `PIL` 있음 | 코어는 표준 라이브러리 + yaml만 쓴다. 새 의존성은 정말 필요할 때만 |
| CLI 실행 | `--nodes fixture_nodes`가 필요하면 `PYTHONPATH`에 repo 루트와 `tests`를 넣어야 한다 | Git Bash의 `$PWD`는 POSIX 경로라 Windows Python이 못 읽는다. **Windows 경로로 지정할 것** |
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
