# 10. 단계별 구축 순서

원칙: **UI보다 코어와 스펙을 먼저.** UI는 스펙을 편집하는 뷰이므로, 스펙과 게이트가 없으면 UI가 만들 수 있는 것도 없다. 각 단계는 그 자체로 쓸모가 있어야 하고(그 단계에서 멈춰도 가치가 남는다), 완료 조건은 실행 가능한 검사여야 한다.

## Phase 0 — 타입 시스템과 레지스트리

- `core/types.py`, `core/unify.py`, `core/registry.py`, `core/node.py`
- 노드 3개만 등록(하나는 Input, 하나는 Processing, 하나는 Output)해 3분류 강제를 검증

**완료 조건**
1. 타입 호환성 표(2.3의 7규칙)를 커버하는 유닛 테스트가 전부 통과. 특히 `u8/HWC/0-255` ↔ `f32/CHW/imagenet` 거부, `frame` 불일치 거부, list↔scalar 거부.
2. 포트 형상이 3분류 선언과 어긋나는 노드는 임포트 시점에 실패한다.
3. 심볼 dim 단일화가 `[?H,?W,3]`와 `[448,448,3]`을 바인딩하고, 충돌 시 거부한다.

## Phase 1 — 스펙과 컴파일러

- `spec/loader.py`, `spec/canonical.py`, `spec/decompile.py`, `spec/lock.py`
- `core/compiler.py` + `checks/` + `core/errors.py`
- CLI `compile` / `decompile`

**완료 조건**
1. `canonical(compile(decompile(compile(s)))) == canonical(compile(s))`가 예제 스펙 5개에서 성립.
2. G2 검사 6종(미연결 포트, 미해결 제네릭, Output 없음, 도달 불가, 팬인, 버전 해소 실패)이 각각 전용 테스트 스펙에서 잡힌다.
3. 에러 메시지가 "안 잡혔다면 언제 어디서 터졌을지"를 포함한다(스냅샷 테스트).
4. Procedure 인라인이 네임스페이스를 정확히 붙이고, `keep_procedures=True` decompile이 원형으로 되접는다.

## Phase 2 — 최소 노드 세트와 실행 엔진

- 문서 3.14의 22개 노드
- `engine/scheduler.py`, `cache.py`, `worker.py`, `isolate.py`
- CLI `dryrun`, `preview`

**완료 조건**
1. Triad 미니 스펙(샘플 20건)이 dry-run을 통과하고 `sample.assemble`까지 값이 흐른다.
2. 같은 명령을 두 번 실행하면 두 번째는 전 노드가 `cached` 상태가 된다.
3. 노드 하나만 실행하는 `preview`가 상류만 계산하고, 두 번째 호출에서는 대상 노드만 실행한다.
4. 워커 프로세스를 강제 종료해도 엔진이 살아 있고 해당 노드만 `failed`로 표시된다.
5. 결정성 감사가 의도적으로 비결정적인 테스트 노드를 잡아낸다.

## Phase 3 — 자원 예산 게이트

- `engine/budget.py`, `plugins/base.py`의 `BackboneAdapter.spec()`
- CLI `budget --what-if`

**완료 조건**
1. 문서 7.4의 세 열(stage1 / stage2 LoRA / stage2 full)에 대한 산출값이 표로 나온다.
2. 24GB를 초과하는 설정이 **학습 시작 전에** 거부되고, 초과 기여를 큰 순으로 지목한다.
3. `images_per_sample`, `max_tiles`, `max_len`을 바꿨을 때의 민감도 표가 함께 출력된다.
4. context window 초과가 truncation이 아니라 거부로 처리된다(`truncation: forbid` 기본).

## Phase 4 — 물질화와 재개

- `engine/materialize.py`, `journal.py`, `train/shards.py`
- CLI `materialize --resume`

**완료 조건**
1. 물질화 도중 프로세스를 죽이고 `--resume`하면 커밋된 shard를 다시 만들지 않는다.
2. shard 커밋이 원자적이다(부분 tar가 매니페스트에 남지 않는다).
3. `external_call` 노드가 물질화 종료 시 프로세스째 종료되어 VRAM 사용량이 0으로 돌아온다.
4. 경계 뒤에 `expert.*`를 두면 G2가 거부한다.
5. quarantine 비율이 임계를 넘으면 물질화가 중단된다.

## Phase 5 — Trainer와 추론 계약

- `train/loop.py`, `freeze.py`, `contract.py`, `nodes/train/`
- 백본 어댑터 1종 실구현

**완료 조건**
1. Triad 2단계(projector 정렬 → LoRA 미세조정)가 단일 4090에서 완주한다.
2. 학습 중단 후 `--resume`이 step 단위로 이어지고 데이터로더 순서가 재현된다.
3. `inference_contract.json`과 `inference_graph.yaml`이 생성되고, 그 그래프로 만든 프롬프트가 학습 시 프롬프트와 **바이트 단위로 동일**하다(테스트).
4. `freeze` 정책이 실제 `requires_grad` 상태와 일치함을 검사하는 테스트가 있다.
5. Windows 미지원 옵션 선택 시 G2가 대체안과 함께 거부한다.

## Phase 6 — Parameter Recipe와 스윕

- `spec/recipe.py`, CLI `recipe`, `sweep`

**완료 조건**
1. 그래프 1개 + 레시피 3개로 실험 3개가 코드 수정 없이 순차 실행된다.
2. 화이트리스트 밖 파라미터를 덮어쓰려 하면 거부된다.
3. `type_affecting` 파라미터를 덮어쓰면 G1/G2/G4가 재실행된다.
4. 스윕 전개 시 **모든 레시피의 예산 검사를 먼저** 돌려, 초과하는 레시피는 큐에 넣지 않고 목록으로 보고한다.
5. 프로젝트 값을 직접 수정하면 활성 레시피가 `Customized`로 전환된다(Mech-Vision 규약 모방).

## Phase 7 — UI

- `ui/server.py`, `ui/web/` — 7파티션, 수직 캔버스, Debug Output, Step Parameters, Node Quick Info, History, Comment List
- 문서 12의 토큰 적용

**완료 조건**
1. 7개 파티션이 문서 12의 와이어프레임대로 배치되고 도킹·접기가 동작한다.
2. 타입이 맞지 않는 배선은 **드롭 자체가 되지 않는다**(연결 후 경고가 아니라 연결 불가).
3. 노드 상태 6종이 캔버스에서 시각적으로 구분된다.
4. Debug Output 토글 off일 때 preview가 생성되지 않는다(계산량으로 확인).
5. History 항목 클릭이 그 시점 스펙으로 되감고, 좌표 변경은 스펙 해시를 바꾸지 않는다.
6. UI로 만든 그래프가 저장되면 CLI가 같은 결과로 실행한다(UI 전용 경로 없음을 테스트로 고정).

## Phase 8 — 확장 지점

- 다중 GPU / 다중 노드 전략, 원격 실행, 캐시 백엔드 교체
- 기본 프로파일(`windows_single_gpu`)이 이것들에 의존하지 않음을 회귀 테스트로 고정

**완료 조건**
1. `distributed.enabled: true`가 인터페이스 수준에서 동작하되, 기본 프로파일 테스트 전부가 여전히 단일 GPU에서 통과한다.
2. `StorageBackend`를 교체해도 캐시 키 계산이 동일하다.

## 순서를 이렇게 잡은 이유

- **타입 → 컴파일 → 실행 → 예산 → 학습** 순서는 게이트를 값싼 것부터 세우는 순서다. 가장 싼 검사(G1)가 가장 비싼 오류(조용한 정규화 오류)를 잡는다.
- UI가 Phase 7인 이유는, UI가 스펙에 없는 기능을 갖게 되는 순간 재현성이 무너지기 때문이다. 코어가 먼저 완결되어야 UI가 뷰로만 남는다.
- Parameter Recipe(Phase 6)가 Trainer(Phase 5)보다 뒤인 이유는, 레시피가 덮어쓸 값의 목록이 Trainer 스키마가 확정된 뒤에야 정해지기 때문이다.
