# 1. 플랫폼 레이어 구조와 데이터 흐름

## 1.1 레이어 스택

```
┌─────────────────────────────────────────────────────────────────────┐
│ L4  Interface Layer                                                 │
│     Editor UI (7 partitions)          CLI (vlmt)                    │
│       - graph edit / preview            compile dryrun budget       │
│       - Debug Output / History          run preview decompile       │
│       - Parameter Recipe editor         recipe sweep resume         │
│     둘 다 L3 이하만 호출한다. UI 전용 실행 경로는 존재하지 않는다.  │
├─────────────────────────────────────────────────────────────────────┤
│ L3  Spec Layer  (진실의 원천 / 버전 관리 단위)                      │
│     solution.yaml · project.yaml · procedures/*.yaml · recipes.yaml │
│     graph.lock · answer_schema.json · layout.json(sidecar, 의미 없음)│
│         compile ↓            ↑ decompile                            │
├─────────────────────────────────────────────────────────────────────┤
│ L2  Core                                                            │
│     TypeSystem   NodeRegistry   GraphModel   Compiler(G1·G2)        │
│     제네릭 해소 · 위상 순서 · 캐시 키 계산 · CompiledGraph 산출     │
├─────────────────────────────────────────────────────────────────────┤
│ L1  Execution Engine                                                │
│     Scheduler   CacheStore   Materializer   WorkerPool   RunJournal │
│     PreviewBus  BudgetEstimator(G4)  DryRunner(G3)  FailureIsolator │
├─────────────────────────────────────────────────────────────────────┤
│ L0  Node & Plugin Layer                                             │
│     내장 노드 카탈로그(카테고리별)                                  │
│     플러그인 호스트: ExpertProposer · BackboneAdapter · Storage      │
└─────────────────────────────────────────────────────────────────────┘
```

핵심 규칙: **UI는 L3의 스펙을 편집하는 뷰일 뿐이다.** UI에서만 되고 스펙에서는 안 되는 기능은 만들지 않는다. 재현성·버전 관리는 전부 L3 텍스트 위에서 일어난다.

## 1.2 4계층 (Mech-Vision 모방 `[문서확인]`)

```
Solution  ── 하나의 연구 과제. 레시피 묶음 + 공용 자산
   │
   ├── Project (01_baseline, 02_triad_ecg, …)  ── 하나의 학습 레시피 = 그래프 1개 + Trainer 1개
   │        │
   │        ├── Procedure 인스턴스 (재사용 서브그래프의 호출)
   │        │        └── Node
   │        └── Node
   │
   ├── procedures/   ── Solution 범위의 Procedure 라이브러리
   ├── schemas/      ── 정답 Text 스키마
   ├── knowledge/    ── 도메인 지식 텍스트 자산
   └── data/         ── 데이터셋 인덱스(가리키기만, 원본은 외부 경로 허용)
```

Mech-Vision과 동일하게 **Project는 단독으로 존재하지 못하고 반드시 Solution에 속한다.** `[문서확인]`
Project는 Project List에서 번호 접두를 갖고, 이 번호가 CLI/외부 트리거의 프로젝트 식별자다. `[이미지확인]` `[문서확인: Vision_Proj_Num]`

### 계층별 저장·버전·상속

| 계층 | 저장 단위 | 버전 관리 단위 | 파라미터 상속 |
|---|---|---|---|
| **Solution** | 디렉터리 + `solution.yaml` | git 리포지토리(커밋 단위). Solution 전체가 한 번에 커밋된다 | `defaults:` 블록이 하위 전 Project의 노드 파라미터 기본값을 제공. 키는 `<node_type>.<param>` 형태 |
| **Project** | `projects/NN_name/` 디렉터리. `project.yaml`(의미) + `layout.json`(좌표, 의미 없음) + `graph.lock`(해소된 노드·플러그인 버전) + `recipes.yaml` | 같은 git 커밋 + `graph.lock`의 해시. 실행 시 `runs/<run_id>/spec.snapshot.yaml`으로 동결 | Solution defaults를 상속. Project `defaults:`로 재정의 가능 |
| **Procedure** | `procedures/<name>.yaml`, 파일 내 `version: <semver>` | 이름+semver로 불변 게시. 참조는 `ref: expert_region_crop@1.2.0`로 **핀 고정**. 수정은 새 버전 발행 | Procedure는 **노출 파라미터(exposed params)만** 외부에서 받는다 `[문서확인: 내부 Step 파라미터를 Procedure 파라미터로 노출]`. 노출되지 않은 내부 파라미터는 상속·오버라이드 대상이 아니다(캡슐화) |
| **Node** | 코드 + 레지스트리 매니페스트. 스펙에는 `type: image.crop@1.0.0`로 등장 | 노드 타입 semver. major 변경은 자동 마이그레이션하지 않고 compile 실패 후 마이그레이션 명령 안내 | Solution defaults → Project defaults → Procedure 노출값 → 노드 인스턴스 `params:` → Parameter Recipe 오버레이 순으로 **뒤가 이긴다** |

파라미터 우선순위(최종):

```
solution.defaults
  < project.defaults
  < procedure instance params (노출된 것만)
  < node instance params
  < parameter recipe overlay   ← 가장 강함, 단 화이트리스트 안에서만 (문서 13)
```

Parameter Recipe는 **값만** 덮어쓴다. 노드 추가/삭제/배선 변경은 어떤 경우에도 불가. `[각색: Mech-Vision과 동일 원칙, 우리는 정적 검사로 강제]`

## 1.3 데이터 흐름 — 편집에서 학습까지

```
[편집]  Editor UI ──(graph mutation commands)──▶ GraphModel ──▶ History Journal
                                                    │
                                          decompile │ compile
                                                    ▼
[스펙]  project.yaml  +  recipes.yaml  +  procedures/*.yaml      ← git이 보는 유일한 것
                                                    │
                                                    ▼
[게이트] G1 편집시 정적 ─ G2 compile ─ G3 dry-run(샘플 1건) ─ G4 자원 예산
                                                    │  넷 다 통과해야 아래로 내려간다
                                                    ▼
[물질화] Input/Processing 노드를 샘플 전수 실행 → shard(디스크)  ← 여기까지 GPU를 잠깐 씀
                                                    │            (전문가 모델 로드 후 언로드)
                                                    ▼
[학습]   Trainer(Output 노드) — shard만 읽음 → 단계별 학습 → checkpoint + inference_contract.json
```

## 1.4 데이터 흐름 — 그래프 내부(런타임 의미론)

그래프는 **샘플 1건에 대한 계산**을 기술한다. 배치·병렬·순서는 엔진의 몫이고 그래프에 등장하지 않는다.

```
sample_space (Project 수준 선언: 인덱스 파일 + 키 + split)
        │  N개의 sample_id를 정의
        ▼
┌───────────────── 그래프(수직 DAG, 위→아래) ─────────────────┐
│  Input 레인 (최상단)                                        │
│   source.image        source.timeseries   source.text_asset │
│   source.field(label) schema.define                         │
│        │                   │                    │           │
│  Processing 레인                                            │
│   expert.propose ─▶ adapt.frame ─▶ image.crop_by_regions     │
│   ts.window ─▶ ts.stats_to_text                             │
│   text.template(prompt)      answer.stepwise(정답)          │
│   answer.validate  answer.leakage_guard                     │
│        └──────────▶ sample.assemble ◀───────────┘           │
│                          │                                  │
│  Output 레인 (최하단)                                       │
│   io.dataset_export     train.vlm_trainer    debug.data_storage │
└─────────────────────────────────────────────────────────────┘
```

- **Input 노드는 배선으로 데이터를 받지 않는다.** 대신 Project의 `sample_space` 선언에서 컬럼을 파라미터로 지정해 값을 주입한다. 이것이 "입력 포트 없음" 규약과 "여러 소스의 키 조인"을 동시에 만족시키는 유일한 방법이다. `[신설: Mech-Vision의 Camera Step이 외부 장치에서 데이터를 주입하는 자리에 대응]`
- Processing 노드는 파일 시스템·네트워크·시계에 절대 접근하지 않는다. 외부 상태 의존은 Input 노드에만 허용된다.
- Output 노드가 유일한 부작용 지점이며 그래프는 최소 하나의 Output으로 끝나야 한다.

## 1.5 흐름 방향 규약

배선은 **어떤 노드의 하단 출력 포트 → 다른 노드의 상단 입력 포트**로만 생성된다. `[이미지확인: canvas의 수직 흐름과 상·하단 포트 칩]`
아래에서 위로 향하는 배선은 편집기에서 만들 수 없고, 이 규칙 자체가 순환을 구조적으로 금지하므로 compile 단계에 별도의 순환 검사는 두지 않는다. `[모방: 수직 흐름은 Mech-Vision 캔버스에서 관찰. 순환 검사를 생략한다는 결론만 우리 결정]`
