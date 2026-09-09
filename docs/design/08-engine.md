# 8. 그래프 실행 엔진

엔진의 목표는 하나다. **학습 시작 전에 잡을 수 있는 오류를 학습 중에 터뜨리지 않는 것**, 그리고 이미 계산한 것을 두 번 계산하지 않는 것. GPU가 한 장뿐이라 잘못된 실행 하나가 다른 모든 실험을 막기 때문이다.

## 8.1 실행 단위와 순서

```
run
 └─ phase: dryrun → materialize → train
     └─ stage (materialize: shard 단위 / train: TrainerConfig.stages)
         └─ node (위상 순서)
             └─ sample (샘플 키 단위)
```

- 위상 순서는 compile이 산출한 것을 그대로 쓴다. 순환은 캔버스 규약이 구조적으로 막으므로 런타임 검사가 없다.
- 노드 3분류가 정책을 결정한다.

| 분류 | 실행 시점 | 캐시 | 재실행 | 미리보기 | 실패 시 |
|---|---|---|---|---|---|
| Input | 물질화 단계 | 지문 포함 키로 캐시 | 지문이 같으면 스킵 | 허용 | 해당 샘플 quarantine |
| Processing | 물질화 단계(경계 앞) 또는 학습 루프(경계 뒤) | 캐시 | 키 동일하면 스킵 | 허용 | 해당 (노드, 샘플) 격리 |
| Output | 마지막 | 캐시 금지 | 항상 실행 | 금지 | 실행 중단 + 저널 커밋 지점 기록 |

## 8.2 물질화 경계

```
Input ─ Processing ─ … ─ [경계] ══════ 디스크 shard ══════ Trainer
                          ↑ 여기까지 미리 굽는다        ↑ 여기는 shard만 읽는다
```

기본 경로는 **경계까지 전량 물질화**다. 전처리(전문가 모델 호출, crop, 프롬프트 조립, 정답 생성)가 학습과 GPU를 두고 경쟁하지 않게 하기 위해서다.

- 경계는 `materialize.boundary`로 스펙에 지정한다. 보통 `sample.assemble` 노드.
- 경계 앞의 모든 `external_call` 노드는 물질화 단계에서만 살고, 끝나면 프로세스째 종료되어 VRAM을 완전히 반납한다.
- 경계를 비우면 전 그래프가 학습 루프 안에서 온더플라이로 돌고, G4가 예상 지연과 VRAM 경합을 경고한다.
- 물질화 산출물은 `webdataset` tar shard(기본) 또는 arrow. shard는 **원자적으로 커밋**된다(temp → `os.replace` → 매니페스트 append).

물질화 매니페스트:

```jsonl
{"shard": "00017.tar", "n": 512, "bytes": 536870912, "sha256": "…", "keys": ["…"], "split": "train", "committed_at": "…"}
```

## 8.3 캐시 키와 경로

```
key(node) = blake3(
    engine_abi_version ||
    node_type@version ||
    canonical_params(노드 파라미터, Recipe 오버레이 적용 후) ||
    sorted(upstream_output_keys) ||
    (kind == INPUT ? data_fingerprint : "") ||
    (deterministic == false ? seed : "")
)
data_fingerprint = blake3(정규화 절대경로 || size || mtime_ns || content_hash)
```

`content_hash`는 파일 크기가 임계(기본 64MB) 이하면 전체 해시, 넘으면 `size + mtime + 앞뒤 1MB` 해시로 대체한다(속도 타협, `--strict-fingerprint`로 전체 해시 강제 가능).

경로 규칙 — **Windows 경로 길이 상한 때문에 서술적 이름을 쓰지 않는다.**

```
.cache/
  ab/                      key[0:2]
    cd3f19e77b0a2145/      key[2:18]   ← 16자에서 절단
      out.bin              값
      meta.json            {full_key, node_type, created, bytes, hit_count}
```

전체 경로 길이가 상수로 유지된다. 사람이 읽어야 할 때는 `vlmt cache explain <key>`가 매핑을 보여준다.

무효화: 노드 버전, 파라미터, 상류 키, 데이터 지문, 플러그인 가중치 sha, 규칙 파일 해시 중 **어느 하나만 바뀌어도** 키가 바뀐다. 수동 무효화 명령(`vlmt cache clear --node n_crop`)도 둔다.

GC: LRU + 총량 상한(`solution.yaml`의 `cache.max_gb`). 현재 run이 참조하는 키는 절대 제거하지 않는다.

## 8.4 노드 단위 미리보기 (Mech-Vision 모방)

Mech-Vision은 그래프 전체를 돌리지 않고 Step 하나만 실행해 시각화 출력을 본다. `[문서확인: 개별 Step 단독 실행]` 우리도 동일하다.

```
vlmt preview project.yaml --node n_crop [--sample 12]
```

1. `n_crop`의 **상류 서브그래프만** 위상 실행한다.
2. 캐시 히트는 그대로 쓴다. 보통 두 번째 미리보기부터는 대상 노드만 실행된다.
3. Output 노드는 미리보기 금지(부작용). 대신 "무엇을 쓰려 하는지" 요약(dry)만 표시한다.
4. 결과는 노드 타입이 선언한 `preview` 렌더러로 변환된다.

### 노드 타입별 "시각화 출력"의 정의

| 노드 타입 | preview 키 | 패널에 표시되는 것 |
|---|---|---|
| `source.image`, `image.*`, `adapt.image_*` | `image` / `image_grid` | 썸네일(원본 크기·dtype·value_range·layout·frame을 캡션으로 함께 표시). 정규화된 이미지는 표시용으로만 역정규화하고 그 사실을 명시 |
| `source.timeseries`, `ts.window`, `ts.resample` | `timeseries_plot` | 채널별 플롯. x축 단위(`ts_seconds`/`ts_index`)와 hz를 축 라벨에 표시 |
| `expert.propose` | `regions_overlay` | 이미지면 박스 오버레이 + 점수, 시계열이면 구간 음영 + 점수. 상위 k개 순위 표 동반 |
| `ts.stats`, `table.*` | `table` | 컬럼·dtype·값 표 |
| `text.template`, `prompt.*` | `prompt_render` | **슬롯이 모두 치환된 최종 텍스트**. 슬롯 경계를 색으로 구분, 토큰 수와 `max_len` 대비 비율, 이미지 플레이스홀더 개수 표시 |
| `answer.stepwise`, `answer.from_template` | `answer_render` | 렌더된 정답 Text + 단계 구분 표시 |
| `answer.validate` | `answer_validated` | **스키마 검증 결과가 붙은 텍스트**. 통과 단계는 체크, 위반 단계는 규칙 id와 이유를 인라인 표시 |
| `answer.leakage_guard` | `leak_report` | 프롬프트 원문에서 누설 의심 구간 하이라이트 + 제외 구간 표기 |
| `sample.assemble` | `sample_card` | 이미지 썸네일 + 프롬프트 + 정답 + 메타를 한 장의 카드로. 실제 학습에 들어가는 형태 그대로 |
| `eval.*` | `report` | 집계 표 + 임계 대비 상태 |
| `io.dataset_export`(Output) | `dry_summary` | 쓰려는 경로, 예상 shard 수·바이트, split 분포 (실제 쓰기 없음) |
| `train.vlm_trainer`(Output) | `budget_table` | 문서 7.4의 단계별 VRAM 표와 통과/거부 |

## 8.5 Debug Output 토글 (Mech-Vision 모방)

`[문서확인: Project Toolbar의 Debug Output을 켜면 실행 중 각 Step 출력이 패널에 표시된다]`

엔진 수준 계약:

```python
@dataclass
class RunContext:
    trigger: Literal["ui", "cli", "external"]
    debug_output: bool
    ...
```

| 조건 | 동작 |
|---|---|
| `trigger == "ui"` and `debug_output == True` | 각 노드 완료 시 preview 페이로드를 PreviewBus로 push. 패널이 구독 |
| `trigger == "ui"` and `debug_output == False` | preview를 **생성조차 하지 않는다**(계산 절약). 로그에도 남기지 않는다 |
| `trigger == "external"` | `debug_output` 값과 무관하게 **항상 off**. 패널에 찍지 않는다 |
| `trigger == "cli"` | 기본 off. `--debug-output`을 주면 preview를 파일로 떨어뜨린다(패널이 없으므로) |

preview 페이로드는 값 자체가 아니라 렌더된 표현(썸네일 PNG, 텍스트, 표)이며 크기 상한(기본 2MB/노드)을 갖는다. 상한 초과 시 잘라내고 그 사실을 표시한다.

## 8.6 노드 상태와 실패 격리

상태 머신(캔버스와 Log 패널에 그대로 반영, 문서 12):

```
pending → queued → running → success
                          ↘ cached          (키 히트로 건너뜀)
                          ↘ failed          (NodeError)
                          ↘ skipped         (상류 실패로 도달 불가)
                          ↘ partial         (일부 샘플만 quarantine)
```

- 노드는 워커 프로세스에서 실행된다. 워커가 크래시해도 엔진은 살아 있고 그 노드만 `failed`가 된다.
- 샘플 단위 실패 정책: `on_sample_error: skip | quarantine(기본) | abort`. quarantine된 샘플은 키·노드·예외와 함께 `runs/<run_id>/quarantine/`에 기록되고 집계된다.
- `quarantine_ratio_threshold`(기본 5%)를 넘으면 물질화를 중단한다. 데이터의 20%가 조용히 사라진 채 학습이 도는 것을 막는다.
- Output 노드 실패는 격리하지 않는다. `rollback()`으로 부분 산출물을 되돌리고 저널에 마지막 커밋 지점을 남긴다.

## 8.7 중단된 실행의 재개

`runs/<run_id>/journal.jsonl` — append-only, 각 줄이 하나의 커밋 사실.

```jsonl
{"t":"…","phase":"materialize","event":"shard_committed","shard":"00017.tar","n":512,"keys_hash":"…"}
{"t":"…","phase":"train","event":"ckpt","stage":"full_ft","step":1400,"path":"…","dl_state":"…"}
```

```
vlmt run project.yaml --resume 2026-09-09T1130_ab12
```

| 단계 | 재개 방식 |
|---|---|
| 물질화 | 저널의 커밋된 shard를 그대로 인정하고, 남은 샘플 키 집합만 다시 계산. 캐시 히트로 대부분 즉시 통과 |
| 학습 | 마지막 체크포인트 + 옵티마이저 상태 + **데이터로더 상태**(에폭, 순열 seed, 소비한 샘플 수)로 step 단위 재개 |
| 스펙 검사 | 재개 전 canonical spec 해시를 비교. 다르면 거부하고 무엇이 바뀌었는지 diff로 보여준다(`strict_spec_match: true` 기본) |

## 8.8 History 되감기 (Mech-Vision 모방)

`[문서확인: Project Configuration Pane에 History 탭 존재]` `[추정: 클릭 시 그 시점으로 되돌아가는 동작]`

- 편집은 전부 **커맨드 객체**로 표현된다(`AddNode`, `RemoveNode`, `Connect`, `Disconnect`, `SetParam`, `InsertProcedure`, `Move`).
- 각 커맨드는 실행 후 canonical spec의 해시와 함께 `edit_history.jsonl`에 기록된다. 저장되는 것은 스냅샷 전체가 아니라 **직전 스펙과의 diff**다(텍스트 스펙이라 diff가 작다).
- History 탭의 항목을 클릭하면 그 시점까지 diff를 되감아 스펙을 복원하고 GraphModel을 다시 만든다. Undo/Redo는 같은 저널의 커서 이동일 뿐이다.
- `Move`(좌표 변경)는 `layout.json`에만 반영되고 스펙 해시를 바꾸지 않는다. History에는 남되 "의미 없음"으로 표시된다.

## 8.9 Data Storage (Mech-Vision 모방)

`[문서확인: Data Storage 기능이 Project Assistant 아래 존재]` `[추정: 저장 항목 세부]`

```yaml
debug:
  data_storage:
    enabled: true
    path: "D:/vlmt_storage"
    what: [inputs, previews, answers, quarantine]
    keep_runs: 5
    on_full: stop        # stop | rotate
```

- 활성화 시 지정 경로에 실행 결과를 남긴다. 저장 자체가 부작용이므로 `debug.data_storage` **Output 노드**를 통해서만 일어난다(엔진이 암묵적으로 쓰지 않는다).
- 경로가 짧은 해시 구조가 아니라 사람이 지정한 경로이므로, 하위 구조는 `<run_id>/<node_id>/<sample_key>.<ext>`로 얕게 유지한다(경로 길이 상한 대비).
- 디스크 여유 검사는 G4의 일부다.

## 8.10 워커 프로세스 모델 (Windows / 단일 GPU)

```
메인 프로세스 (스케줄러, 저널, PreviewBus, UI/CLI 통신)
 ├── CPU 워커 풀 (spawn, N=논리코어/2 기본)
 │     Processing 노드 중 GPU 불필요한 것: crop, resize, 템플릿, 정답 생성, 검증
 ├── GPU 워커 (단 하나, 배타적)
 │     expert.* 노드. 물질화 단계 동안만 생존, 끝나면 종료
 └── 학습 프로세스 (단 하나)
       GPU 워커가 완전히 종료된 뒤에만 시작. dataloader 워커는 spawn 자식
```

규칙:

1. **GPU를 잡는 프로세스는 동시에 하나만 존재한다.** 파일 락(`.gpu.lock`)으로 강제하며, 다른 run이 이미 GPU를 잡고 있으면 큐에서 대기한다(단일 4090 전제의 핵심).
2. 워커에는 스펙 조각·경로·캐시 키만 보낸다. 큰 값은 캐시 파일 경로로 전달하고, 텐서를 pickle로 넘기지 않는다.
3. 모든 노드 구현은 모듈 최상위에서 임포트 가능해야 한다(spawn 요구). 레지스트리가 등록 시 `__module__` 도달성을 검사한다.
4. 워커 로그는 큐를 통해 메인으로 모여 Log 패널 한 곳에 표시된다.
5. 워커 크래시(예: 네이티브 세그폴트)는 종료 코드로 감지해 해당 노드를 `failed`로 표시하고 풀을 재생성한다.

## 8.11 dry-run (G3)의 엔진 구현

```
vlmt dryrun project.yaml --samples 3
```

1. `sample_space`에서 샘플 N건을 결정적으로 뽑는다(seed 고정, split별 최소 1건).
2. 전 노드를 **실제로** 실행한다. 캐시는 쓰되 기록은 임시 영역에.
3. 각 노드에서 선언 타입 vs 실측(shape, dtype, range, layout)을 비교한다.
4. 결정성 감사: 같은 샘플을 2회 실행해 출력 해시 비교.
5. `answer.validate` 결과를 집계해 위반율을 임계와 비교.
6. 토크나이즈까지 수행해 실제 토큰 길이를 재고, G4의 정적 추정과 대조한다(추정이 실측보다 작으면 추정 모델의 상수를 보정하도록 경고).
7. Output 노드는 실행하지 않고 `dry_summary`만 만든다.

dry-run 비용은 보통 수십 초다. 그 대가로 학습 시작 후에야 드러날 오류를 전부 앞으로 당긴다.
