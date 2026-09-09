# 4. Project Spec 스키마와 compile / decompile

용어: 여기서 다루는 것은 **하나의 학습 레시피 = Project Spec**이다. Mech-Vision의 Parameter Recipe에 대응하는 값 오버레이는 문서 13에서 따로 다룬다.

## 4.1 파일 구성

```
solutions/ecg_anomaly/                 <- Solution 루트 (= git 리포지토리 단위)
  solution.yaml                        의미: 과제 메타 + 전역 defaults + 실행 프로파일
  procedures/
    expert_region_crop.yaml            재사용 서브그래프 (name@semver로 게시)
  schemas/
    triad_answer.yaml                  정답 Text 스키마 (문서 6)
  knowledge/
    ecg_rules.md                       도메인 지식 텍스트 자산
  data/
    index.jsonl                        sample_space 인덱스 (원본 파일 경로는 외부 허용)
  projects/
    02_triad_ecg/
      project.yaml                     의미: 그래프 + Trainer 설정  <- 진실의 원천
      recipes.yaml                     Parameter Recipe (문서 13)
      layout.json                      노드 좌표. 의미 없음, compile 입력 아님
      graph.lock                       해소된 노드/플러그인/백본 버전
```

**`project.yaml`만이 의미를 갖는다.** `layout.json`은 sidecar이며 decompile 왕복에서 의미 비교 대상이 아니다. 좌표가 사라져도 실행 결과는 동일하다.

## 4.2 Project Spec 스키마

```yaml
spec_version: 1
kind: Project
id: "02_triad_ecg"          # 디렉터리명과 일치. 앞의 번호가 Project List 번호 (Mech-Vision 모방)
name: "Triad ECG 이상 판정"
solution: "../../solution.yaml"

# ── 샘플 공간: Input 노드들이 값을 길어오는 원천. 그래프 밖의 프로젝트 수준 선언 ──
sample_space:
  index: "../../data/index.jsonl"
  key: "sample_id"
  splits:
    strategy: group_holdout        # random | group_holdout | column
    group_by: "patient_id"         # 누설 방지: 같은 환자는 한 split에만
    ratios: {train: 0.9, val: 0.1}
    seed: 20260909
  filter: "quality_flag == 'ok'"   # 순수 식만 허용

defaults:                          # 이 프로젝트 안 모든 노드에 적용되는 기본값
  adapt.image_resize: {mode: bicubic, keep_aspect: true}

# ── 노드 ──
nodes:
  - id: n_img
    type: source.image@1.0.0
    params: {column: image_path, color_space: RGB}
  # ...

# ── 배선: 항상 (출력 포트 -> 입력 포트), 위에서 아래로 ──
edges:
  - {from: "n_img:image", to: "p_expert:image"}

# ── Procedure 인스턴스 ──
procedures:
  - id: p_expert
    ref: "expert_region_crop@1.2.0"
    params: {topk: 3, padding_ratio: 0.15}     # Procedure가 노출한 파라미터만

# ── 물질화 경계: 이 노드들의 출력까지 디스크에 굽고, 학습은 그것만 읽는다 ──
materialize:
  boundary: [n_sample]
  format: webdataset
  shard_size_mb: 512
  out_dir: "runs/{run_id}/materialized"    # 또는 재사용 가능한 고정 경로

# ── 실행 프로파일 ──
runtime_profile: windows_single_gpu        # solution.yaml에서 상속 가능

# ── 디버깅 (Mech-Vision 규약) ──
debug:
  debug_output: false        # Project Toolbar 토글의 영속값
  data_storage: {enabled: false, path: "D:/vlmt_storage", what: [previews, answers], keep_runs: 5}
```

### 필드 규칙

| 필드 | 규칙 |
|---|---|
| `nodes[].id` | 프로젝트 내 유일. 소문자+숫자+`_`. Procedure 내부 노드는 compile 시 `p_expert/n_crop`으로 네임스페이스가 붙는다 |
| `nodes[].type` | `이름@semver` 필수. `@` 생략은 사람이 쓸 때만 허용되고 canonical 형태에서는 항상 채워진다 |
| `edges[]` | `"노드id:포트명"` 문자열. 팬인 금지(같은 `to`가 두 번 나오면 compile 실패) |
| `procedures[]` | `ref`는 `name@semver`로 고정 핀. 범위 지정(`^1.2`) 불가 — 재현성 우선 |
| `materialize.boundary` | 비면 전 그래프가 학습 루프 안에서 온더플라이 실행된다(허용하되 G4가 경고와 함께 예상 지연을 표시) |
| `trainer` | 별도 필드 없음. Output 노드 중 `train.*`가 그 역할이며 0개 또는 1개 |

## 4.3 canonical form

compile은 사람이 쓴 스펙을 **canonical spec**으로 정규화한다. 정규화 규칙:

1. 모든 노드 타입에 버전을 명시적으로 채운다(lock 기준).
2. **모든 파라미터의 기본값을 명시적으로 기록한다.** 생략하지 않는다. 노드 버전이 올라가며 기본값이 바뀌어도 과거 실험이 재현되어야 하기 때문이다.
3. 노드는 위상 순서로 정렬하고, 동순위는 `(type, id)` 사전순으로 정렬한다.
4. 매핑 키는 사전순, 리스트 순서는 의미가 있으므로 보존한다.
5. 부동소수는 `repr` 왕복이 되는 최단 표기로 기록한다.
6. 주석·좌표·UI 상태는 제거한다.

canonical spec의 blake3 해시가 **실험 동일성의 정의**다. 같은 해시 = 같은 실험.

## 4.4 compile

```python
def compile(spec_path: Path, recipe: int | None = None) -> CompiledGraph
```

| 단계 | 하는 일 | 실패 시 |
|---|---|---|
| 1. parse | YAML 로드, 스펙 스키마 검증 | 문법/필드 오류를 줄 번호와 함께 |
| 2. resolve | 노드·Procedure·플러그인·백본 버전 해소 → `graph.lock` 갱신 | G2: 버전 해소 실패 |
| 3. overlay | Parameter Recipe 적용(있으면). 화이트리스트 밖 키는 즉시 거부 | G2: 오버라이드 금지 필드 |
| 4. inline | Procedure를 재귀적으로 펼치고 id에 네임스페이스 부여 | G2: 순환 참조, 미노출 포트 바인딩 |
| 5. typecheck | 배선마다 §2.3 판정 + 심볼 dim 단일화 + 제네릭 해소 | G1/G2: 타입 불일치, 미해결 변수 |
| 6. structure | 미연결 필수 포트, 도달 불가 노드, Output 존재, 팬인 위반, `external_call` 배치 | G2 |
| 7. policy | 누설 taint 도달성, Windows 프로파일 미지원 옵션, Processing 순수성 선언 검사 | G2 |
| 8. order | 위상 순서 산출(수직 레인) + 노드별 캐시 키 계산 | — |
| 9. emit | `compiled.json`(실행 계획) + `graph.lock` + canonical spec 기록 | — |

순환 검사는 없다. 캔버스가 상단 입력 → 하단 출력만 허용해 순환을 구조적으로 만들 수 없기 때문이다(문서 1.5).

`compiled.json`에는 노드별 `{id, type@ver, params(정규화), inputs, outputs, resolved_types, cache_key, kind, lane}`가 들어간다. 실행 엔진은 스펙을 다시 읽지 않고 이것만 읽는다.

## 4.5 decompile

```python
def decompile(g: CompiledGraph | GraphModel, *, keep_procedures: bool = True) -> Spec
```

- **왕복 보장**: `canonical(compile(decompile(compile(s)))) == canonical(compile(s))`. 즉 의미는 보존되고 주석·좌표는 보존되지 않는다.
- `keep_procedures=True`면 인라인된 노드를 원래 Procedure 인스턴스로 되접는다(`graph.lock`의 출처 기록 사용). `False`면 완전히 펼쳐진 평면 그래프를 낸다.
- UI에서 그래프를 편집하면 GraphModel이 바뀌고, 저장 시 decompile로 `project.yaml`이 다시 쓰인다. 좌표만 바뀐 편집은 `project.yaml`을 건드리지 않고 `layout.json`만 갱신한다(불필요한 diff 방지).

## 4.6 CLI

```
vlmt compile   projects/02_triad_ecg/project.yaml [--recipe N]
vlmt dryrun    projects/02_triad_ecg/project.yaml [--samples 3] [--recipe N]
vlmt budget    projects/02_triad_ecg/project.yaml [--recipe N] [--what-if images=3,tiles=6]
vlmt materialize projects/02_triad_ecg/project.yaml [--resume]
vlmt run       projects/02_triad_ecg/project.yaml [--recipe N] [--resume RUN_ID]
vlmt preview   projects/02_triad_ecg/project.yaml --node n_crop [--sample 12]
vlmt decompile runs/2026-09-09T1130_ab12/compiled.json -o restored.yaml
vlmt sweep     projects/02_triad_ecg/project.yaml --recipes 10-21
```

`run`은 내부적으로 compile → dryrun → budget → materialize → train 순서로 게이트를 전부 통과시킨다. 게이트를 건너뛰는 플래그는 없다.

## 4.7 검증 케이스 — Triad(ICCV 2025) 구조의 실제 스펙

전문가 모델이 의심 영역을 지목 → crop을 원본과 함께 입력 → 도메인 지식 텍스트 주입 → 단계별 구조의 정답 Text → Projector 정렬 후 전체 미세조정.

### 4.7.1 재사용 Procedure

```yaml
# procedures/expert_region_crop.yaml
kind: Procedure
name: expert_region_crop
version: "1.2.0"
summary: "전문가 모델로 의심 영역을 지목하고 crop을 만든다. 이미지/시계열 공통."

exposed_inputs:
  subject: {port: "n_expert:subject"}       # 제네릭 T
exposed_outputs:
  regions: {port: "n_topk:regions"}
  crops:   {port: "n_resize:images"}
exposed_params:
  plugin:         {node: n_expert, param: plugin}
  topk:           {node: n_topk,   param: k}
  score_threshold:{node: n_topk,   param: min_score}
  padding_ratio:  {node: n_crop,   param: padding_ratio}
  out_size:       {node: n_resize, param: size}

nodes:
  - {id: n_expert, type: expert.propose@1.0.0,        params: {plugin: null, device: cuda, dtype: fp16, seed: 0}}
  - {id: n_topk,   type: adapt.regions_topk@1.0.0,    params: {k: 3, min_score: 0.3, sort_by: score}}
  - {id: n_frame,  type: adapt.frame@1.0.0,           params: {to: orig_px, clip_to_bounds: true}}
  - {id: n_crop,   type: image.crop_by_regions@1.0.0, params: {padding_ratio: 0.15, square_pad: true, max_n: 3, min_side_px: 32}}
  - {id: n_resize, type: adapt.image_resize@1.0.0,    params: {size: [448, 448], mode: bicubic, keep_aspect: true, pad_value: 0}}
edges:
  - {from: "n_expert:regions", to: "n_topk:regions"}
  - {from: "n_topk:regions",   to: "n_frame:regions"}
  - {from: "n_frame:regions",  to: "n_crop:regions"}
  - {from: "n_crop:crops",     to: "n_resize:images"}
```

이 Procedure는 `subject`가 `Image`면 `n_crop`이, `TimeSeries`면 `ts.window` 변형이 필요하므로 실제로는 두 벌(`expert_region_crop`, `expert_interval_window`)로 게시한다. 전문가 노드 인터페이스는 하나(`expert.propose`)로 공유된다(문서 5).

### 4.7.2 Project Spec

```yaml
spec_version: 1
kind: Project
id: "02_triad_ecg"
name: "Triad ECG 이상 판정"
solution: "../../solution.yaml"

sample_space:
  index: "../../data/index.jsonl"     # {sample_id, patient_id, image_path, ecg_path, label, quality_flag}
  key: sample_id
  splits: {strategy: group_holdout, group_by: patient_id, ratios: {train: 0.9, val: 0.1}, seed: 20260909}

nodes:
  # ── Input 레인 ─────────────────────────────────────────────
  - {id: n_img,    type: source.image@1.0.0,      params: {column: image_path, color_space: RGB, on_missing: skip}}
  - {id: n_ts,     type: source.timeseries@1.0.0, params: {column: ecg_path, format: csv, hz: 500, channels: 12, t0_policy: sample_start}}
  - {id: n_kb,     type: source.text_asset@1.0.0, params: {path: "../../knowledge/ecg_rules.md", max_chars: 2400}}
  - {id: n_label,  type: source.field@1.0.0,      params: {column: label, semantic: label}}
  - {id: n_schema, type: schema.define@1.0.0,     params: {path: "../../schemas/triad_answer.yaml"}}

  # ── Processing 레인 ────────────────────────────────────────
  - {id: n_tsplot, type: ts.plot@1.0.0,           params: {size_px: [1024, 512], channels_per_row: 3, mark_regions: true, theme: light}}
  - {id: n_stats,  type: ts.stats@1.0.0,          params: {features: [trend_slope, acf_peak, acf_period_s, spike_count, spike_max_z], window_s: 10}}
  - {id: n_ev,     type: answer.evidence_rules@1.0.0, params: {rules: "../../schemas/triad_evidence.yaml", precision: 2}}

  - {id: n_imgadapt, type: adapt.image_resize@1.0.0, params: {size: [896, 896], mode: bicubic, keep_aspect: true, pad_value: 0}}
  - {id: n_imglist,  type: list.wrap@1.0.0,         params: {}}
  - {id: n_allimgs,  type: list.concat@1.0.0,       params: {max_n: 4, on_overflow: fail}}

  - {id: n_prompt0, type: text.template@1.0.0, params:
      slots: {task: Text, hint_n: Text}
      template: |
        당신은 심전도 판독 보조 시스템입니다. 원본 파형 이미지와, 전문가 모델이 지목한
        의심 구간 {hint_n}개의 확대 이미지가 주어집니다. 아래 순서로만 답하십시오.
        (1) 추세 (2) 주기성 (3) 스파이크 확인 (4) 최종 판정
        과제: {task}}
  - {id: n_prompt1, type: prompt.knowledge_inject@1.0.0, params: {position: before, header: "## 도메인 규칙", max_chars: 2400, dedup: true}}
  - {id: n_prompt2, type: prompt.image_slots@1.0.0,      params: {placeholder: "<image>", policy: prepend}}

  - {id: n_answer,  type: answer.stepwise@1.0.0,  params: {render: tagged, step_order: [trend, periodicity, spike, verdict], locale: ko}}
  - {id: n_val,     type: answer.validate@1.0.0,  params: {on_violation: quarantine, max_tokens: 320}}
  - {id: n_leak,    type: answer.leakage_guard@1.0.0, params: {mode: normalized, on_leak: fail}}
  - {id: n_sample,  type: sample.assemble@1.0.0,  params: {id_from: sample_id, require_validated: true, keep_meta_cols: [patient_id]}}

  # ── Output 레인 ────────────────────────────────────────────
  - {id: n_export,  type: io.dataset_export@1.0.0, params: {format: webdataset, out_dir: "runs/{run_id}/materialized", shard_size: 512, image_encoding: "jpeg:95", split_by: split}}
  - {id: n_train,   type: train.vlm_trainer@1.0.0, params: {trainer_config: "./trainer.yaml"}}   # 문서 7

procedures:
  - id: p_expert
    ref: "expert_region_crop@1.2.0"
    params: {plugin: "ecg_region_proposer@0.3.0", topk: 3, score_threshold: 0.35, padding_ratio: 0.15, out_size: [448, 448]}

edges:
  # 전문가 지목: 원본 파형 이미지를 대상으로
  - {from: "n_tsplot:plot",     to: "p_expert:subject"}
  # 원본 이미지 경로도 함께 (촬영본이 별도로 있는 경우)
  - {from: "n_img:image",       to: "n_imgadapt:image"}
  - {from: "n_imgadapt:image",  to: "n_imglist:item"}
  - {from: "n_imglist:items",   to: "n_allimgs:a"}
  - {from: "p_expert:crops",    to: "n_allimgs:b"}
  # 시계열 통계 -> 근거 문장 -> 단계별 정답
  - {from: "n_ts:series",       to: "n_tsplot:series"}
  - {from: "n_ts:series",       to: "n_stats:series"}
  - {from: "n_stats:stats",     to: "n_ev:stats"}
  - {from: "p_expert:regions",  to: "n_ev:regions"}
  - {from: "n_tsplot:plot",     to: "n_imglist:item"}      # (예시) 파형 플롯도 입력에 포함
  - {from: "n_schema:schema",   to: "n_answer:schema"}
  - {from: "n_stats:stats",     to: "n_answer:fields"}
  - {from: "n_ev:evidence",     to: "n_answer:evidence"}
  # 프롬프트 조립
  - {from: "n_kb:text",         to: "n_prompt1:knowledge"}
  - {from: "n_prompt0:text",    to: "n_prompt1:text"}
  - {from: "n_prompt1:text",    to: "n_prompt2:text"}
  - {from: "n_allimgs:out",     to: "n_prompt2:images"}
  # 검증과 누설 차단
  - {from: "n_answer:answer",   to: "n_val:answer"}
  - {from: "n_schema:schema",   to: "n_val:schema"}
  - {from: "n_prompt2:text",    to: "n_leak:prompt"}
  - {from: "n_val:answer",      to: "n_leak:answer"}
  # 샘플 조립과 출력
  - {from: "n_allimgs:out",     to: "n_sample:images"}
  - {from: "n_leak:prompt",     to: "n_sample:prompt"}
  - {from: "n_val:answer",      to: "n_sample:answer"}
  - {from: "n_sample:sample",   to: "n_export:sample"}
  - {from: "n_sample:sample",   to: "n_train:sample"}
  - {from: "n_schema:schema",   to: "n_train:schema"}

materialize:
  boundary: [n_sample]
  format: webdataset
  shard_size_mb: 512

runtime_profile: windows_single_gpu
debug: {debug_output: false, data_storage: {enabled: false}}
```

`n_label`은 어떤 배선에도 쓰이지 않는다. 정답 Text는 `n_stats`/`n_ev`에서 규칙으로 생성되고, `label`은 검증용 메타로만 남기거나 `answer.stepwise`의 `verdict` 슬롯에 연결한다. 후자를 택하면 `sem=label`이 `n_answer`를 거쳐 `answer` 태그로 흐르므로 정상이고, 실수로 `n_prompt*`에 연결하면 G2의 taint 검사가 거부한다(문서 6.4).

### 4.7.3 학습 순서

Projector 정렬 → 전체 미세조정은 그래프가 아니라 `trainer.yaml`의 `stages`로 표현된다(문서 7). 조건 분기·반복을 그래프로 표현하지 않는다는 원칙의 적용점이다.

### 4.7.4 시계열 구간 지목으로 바꾸려면

`p_expert`의 `ref`를 `expert_interval_window@1.0.0`으로, `plugin`을 `ecg_interval_proposer@0.2.0`으로 바꾸고 `subject` 배선을 `n_tsplot:plot` 대신 `n_ts:series`로 옮긴다. **코드 수정 없음, 배선 1개와 파라미터 2개 변경.** 출력 타입은 `Regions{domain=series1d, frame=ts_seconds}`가 되고, 하류의 `image.crop_by_regions`는 타입 불일치로 G1에서 거부되므로 `ts.window` 경로로 바꾸라는 지시가 즉시 뜬다.
