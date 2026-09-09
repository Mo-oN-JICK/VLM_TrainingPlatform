# 6. Output 설계 모듈 — 정답 Text 스키마

"Output 설계"는 그래프의 Output 분류와 다른 개념이다. 여기서는 **모델이 내야 할 출력의 형태를 정의하고 강제하는 모듈**을 말한다. 학습 시 정답 Text의 형태이자, 추론 시 파서의 계약이다.

## 6.1 Answer Schema

```yaml
# schemas/triad_answer.yaml
kind: AnswerSchema
version: "1.0.0"
id: triad_stepwise

render: tagged          # json | tagged | markdown
locale: ko

steps:
  - id: trend
    label: "추세"
    type: enum
    values: [rising, falling, flat, unstable]
    evidence: required            # 근거 문장 필수
  - id: periodicity
    label: "주기성"
    type: object
    fields:
      present: {type: bool}
      period_s: {type: float, range: [0.2, 5.0], required_if: "present == true"}
    evidence: required
  - id: spike
    label: "스파이크"
    type: object
    fields:
      found: {type: bool}
      count: {type: int, range: [0, 200]}
      max_z: {type: float}
    evidence: required
  - id: verdict
    label: "최종 판정"
    type: enum
    values: [normal, abnormal, inconclusive]
    depends_on: [trend, periodicity, spike]
    evidence: required
    rules:                        # 단계 간 일관성 규칙
      - "spike.found == true and spike.max_z >= 5 => verdict != normal"
      - "trend == unstable => verdict != normal"

constraints:
  max_tokens: 320
  step_order: strict              # 선언 순서를 벗어나면 위반
  no_extra_steps: true
  vocabulary: closed              # enum 밖 토큰 금지

templates:
  tagged: |
    <trend>{trend} — {trend.evidence}</trend>
    <periodicity>{periodicity.present} (주기 {periodicity.period_s}s) — {periodicity.evidence}</periodicity>
    <spike>{spike.found} (n={spike.count}, max_z={spike.max_z}) — {spike.evidence}</spike>
    <verdict>{verdict} — {verdict.evidence}</verdict>
  json: auto                      # steps 정의에서 자동 생성
```

**렌더러와 파서는 같은 정의에서 생성된다.** 학습 때 문자열을 만든 규칙이 추론 때 문자열을 읽는 규칙과 다를 수 없도록, 둘 다 `templates`에서 파생시킨다.

## 6.2 위반 샘플 검증

`answer.validate` 노드가 수행한다.

| 검사 | 예시 위반 |
|---|---|
| 파싱 | 태그 누락, 중첩 오류, JSON 문법 오류 |
| 단계 완전성 | `spike` 단계 없음 |
| 단계 순서 | `verdict`가 `spike`보다 먼저 |
| 값 도메인 | `trend: "약간 상승"` (enum 밖) |
| 범위 | `period_s: 42.0` |
| 조건부 필수 | `present: true`인데 `period_s` 없음 |
| 근거 존재 | `evidence: required`인 단계에 근거 문장 없음 또는 8자 미만 |
| 단계 간 규칙 | `max_z: 7.1`인데 `verdict: normal` |
| 길이 | 렌더 결과가 `max_tokens` 초과 |

정책:

```yaml
on_violation: quarantine     # quarantine(기본) | drop | fail
```

- `quarantine`: 샘플을 학습에서 제외하고 `runs/<run_id>/quarantine/<rule>/<sample_id>.json`에 원문·위반 규칙·근거와 함께 격리. 리포트에 규칙별 집계.
- `drop`: 조용히 제외(집계만).
- `fail`: 한 건이라도 위반이면 실행 중단.

**G3 dry-run은 별도로 `fail_ratio_threshold`(기본 2%)를 본다.** 샘플 100건을 검증해 위반율이 임계를 넘으면 학습을 시작하지 않는다. 정답 생성 규칙이 잘못되어 전량이 쓰레기인 상태로 GPU를 태우는 것이 가장 흔한 낭비이기 때문이다.

## 6.3 단계별 근거 문장 자동 생성

`answer.evidence_rules` 노드. 결정적 규칙 기반이 기본이다.

```yaml
# schemas/triad_evidence.yaml
kind: EvidenceRules
version: "1.0.0"
for_schema: triad_stepwise

trend:
  - when: "trend_slope > 0.05"     then: "구간 전반에 걸쳐 기저선이 분당 {trend_slope:.2f} 단위로 상승합니다."
  - when: "trend_slope < -0.05"    then: "기저선이 분당 {trend_slope:.2f} 단위로 하강합니다."
  - default:                        "기저선 변동이 {trend_slope:.2f}로 유의미하지 않습니다."
periodicity:
  - when: "acf_peak >= 0.6"        then: "자기상관 최대값 {acf_peak:.2f}에서 주기 {acf_period_s:.2f}초가 관찰됩니다."
  - default:                        "자기상관 최대값이 {acf_peak:.2f}로 뚜렷한 주기가 없습니다."
spike:
  - when: "spike_count > 0"        then: "z>{z_thresh}를 넘는 스파이크가 {spike_count}회, 최대 z={spike_max_z:.1f}입니다."
  - when: "region_count > 0"       then: "전문가 모델이 {region_count}개 구간(최고 점수 {region_top_score:.2f})을 지목했습니다."
  - default:                        "임계 이상 스파이크가 없습니다."
verdict:
  - compose: [trend, periodicity, spike]     # 앞 단계 근거를 요약해 결론 문장 생성
    template: "위 세 관찰을 종합하면 {verdict}로 판단합니다."
```

- 규칙 평가는 순수하다. 입력은 `ts.stats`의 수치와 `Regions`의 통계뿐이고, 슬롯은 수치 포맷팅만 한다.
- LLM으로 근거 문장을 윤문하는 것은 **옵션 플러그인**(`answer.llm_polish`)으로 두되, `deterministic: false`이므로 seed 고정 + 캐시 키 포함이 강제되고 물질화 경계 앞에 배치된다. 기본 경로에는 넣지 않는다.
- 규칙 파일 자체가 `graph.lock`에 해시로 기록되어, 문구를 바꾸면 전처리 캐시가 무효화된다.

## 6.4 정답 누설 차단

두 층으로 막는다.

### 층 1 — 정적 taint 도달성 (G2, compile 시점)

`semantic` 태그는 하류로 전파된다. `sem=label` 또는 `sem=answer`가 붙은 값이 `sem=prompt`를 산출하는 노드에 **도달 가능하면** 컴파일을 거부한다.

```
검사: reachable(sem∈{label, answer}, any node producing sem=prompt) == ∅
```

예외는 명시적 차단 노드를 거친 경로뿐이다. `answer.leakage_guard`는 자기 출력에서 `label` taint를 제거하는 유일한 노드이며, 그 대가로 실제 문자열 검사를 수행한다.

이 검사는 **공짜에 가깝고 가장 비싼 실수를 막는다.** 정답이 프롬프트에 섞인 실험은 완주하고, 점수가 높게 나오고, 그 사실을 아무도 눈치채지 못한다.

### 층 2 — 문자열 검사 (`answer.leakage_guard`, 샘플 단위)

```yaml
mode: normalized        # exact | normalized(공백/대소문자/조사 정규화) | regex
vocab: [normal, abnormal, 정상, 비정상, inconclusive]
window: 0               # 정답 토큰이 프롬프트에 그대로 등장하는지
on_leak: fail           # fail | mask
```

- `verdict` enum 값과 그 한국어 표기, 근거 문장의 결정적 어구가 프롬프트 안에 있는지 검사.
- 도메인 지식 텍스트(`sem=domain_knowledge`)는 규칙을 담고 있어 `normal/abnormal` 같은 단어를 정당하게 포함할 수 있다. 그래서 검사 대상에서 **지식 블록을 제외**하는 `exclude_sections` 파라미터를 둔다. 제외 구간은 리포트에 항상 표시되어 사용자가 눈으로 확인한다.

## 6.5 추론 시 재사용 — inference contract

학습에서 정의한 스키마와 전처리를 추론에서 다시 손으로 짜면 그 즉시 어긋난다. 그래서 Trainer(Output)가 체크포인트 옆에 **추론 계약**을 함께 기록한다.

```
runs/<run_id>/ckpt/stage2/
  model/                         가중치 또는 어댑터
  inference_contract.json        <- 아래 내용
  inference_graph.yaml           <- 추론용으로 잘라낸 그래프 스펙
  answer_schema.yaml             학습에 쓴 스키마 사본
```

`inference_contract.json`:

```json
{
  "spec_hash": "b3:9f21…",
  "backbone": {"id": "…", "adapter": "…@1.1.0", "dtype": "bf16", "quantization": {"mode": "nf4"}},
  "image_pipeline": [
    {"type": "adapt.image_resize@1.0.0", "params": {"size": [896,896], "mode": "bicubic", "keep_aspect": true}},
    {"type": "adapt.colorspace@1.0.0",   "params": {"to": "RGB"}}
  ],
  "image_slots": {"placeholder": "<image>", "count": 4, "policy": "prepend"},
  "prompt_template": "…렌더된 템플릿, 슬롯 이름 포함…",
  "knowledge_block": {"source": "knowledge/ecg_rules.md", "sha256": "…", "position": "before"},
  "tokenizer": {"id": "…", "add_special": true, "max_len": 4096},
  "answer_schema": {"id": "triad_stepwise", "version": "1.0.0", "render": "tagged"},
  "parser": {"kind": "generated_from_schema", "stop": ["</verdict>"]},
  "expected_vision_tokens_per_sample": 2304
}
```

### `inference_graph.yaml` 생성 규칙

학습 그래프에서 **정답 생성 경로를 잘라내고 프롬프트 경로만 남긴 서브그래프**를 자동 추출한다.

```
keep   = ancestors(n_prompt2) ∪ ancestors(images 입력)          # 프롬프트·이미지 조립 전부
drop   = ancestors(answer 경로) \ keep                          # ts.stats, evidence_rules, answer.* 등
        + 모든 Output 노드
add    = infer.request(Input)  … 추론 시 샘플 주입점
        + infer.generate(Output) … 모델 호출과 파싱
```

즉 추론 파이프라인도 같은 노드 타입, 같은 타입 검사, 같은 Adapter 체인을 쓴다. 전처리가 학습과 다를 수 없는 구조다.

`vlmt infer-graph runs/<run_id> -o infer.yaml`로 꺼내고, `vlmt preview --spec infer.yaml`로 프롬프트 렌더 결과를 눈으로 확인할 수 있다.

> 범위 주의: 이 플랫폼은 학습 도구다. `infer.generate`는 **추론 계약이 실제로 재현되는지 확인하는 검증 수단**으로만 존재하며, 서빙·배포·최종 사용자 워크플로는 이 플랫폼의 범위 밖이다.

## 6.6 스키마 버전과 마이그레이션

- `AnswerSchema`는 semver를 갖고 `graph.lock`에 해시로 고정된다.
- minor 변경(문구, 근거 규칙)은 전처리 캐시만 무효화한다.
- major 변경(단계 추가/삭제, enum 변경)은 기존 체크포인트의 `inference_contract`와 불일치하므로, 그 체크포인트에서의 재개(`train.resume`)를 거부한다.
