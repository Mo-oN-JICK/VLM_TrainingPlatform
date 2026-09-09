# 13. Parameter Recipe 시스템

Mech-Vision의 Parameter Recipe를 그대로 가져오되 목적을 확장한다. 원래 목적은 "로직은 같고 파라미터만 다른 프로젝트를 중복 구축하지 않기" `[문서확인]`, 우리는 여기에 **실험 스윕**을 얹는다. 그래프가 같으면 Recipe만 다르게 여러 실험이 돌아야 한다.

## 13.1 노드의 화이트리스트 선언

레시피가 아무 파라미터나 덮으면 타입과 예산이 소리 없이 깨진다. 무엇을 덮어도 되는지는 노드가 안다.

```python
@register(
    type="adapt.image_resize", version="1.0.0", kind=NodeKind.PROCESSING,
    params=ResizeParams,
    recipe_overridable=["size", "mode", "keep_aspect"],   # 이 셋만 레시피가 덮을 수 있다
    type_affecting=["size"],                              # 덮으면 타입 재검사 필요
    ...
)
```

| 선언 | 의미 |
|---|---|
| `recipe_overridable` | 레시피 오버레이가 건드릴 수 있는 파라미터 화이트리스트. 여기 없는 키를 레시피가 지정하면 **compile 즉시 거부** |
| `type_affecting` | 오버라이드 시 포트 타입이 바뀌는 파라미터. 지정되면 오버레이 적용 후 G1·G2·G4를 전부 다시 돌린다 |

기본값은 **빈 화이트리스트**다. 노드 작성자가 명시적으로 열어야 열린다. 닫힌 쪽이 안전한 기본값이기 때문이다.

Trainer 노드는 예외적으로 넓게 연다(`stages[*].optimizer.lr`, `stages[*].lora.r`, `batch.*`, `quantization.mode` 등). 이것들이 스윕의 주 대상이고, 전부 `type_affecting`은 아니지만 **전부 budget_affecting**이라 오버레이 후 G4는 항상 재실행된다.

## 13.2 Recipe Spec 스키마

```yaml
# projects/02_triad_ecg/recipes.yaml
kind: ParameterRecipes
spec_version: 1
project: "02_triad_ecg"
active: 2                      # 현재 활성 레시피 번호. null이면 프로젝트 기본값
customized: false              # 프로젝트를 직접 수정하면 true로 전환 (Mech-Vision 모방)

display:                       # 파라미터 표시 이름 커스텀 (Mech-Vision 모방)
  "n_train.stages[1].lora.r":            "LoRA rank"
  "n_train.stages[1].optimizer.lr":      "미세조정 학습률"
  "p_expert.score_threshold":            "전문가 점수 임계"

recipes:
  - id: 1
    name: "baseline"
    note: "LoRA r=16, 임계 0.35"
    overrides:
      p_expert.score_threshold: 0.35
      n_train.stages[1].lora.r: 16
      n_train.stages[1].optimizer.lr: 1.0e-4

  - id: 2
    name: "high_rank"
    overrides:
      p_expert.score_threshold: 0.35
      n_train.stages[1].lora.r: 64
      n_train.stages[1].optimizer.lr: 5.0e-5

  - id: 3
    name: "tight_expert"
    overrides:
      p_expert.score_threshold: 0.6
      p_expert.topk: 2
      n_train.stages[1].lora.r: 16
      n_train.stages[1].optimizer.lr: 1.0e-4
```

| 필드 | 규칙 |
|---|---|
| `id` | 1~99 정수. Mech-Vision의 레시피 번호 규약을 그대로 따른다 `[문서확인]` |
| `name` | 사람이 읽는 이름. run 디렉터리 명명에 사용 |
| `overrides` | `노드id.파라미터경로: 값`. 경로는 점 표기 + 배열 인덱스. **값만** 가능 |
| `display` | 파라미터 표시 이름. UI의 Recipe 편집기와 Node Parameters 탭에서 사용 |
| `active` | 활성 레시피. CLI `--recipe N`이 이를 덮어쓴다 |
| `customized` | 프로젝트 파라미터를 직접 수정하면 `true`가 되고 `active`는 무효화된다 `[문서확인]` |

**금지 사항**(전부 compile 시 거부):

- 노드 추가·삭제, 배선 추가·삭제·변경
- `sample_space`, `materialize.boundary`, `runtime_profile` 변경
- 화이트리스트 밖 파라미터 오버라이드
- 존재하지 않는 노드 id 참조

## 13.3 Project Spec과의 관계

```
project.yaml   구조(노드·배선·Procedure) + 파라미터 기본값     ← 그래프의 정체성
recipes.yaml   값 오버레이만                                   ← 실험의 변주
graph.lock     해소된 버전·해시                                 ← 재현의 기준
```

적용 순서(문서 1.2의 상속 규칙 끝단):

```
solution.defaults → project.defaults → procedure exposed params
  → node params → recipe overlay
```

compile 파이프라인에서 오버레이는 **3단계(inline 이전)** 에 적용된다. Procedure 내부 노드에도 오버라이드를 걸 수 있어야 하므로, 경로는 인라인 후의 네임스페이스 경로(`p_expert/n_topk.k`)와 Procedure 노출 파라미터 경로(`p_expert.topk`) 둘 다 허용한다. 후자를 권장한다(캡슐화 유지).

**canonical spec에는 오버레이가 적용된 최종 값이 기록된다.** 즉 run 스냅샷만 보면 어떤 레시피였는지와 무관하게 실험이 완전히 재현된다. 레시피 번호는 메타로만 남는다.

## 13.4 스윕 전개 규칙

```yaml
sweeps:
  - name: "lora_rank_x_prompt"
    id_range: [10, 99]           # 전개된 레시피에 부여할 번호 범위
    base: 1                      # 이 레시피에서 출발해 축만 바꾼다
    strategy: grid               # grid | list | random
    axes:
      "n_train.stages[1].lora.r":       [16, 32, 64]
      "n_prompt0.template":             ["tmpl_a", "tmpl_b"]
      "p_expert.score_threshold":       [0.35, 0.6]
    naming: "r{lora.r}_{template}_th{score_threshold}"
    # strategy: random 인 경우
    # samples: 8
    # seed: 20260909
```

전개 규칙:

1. `grid`는 축의 데카르트 곱. 위 예시는 3 × 2 × 2 = 12개.
2. 번호는 `id_range` 앞에서부터 순서대로 부여하고, 이미 사용 중인 번호는 건너뛴다. 범위가 모자라면 전개 실패.
3. 전개 결과는 `recipes.lock.yaml`에 **구체 레시피 목록으로 고정**된다. 스윕 정의를 나중에 바꿔도 이미 실행한 번호의 의미가 변하지 않는다.
4. `list` 전략은 축 값들을 zip한다(길이가 다르면 실패).
5. `random`은 seed 고정이며, 뽑힌 조합이 lock에 기록되어 재현된다.

**전개 시점에 전 레시피의 예산 검사(G4)를 먼저 돌린다.** 단일 GPU에서 20개 스윕 중 7개가 OOM으로 죽는 것을 다음 날 아침에 발견하는 대신, 전개 즉시 목록으로 안다.

```
$ vlmt recipe expand project.yaml --sweep lora_rank_x_prompt

전개: 12개 (번호 10~21)
  예산 통과: 8개  (10,11,12,13,14,16,17,18)
  예산 초과: 4개  (15,19,20,21)
    15: r=64 + tmpl_b(+380 토큰) → 24.9 GB > 22.5 GB, 초과 기여 1위 활성화(+2.1 GB)
        해결 후보: batch.per_device 1 유지 + grad_accum 32, 또는 max_len 3584
  → 초과 4개는 큐에 넣지 않았습니다. `--include-over-budget`로 강제 포함 가능(권장하지 않음)
```

## 13.5 CLI 경로

```
vlmt recipe list    projects/02_triad_ecg/project.yaml
vlmt recipe show    projects/02_triad_ecg/project.yaml --recipe 2
vlmt recipe diff    projects/02_triad_ecg/project.yaml --recipe 1 --recipe 2
vlmt recipe expand  projects/02_triad_ecg/project.yaml --sweep lora_rank_x_prompt
vlmt recipe set-active projects/02_triad_ecg/project.yaml --recipe 2

vlmt run   projects/02_triad_ecg/project.yaml --recipe 2
vlmt sweep projects/02_triad_ecg/project.yaml --recipes 10-21 [--stop-on-fail]
```

`sweep` 실행 모델(단일 4090 전제):

1. 대상 레시피 전부에 대해 compile + dry-run + budget을 **먼저** 돌린다. 하나라도 게이트에 걸리면 그 레시피는 큐에서 제외되고 목록으로 보고된다.
2. **물질화 캐시를 공유한다.** 레시피가 전처리 파라미터를 건드리지 않으면(예: LoRA rank만 다르면) 캐시 키가 동일하므로 물질화가 한 번만 일어나고 나머지는 shard를 재사용한다. 이것이 스윕의 실질 비용을 결정한다.
3. 학습은 **순차 큐**로 돈다. GPU 배타 락(문서 8.10)이 동시 실행을 구조적으로 막는다.
4. 각 run은 `runs/<timestamp>_<recipe_id>_<name>/`에 기록되고, 스냅샷 스펙에 오버레이 적용 후 최종값이 들어간다.
5. `--stop-on-fail` 없이는 한 레시피의 실패가 큐를 멈추지 않는다.

## 13.6 UI 경로

`[문서확인: Parameter Recipe는 Project Assistant에서 진입, Parameter Recipe Editor에서 Add Recipe, 새 레시피는 현재 프로젝트 값을 기본 상속]`

- **Project Assistant 탭 → Parameter Recipe** 진입점. Mech-Vision과 동일 위치.
- Recipe 편집기: 레시피 추가/삭제/복제, 값 편집, "현재 프로젝트 값으로 갱신", "다른 레시피로 값 복사", 표시 이름 편집.
- Project Toolbar에 활성 레시피 드롭다운. 전환하면 Node Parameters 탭의 값이 즉시 바뀌고, `type_affecting` 파라미터가 포함되어 있으면 배선 타입이 재검사된다.
- 프로젝트 파라미터를 직접 수정하면 드롭다운이 **Customized**로 바뀐다. `[문서확인]`
- 스윕은 편집기 안의 별도 탭에서 축을 정의하고 `expand`를 눌러 전개한다. 전개 결과는 예산 통과/초과가 색으로 구분된 목록으로 표시된다.
