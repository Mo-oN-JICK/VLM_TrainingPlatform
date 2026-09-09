# 5. 외부 전문가 모델 플러그인 인터페이스

목표: **이미지 영역 지목과 시계열 구간 지목이 같은 인터페이스로 들어온다.** 노드 타입은 `expert.propose` 하나이고, 도메인 차이는 플러그인 매니페스트가 선언하는 타입에서만 드러난다.

## 5.1 통일 타입: Regions

두 도메인을 하나의 값 타입으로 흡수한다.

```python
@dataclass(frozen=True)
class Region:
    extent: Box2D | Interval1D
    score: float                 # 0..1, 내림차순 정렬 보장
    label: str | None
    meta: dict[str, JSON]        # 플러그인 고유 부가정보. 하류에서 선택적으로 사용

Box2D      = tuple[float, float, float, float]    # x0,y0,x1,y1  (frame이 단위를 결정)
Interval1D = tuple[float, float]                  # t0,t1        (time_base가 단위를 결정)
```

포트 타입으로서의 `Regions`:

```
Regions{
  domain:    image2d | series1d,     # 어떤 extent를 담는지
  frame:     orig_px | crop_px | norm01        (domain=image2d)
  time_base: ts_index | ts_seconds(hz)         (domain=series1d)
  max_n:     int,
  sorted_by: score | position,
  sem:       {expert_hint} 등
}
```

`domain`이 다르면 **다른 타입**이다. 그래서 이미지용 crop 노드에 시계열 구간을 꽂으면 G1에서 즉시 거부된다. 인터페이스는 하나지만 타입은 갈라진다 — 이것이 "같은 인터페이스로 받되 오배선은 막는다"의 구현이다.

## 5.2 플러그인 프로토콜

```python
class ExpertPlugin(Protocol):
    # ── 정적 선언 (로드 없이 읽힌다. compile/G4가 사용) ──
    id: str                      # "ecg_region_proposer"
    version: str                 # semver
    domain: Literal["image2d", "series1d"]
    accepts: PortType            # 예: Image{u8,HWC,RGB,0-255,frame=orig_px}
    produces: PortType           # 예: Regions{image2d, orig_px, max_n=64}
    deterministic: bool          # False면 seed 고정 + 캐시 키에 seed 포함이 강제된다
    external_call: bool = True   # 물질화 경계 앞 배치 강제
    requires: RuntimeReq         # {vram_mb, torch, cuda, os: [windows, linux], py}

    # ── 수명주기 ──
    def load(self, ctx: PluginCtx) -> None: ...
    def warmup(self, ctx: PluginCtx) -> None: ...
    def unload(self) -> None: ...

    # ── 본체 ──
    def propose(self, batch: list[Value], params: ProposeParams,
                ctx: PluginCtx) -> list[Regions]: ...

    # ── 비용 (G4가 호출. 모델 로드 없이 답해야 한다) ──
    def cost(self, in_shape: Shape, params: ProposeParams) -> ResourceCost:
        """ResourceCost(vram_mb, ms_per_sample, out_bytes_per_sample)"""
```

`ProposeParams`는 노드 파라미터에서 온다: `topk`, `score_threshold`, `device`, `dtype`, `seed`, 그리고 플러그인이 선언한 자유 파라미터(`extra_schema`).

## 5.3 매니페스트와 등록

```yaml
# plugins/experts/ecg_region_proposer/plugin.yaml
kind: ExpertPlugin
id: ecg_region_proposer
version: "0.3.0"
entry: "ecg_region_proposer.plugin:Proposer"     # python entry point
domain: image2d
accepts:  {base: Image, dtype: u8, layout: HWC, colorspace: RGB, value_range: "0-255", frame: orig_px}
produces: {base: Regions, domain: image2d, frame: orig_px, max_n: 64, sorted_by: score, sem: [expert_hint]}
deterministic: true
requires: {vram_mb: 2600, cuda: ">=12.1", os: [windows, linux], torch: ">=2.4"}
weights:
  - {name: default, uri: "file://D:/models/ecg_rpn_v3.safetensors", sha256: "..."}
extra_schema:
  nms_iou: {type: float, default: 0.5, range: [0, 1]}
```

- 등록은 entry point 스캔 + 매니페스트 검증. 매니페스트의 `accepts`/`produces`가 곧 타입 변수 해소의 근거다.
- `graph.lock`에 `plugin_id@version + weights.sha256`이 기록된다. 가중치가 바뀌면 캐시 키가 바뀌어 전처리가 자동 무효화된다.
- `requires.os`에 현재 프로파일(`windows_single_gpu`)이 없으면 G2에서 거부.

## 5.4 두 도메인이 같은 인터페이스를 쓰는 방식

| | 이미지 영역 지목 | 시계열 구간 지목 |
|---|---|---|
| 노드 타입 | `expert.propose` | `expert.propose` (동일) |
| 플러그인 | `ecg_region_proposer@0.3.0` | `ecg_interval_proposer@0.2.0` |
| `accepts` | `Image{u8,HWC,RGB,0-255,orig_px}` | `TimeSeries{f32,TC,hz=500,ts_seconds}` |
| `produces` | `Regions{image2d, orig_px}` | `Regions{series1d, ts_seconds}` |
| 하류 소비 | `image.crop_by_regions` | `ts.window` |
| Procedure | `expert_region_crop@1.2.0` | `expert_interval_window@1.0.0` |

플러그인 교체 = `params.plugin` 문자열 변경. 도메인 교체 = 그 위에 배선 1개 + Procedure ref 변경. 노드 코드는 그대로다.

`subject` 포트는 제네릭 `T where T in {Image, TimeSeries}`이고, compile 시 선택된 플러그인의 `accepts`와 단일화되어 확정된다. 플러그인이 미지정이면 `?T` 미해결로 G2 실패.

## 5.5 실행 격리 (단일 GPU 전제)

전문가 모델은 학습과 같은 GPU를 두고 경쟁한다. 그래서:

1. **별도 워커 프로세스**에서만 실행한다. 학습 프로세스와 절대 같은 프로세스에 올리지 않는다.
2. **물질화 단계에서만 산다.** `materialize` 실행 시작 시 `load()`, 완료 시 `unload()` + 프로세스 종료. 학습 프로세스는 그 뒤에 시작된다. VRAM 파편화를 남기지 않기 위해 프로세스 자체를 버린다.
3. G2는 `external_call: true` 노드가 물질화 경계 **앞**에 있는지 검사한다. 경계 뒤에 있으면 컴파일 실패하며, 이유로 "학습 루프 안에서 전문가 모델이 VRAM을 요구해 OOM 또는 심한 감속"을 표시한다.
4. `cost()`가 보고한 `vram_mb`는 물질화 단계의 예산 검사에 쓰인다(학습 단계 예산과 별개로 계산).

## 5.6 결정성 계약

| `deterministic` | 엔진의 처리 |
|---|---|
| `true` | 일반 Processing 노드와 동일. 출력 캐시 재사용 |
| `false` | `seed` 파라미터가 필수가 되고 캐시 키에 포함된다. dry-run의 결정성 감사는 같은 seed로 2회 실행해 비교한다. seed 미지정이면 G2 실패 |

샘플링 기반 모델(생성형 캡셔너 등)을 허용하되, 재현 불가능한 데이터셋이 조용히 만들어지는 경로는 막는다.

## 5.7 다른 플러그인 종류와의 관계

같은 등록 메커니즘을 공유하는 플러그인 계열은 셋이다.

| 계열 | 프로토콜 | 쓰이는 곳 |
|---|---|---|
| `ExpertPlugin` | 위 5.2 | `expert.*` 노드 |
| `BackboneAdapter` | 문서 7.5 | `train.vlm_trainer` — 백본 교체 |
| `StorageBackend` | 문서 8.7 | 캐시·물질화 저장소 교체(로컬 디스크 기본) |

셋 다 `plugin.yaml` + entry point + `requires` 검사 + lock 기록이라는 같은 규약을 따른다. 새 플러그인 종류를 추가할 때 새 메커니즘을 발명하지 않는다.
