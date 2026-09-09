# 2. 노드 인터페이스와 4중 게이트

## 2.1 왜 Mech-Vision보다 엄격한가

Mech-Vision은 포트에 타입 이름표를 붙여 검증한다(`<Image>`, `<NumberList/Roi>`, `<StringList>` 등). `[이미지확인]`
비전 파이프라인에서는 그것으로 충분하다. 오배선을 하면 몇 초 뒤 시각화 출력이 틀린 것을 사람이 눈으로 본다.

우리는 다르다. 잘못된 정규화, 잘못된 좌표계, 잘못된 채널 순서는 **끝까지 조용히 학습되고**, 24시간 뒤 낮은 점수만 남긴다. 최악은 실패조차 하지 않는 오배선이다. 그래서:

> 검증이 지나쳐서 잃는 것은 사용자의 몇 초, 검증이 모자라서 잃는 것은 GPU 수십 시간과 그 사이 막힌 다른 모든 실험. 이 비대칭이 아래 모든 설계의 근거다.

## 2.2 포트 타입 구조체

포트 타입은 이름이 아니라 필드의 곱이다.

```python
@dataclass(frozen=True)
class PortType:
    base: BaseKind            # Image ImageList TimeSeries Text TextList Tokens Embedding
                              # Regions Mask Table Schema Sample Report Model
    dtype: DType | Var        # uint8 float32 bfloat16 int64 str bool
    shape: tuple[Dim, ...]    # Dim = int | SymbolVar("H") | ANY
    layout: Layout | Var      # HWC CHW | TC CT | None
    value_range: Range | Var  # U8_0_255 | UNIT_0_1 | SIGNED_M1_1 | ZSCORE | RAW
    norm: NormSpec | Var      # none | scale01 | imagenet | custom(mean,std) | per_channel_z
    colorspace: Color | Var   # RGB BGR GRAY | None
    frame: Frame | Var        # 좌표계 기준: orig_px | crop_px | norm01 | tile_px
                              # 시간축 기준: ts_index | ts_seconds
    time_base: TimeBase|None  # hz, t0_policy(sample_start|absolute), 길이 단위
    semantic: frozenset[str]  # raw_image roi_crop expert_hint domain_knowledge
                              # prompt answer label evidence
    optional: bool
    list_of: ListSpec | None  # {min_n, max_n, homogeneous}
```

표기 축약(문서 전체에서 사용):

```
Image{u8, HWC, RGB, 0-255, frame=orig_px, shape=[?H,?W,3], sem=raw_image}
Image{f32, CHW, RGB, 0-1, norm=imagenet, shape=[3,448,448], sem=roi_crop}
TimeSeries{f32, TC, shape=[?T,12], hz=500, frame=ts_seconds, sem=raw_signal}
Regions{domain=image2d, frame=orig_px, max_n=8, sem=expert_hint}
Text{sem=prompt, max_tokens=?K}
Sample{images=[...], text=..., answer=...}
```

`Image` 두 개라도 하나가 `u8/HWC/0-255`이고 다른 하나가 `f32/CHW/norm=imagenet`이면 **서로 다른 타입**이고 연결되지 않는다.

## 2.3 호환성 판정 규칙

`connect(src_out: PortType, dst_in: PortType) -> Ok | Reject(reason)`

1. **base kind는 정확히 같아야 한다.** 서브타입·상속·업캐스트 없음. `Image`에서 `ImageList`로의 자동 승격도 없음(`list.wrap` 노드로 명시).
2. **스칼라 필드(dtype, layout, value_range, norm, colorspace, frame, time_base)는 정확히 같아야 한다.** 예외는 목적지가 그 필드를 `ANY`로 선언한 경우뿐이며, `ANY`는 노드가 그 필드에 **무관함을 계약으로 선언**한 것으로 취급한다(예: `io.dataset_export`는 layout 무관).
3. **shape는 단일화(unification)한다.** 구체 정수끼리 다르면 거부. 심볼 변수는 바인딩되고, 같은 스코프에서 같은 변수는 같은 값이어야 한다. rank가 다르면 거부.
4. **semantic 태그**: 목적지가 `requires_sem`을 선언하면 소스의 `semantic`이 이를 포함해야 한다. 소스에만 있는 여분 태그는 통과하며 하류로 전파된다(누설 검사에 사용).
5. **optional**: 목적지가 optional이면 미연결 허용. optional 소스를 non-optional 목적지에 연결하면 거부(`optional.unwrap_or` 노드로 명시).
6. **list**: `list_of`가 있는 쪽과 없는 쪽은 다른 타입이다. 브로드캐스트 없음. `list.map(Procedure)`로 명시.
7. **팬아웃 허용, 팬인 금지**: 하나의 출력 포트는 여러 입력에 연결 가능. 하나의 입력 포트에는 정확히 하나의 배선.

거부는 **이진**이다. 경고 등급, `--allow-unsafe-cast`, strict 해제 스위치는 만들지 않는다.

### 암묵 변환 금지

리사이즈, 정규화, 채널 순서 변경, 색공간 변환, dtype 캐스팅, 좌표계 변환, 토크나이즈는 전부 **Adapter 노드**로만 표현된다(문서 3의 `adapt.*` 카테고리).
편집기는 거부 시 "이 Adapter를 넣으면 연결됩니다"를 제안하되 **자동으로 삽입하지 않는다.** 삽입은 사용자의 명시적 행위여야 스펙에 기록되고, 스펙에 기록되어야 추론 시 같은 전처리를 재현할 수 있다(문서 6의 inference contract).

### 제네릭

```python
T = TypeVar("T", bound=BaseKind.in_({Image, TimeSeries}))
H, W = DimVar("H"), DimVar("W")
```

- 노드 시그니처에 타입 변수와 제약(`where`)을 쓸 수 있다.
- 플러그인 의존 타입도 변수다. 예: `expert.propose`의 출력 `Regions{domain=?D}`는 선택된 플러그인 매니페스트에서 `?D`가 결정된다.
- **compile 종료 시점에 모든 변수는 구체 타입으로 확정되어야 한다.** 미해결 변수가 하나라도 남으면 G2 실패.

## 2.4 Input / Processing / Output 3분류

```
   (입력 포트 없음)              +---------------+
   +---------------+            |  v v  입력    |
   |               |            +---------------+
   |     INPUT     |            |  PROCESSING   |
   +---------------+            +---------------+
   |  v v  출력    |            |  v v  출력    |
   +---------------+            +---------------+

   +---------------+
   |  v v  입력    |
   +---------------+
   |    OUTPUT     |     (출력 포트 없음 — 여기서 그래프가 끝난다)
   +---------------+
```

| 분류 | 포트 형상(강제) | 캐시 | 미리보기 | 부작용 | 실패 격리 |
|---|---|---|---|---|---|
| **Input** | `len(inputs)==0`, `len(outputs)>=1` | 캐시 키에 **데이터 지문**(정규화 경로 + 내용 해시 + mtime + size) 포함 | 허용 | 읽기 전용, 쓰기 금지 | 샘플 단위 격리(파일 손상은 그 샘플만 quarantine) |
| **Processing** | `len(inputs)>=1`, `len(outputs)>=1` | 캐시 대상. 키 = 타입+버전+정규화 파라미터+상류 키 | 허용(노드 단위) | **금지**, 순수 함수 | 노드+샘플 단위 격리, 재시도 안전 |
| **Output** | `len(inputs)>=1`, `len(outputs)==0` | **캐시 금지** | 금지(대신 dry 요약) | 유일한 허용 지점: 체크포인트 기록, dataset export, 학습 실행, Data Storage | 격리 불가. 실패 시 실행 중단 + 저널에 마지막 커밋 지점 기록 |

3분류는 노드 등록 시 선언하고, 레지스트리가 포트 형상과의 정합을 **등록 시점에** 검사한다(불일치는 임포트 실패).

### Processing 순수성의 강제 방법

선언만으로는 계약이 지켜지지 않으므로 엔진이 다음을 강제한다.

1. **샌드박스**: Processing 노드 실행 시 쓰기 가능 경로는 엔진이 준 스크래치 디렉터리뿐. 그 외 경로 쓰기는 감사되어 예외 발생.
2. **네트워크 차단(기본값)**: Processing 노드는 소켓 생성 불가. 네트워크가 필요하면 Input이거나 Output이거나, 전문가 모델처럼 `external_call: true`를 선언하고 물질화 경계 앞에 강제 배치된다.
3. **난수**: 전역 RNG 접근 금지. 시드는 `ctx.rng(node_id, sample_key)`로만 얻는다. 같은 입력이면 같은 난수.
4. **시계/환경**: `ctx`가 제공하는 고정 시각만 접근 가능.
5. **결정성 감사**(dry-run에서 기본 on): 같은 샘플을 2회 실행해 출력 해시를 비교. 불일치는 계약 위반으로 dry-run 실패.

Trainer 노드는 Output이다. 학습 실행이 프로젝트의 종결점이다.

## 2.5 노드 등록

```python
@register(
    type="image.crop_by_regions",
    version="1.0.0",
    category="2D General Processing",       # Node Library 카테고리 (문서 3)
    kind=NodeKind.PROCESSING,               # Input | Processing | Output
    inputs={
        "image":   Port(Image[u8, HWC, RGB, R_0_255, frame=ORIG_PX, shape=(H, W, 3)]),
        "regions": Port(Regions[domain="image2d", frame=ORIG_PX]),
    },
    outputs={
        "crops":   Port(ImageList[u8, HWC, RGB, R_0_255, frame=CROP_PX,
                                  shape=(DimVar("ch"), DimVar("cw"), 3)],
                        list_of=ListSpec(1, 16)),
    },
    params=CropParams,                      # dataclass/pydantic 스키마, JSON 직렬화 가능
    recipe_overridable=["padding_ratio", "max_n"],   # 문서 13
    type_affecting=["square_pad", "out_size"],       # 오버라이드 시 타입 재검사 필요
    preview="image_grid",                   # 문서 8의 미리보기 렌더러 키
    cost_model=CropCost,                    # 샘플당 산출 크기·시간 추정 (G4)
    doc=NodeDoc(summary=..., scenario=..., ports=...),   # Node Quick Info 탭 (문서 12)
)
class CropByRegions(Node):
    def run(self, ctx: RunCtx, image, regions, params) -> dict: ...
    def infer_types(self, inputs: dict[str, PortType], params) -> dict[str, PortType]:
        """심볼 dim 해소. 기본 구현은 시그니처를 그대로 반환한다."""
```

- 등록은 **선언이 곧 계약**이다. `infer_types`가 반환한 타입과 dry-run 실측이 다르면 G3 실패.
- 서드파티 노드는 entry point(`vlm_trainer.nodes`)로 등록되며 같은 검사를 받는다.
- 노드 타입 이름은 `카테고리접두.동작` 소문자 스네이크, 버전은 semver.

## 2.6 노드가 지켜야 할 계약

| # | 계약 | 위반 시 |
|---|---|---|
| C1 | 선언한 포트 타입과 실제 산출물이 일치한다 | G3 실패, 노드·포트 지목 |
| C2 | Processing은 순수하다(2.4의 강제 항목 전부) | 등록 거부 또는 dry-run 실패 |
| C3 | 파라미터는 JSON 직렬화 가능하고 스키마로 검증된다 | 등록 거부 |
| C4 | Input은 `fingerprint()`를 구현한다 | 등록 거부 |
| C5 | Output은 `writes` 목록과 `commit()`/`rollback()`을 구현한다(부분 기록 방지) | 등록 거부 |
| C6 | `cost_model()`이 샘플당 출력 바이트·비전 토큰 기여·실행 시간을 추정한다 | G4에서 "추정 불가" 표시 후 보수적 상한 적용 |
| C7 | 실패는 `NodeError(node_id, port, sample_key, cause, hint)` 타입으로만 던진다 | 격리 실패, 전체 실행 중단 |
| C8 | `preview()`는 순수하고 원본 데이터를 수정하지 않는다 | 미리보기 비활성화 |
| C9 | 노드는 자기 상류를 알지 못한다(전역 그래프 접근 금지) | 등록 거부 |

## 2.7 4중 게이트

넷을 모두 통과해야 Trainer가 GPU를 잡는다.

```
G1 편집 시점        G2 compile         G3 dry-run          G4 자원 예산
배선 단위 즉시      그래프 전체        실제 샘플 1건       정적 산정
   |                   |                  |                   |
   +-- 연결 거부       +-- 컴파일 실패    +-- 실행 중단       +-- 학습 시작 거부
```

### 게이트별 검사 항목과, 걸리지 않았다면 언제 어디서 터졌을지

| 게이트 | 잡는 오류 | 지목 대상 | 이 게이트가 없었다면 |
|---|---|---|---|
| **G1 정적(편집)** | dtype 불일치 (`f32 CHW`를 `u8 HWC` 기대 포트에) | 배선 양끝 포트 | 데이터로더 collate에서 첫 배치 크래시. 세팅 30분 후 즉사 |
| G1 | value_range/norm 불일치 (정규화된 텐서를 다시 정규화) | 배선 + 하류 Adapter | **아무것도 터지지 않는다.** 24시간 학습 후 점수만 낮음. 원인 추적 사실상 불가. 가장 비싼 오류 |
| G1 | 좌표계 불일치 (`crop_px` 박스를 `orig_px` 기대 노드에) | 배선 | 전 샘플이 엉뚱한 영역을 crop. 학습은 완주하고 모델만 쓰레기 |
| G1 | 시간축 기준 불일치 (`ts_index`와 `ts_seconds` 혼용, hz 불일치) | 배선 | 전문가가 지목한 구간이 실제와 어긋남. 조용한 라벨 오염 |
| G1 | list와 scalar 혼선 (`ImageList`를 `Image` 포트에) | 배선 | 첫 샘플에서 TypeError, 혹은 리스트 첫 원소만 조용히 사용 |
| G1 | semantic 위반 (`sem=label` Text를 prompt 조립 노드에) | 배선 | 정답 누설. 검증 점수가 비현실적으로 높고 실제로는 무의미 |
| **G2 compile** | 미연결 필수 입력 포트 | 노드+포트 | 실행 시작 후 첫 샘플에서 KeyError. 전문가 모델 로딩 시간 낭비 |
| G2 | 미해결 제네릭/심볼 dim | 노드+타입변수 | 런타임에 예상 못한 shape가 흘러 배치 조립 실패 또는 조용한 broadcast |
| G2 | Output 노드 없음 | 그래프 | 몇 시간 계산 후 아무 산출물 없이 "완료" |
| G2 | 도달 불가 노드(어떤 Output에도 기여하지 않음) | 노드 | 쓸모없는 전문가 모델 추론으로 전처리 시간 2배 |
| G2 | 노드/플러그인 버전 해소 실패, major 불일치 | 노드 | 실행 중 임포트 실패 또는 시그니처 불일치 크래시 |
| G2 | 정답 누설 taint 도달성 (`sem=label`에서 prompt로 가는 경로 존재) | 경로 전체 | 위와 같음. 실험 전체가 무효인데 그 사실을 아무도 모름 |
| G2 | 물질화 경계 뒤에 `external_call` 노드 배치 | 노드 | 학습 루프 안에서 전문가 모델이 GPU를 뺏어 OOM 또는 극심한 감속 |
| G2 | Windows 프로파일 미지원 옵션(Linux 전용 커널·런타임) | Trainer 파라미터 | 학습 시작 직전 임포트 에러. 큐에 넣고 퇴근한 밤이 통째로 낭비 |
| **G3 dry-run** | 선언 타입과 실측 shape/dtype 불일치 | 노드+포트 | 첫 배치 크래시 또는 조용한 reshape |
| G3 | 정답 Text가 스키마 위반 | 스키마 규칙 + 샘플 키 | 파싱 불가한 정답으로 전량 학습. 출력 형식이 무너진 모델 |
| G3 | 토크나이저 특수 토큰·이미지 플레이스홀더 개수 불일치 | Trainer + prompt 노드 | 이미지가 조용히 무시되고 텍스트만 학습. 손실은 정상적으로 감소 |
| G3 | 전문가 모델 가중치 로드 실패 또는 출력 형식 불일치 | 플러그인 id | 물질화 90% 지점에서 실패, 남은 전처리 폐기 |
| G3 | 결정성 감사 실패(같은 입력, 다른 출력) | 노드 | 캐시가 오염되어 재실행 결과가 달라짐. 재현 불가 |
| **G4 자원 예산** | 샘플당 비전 토큰 + 텍스트 토큰 > context window | 이미지 개수·타일·해상도 파라미터를 가진 노드 | 조용한 truncation으로 이미지 뒷부분이 잘림. 손실은 잘 떨어지고 성능만 나쁨 |
| G4 | 추정 VRAM > 예산(24GB − reserve) | Trainer 단계 + 원인 항목(가중치/그래디언트/옵티마이저/활성화) | 학습 3시간 후 OOM, 체크포인트 없음, 한 장뿐인 GPU가 그동안 점유됨 |
| G4 | 물질화 산출물 예상 크기 > 디스크 여유 | 물질화 경계 노드 | 디스크 포화. 부분 기록된 shard로 학습이 조용히 일부 데이터만 사용 |

### 타입 에러 메시지 형식

```
TypeError [G1] n_prompt.image  <-  n_crop.crops

  기대: ImageList{u8, HWC, RGB, 0-255, frame=crop_px}
  실제: ImageList{f32, CHW, RGB, 0-1, norm=imagenet, frame=crop_px}
  불일치 필드: dtype(u8 != f32), layout(HWC != CHW),
               value_range(0-255 != 0-1), norm(none != imagenet)

  이 배선이 통과했다면:
    학습 시작 후 첫 배치에서는 터지지 않습니다.
    백본 프로세서가 이미 정규화된 텐서를 한 번 더 정규화하고,
    손실은 정상적으로 감소하며, 학습 완주 후 성능 저하로만 드러납니다.
    추정 낭비: 물질화 40분 + 학습 8.5시간.

  해결: adapt.image_denorm -> adapt.image_layout(CHW->HWC) -> adapt.image_dtype(f32->u8)
        또는 n_crop 상류의 adapt.image_norm 제거
```

에러는 항상 **(1) 불일치 필드 (2) 안 잡혔다면 언제 어디서 어떻게 드러났을지 (3) 추정 낭비 시간 (4) 구체적 해결 배선**을 담는다. 검증의 가치를 사용자가 매번 체감해야 게이트가 유지된다.
