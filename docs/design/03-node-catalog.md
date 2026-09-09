# 3. 초기 기본 노드 카탈로그

Mech-Vision의 Step Library처럼 **기능별 카테고리**로 묶는다. `[문서확인 + 이미지확인]`
Mech-Vision 최신 문서의 카테고리는 Data Acquisition / Preprocessing / Recognition / AI Tools / Locating / Postprocessing / Measurement / Pose Processing / Path Planning / Trajectory Processing / Evaluation / Data Processing / File / Transmission / System / Tools / Common Procedure / Advanced Step 이고 `[문서확인]`, 캡처된 구버전 UI는 2D/3D Feature Detector·General Processing·Matching, Arithmetic, Camera, Communication, Deep Learning, Drawing, Label, Mask Processing, Measuring 등으로 나뉜다 `[이미지확인]`.

우리는 이름을 최대한 물려받되 로봇·점군 전용 카테고리를 우리 모달로 치환한다.

| Mech-Vision 카테고리 | 우리 카테고리 | 처리 |
|---|---|---|
| Data Acquisition / Camera | **Data Acquisition** | 모방(이름 유지, 소스가 카메라 대신 데이터셋) |
| Preprocessing / 2D General Processing | **2D General Processing** | 모방 |
| 3D General Processing / 3D Matching | **Time Series Processing** | `[각색]` 우리의 두 번째 모달은 점군이 아니라 센서 시계열 |
| AI Tools / Deep Learning | **Expert Models** | `[각색]` 추론 전용 외부 모델 호출로 한정 |
| Transform / Tools | **Adapters** | `[각색]` 암묵 변환 금지 정책의 실행 수단 |
| — | **Prompt Assembly** | `[신설]` VLM 고유 |
| — | **Answer Design** | `[신설]` VLM 고유 |
| Data Processing | **Data Processing** | 모방 |
| Evaluation | **Evaluation** | 모방 |
| Visualization / Drawing | **Visualization** | 모방 |
| File / Reader and Saver | **File** | 모방 |
| Transmission | **Training** | `[각색]` 로봇으로 결과를 보내는 말단이 우리에게는 학습 실행 |
| System / Common Procedure | **System** | 모방 |

표기: `I` = Input, `P` = Processing, `O` = Output.

---

## 3.1 Data Acquisition (전부 Input)

Input 노드는 배선으로 입력을 받지 않고 Project의 `sample_space` 컬럼을 파라미터로 지정해 값을 주입한다(문서 1.4).

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `source.image` | I | — | `image: Image{u8,HWC,RGB,0-255,frame=orig_px,shape=[?H,?W,3],sem=raw_image}` | `column`, `color_space(RGB\|BGR\|GRAY)`, `on_missing(fail\|skip)` |
| `source.image_list` | I | — | `images: ImageList{u8,HWC,RGB,0-255,frame=orig_px}` | `column`, `max_n`, `order(asc\|manifest)` |
| `source.timeseries` | I | — | `series: TimeSeries{f32,TC,shape=[?T,?C],hz,frame=ts_seconds,sem=raw_signal}` | `column`, `format(csv\|npy\|parquet)`, `hz`, `channels`, `t0_policy` |
| `source.metadata` | I | — | `meta: Table{cols=?}` | `columns`, `dtypes` |
| `source.field` | I | — | `value: Text{sem=?}` | `column`, `semantic`(예: `label`), `strip` |
| `source.text_asset` | I | — | `text: Text{sem=domain_knowledge}` | `path`, `encoding`, `max_chars` |
| `schema.define` | I | — | `schema: Schema` | `path`(답변 스키마 JSON/YAML) 또는 인라인 정의 |
| `source.regions_gt` | I | — | `regions: Regions{domain=?,frame=orig_px,sem=gt}` | `column`, `format(coco\|xyxy\|interval)` |

전부 `fingerprint()`를 구현한다: 정규화된 절대경로 + 내용 해시(blake3, 대용량은 헤더+크기+mtime) + `sample_space` 인덱스 파일 해시.

## 3.2 2D General Processing (전부 Processing)

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `image.crop_by_regions` | P | `image: Image{...orig_px}`, `regions: Regions{image2d,orig_px}` | `crops: ImageList{u8,HWC,RGB,0-255,frame=crop_px}` | `padding_ratio`, `square_pad`, `max_n`, `min_side_px`, `sort_by(score\|area)` |
| `image.crop_fixed` | P | `image` | `crop: Image{...frame=crop_px}` | `box_xyxy`, `relative(bool)` |
| `image.grid_tile` | P | `image` | `tiles: ImageList{...frame=tile_px}` | `tile_px`, `overlap`, `max_tiles` |
| `image.overlay_regions` | P | `image`, `regions` | `vis: Image{u8,HWC,RGB,0-255,sem=visualization}` | `color_by(score\|label)`, `thickness`, `label_text` |
| `image.concat` | P | `a: Image`, `b: Image` | `out: Image` | `axis(h\|v)`, `gap_px`, `align` |
| `image.mask_apply` | P | `image`, `mask: Mask` | `out: Image` | `fill_value`, `invert` |
| `list.wrap` | P | `item: T` | `items: List[T]` | — |
| `list.concat` | P | `a: List[T]`, `b: List[T]` | `out: List[T]` | `max_n`, `on_overflow(trim\|fail)` |

## 3.3 Time Series Processing (전부 Processing) `[각색]`

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `ts.window` | P | `series: TimeSeries{f32,TC,hz,ts_seconds}`, `regions: Regions{series1d,ts_seconds}`(optional) | `windows: List[TimeSeries{...}]` | `length_s`, `stride_s`, `align(center\|start)`, `pad_policy` |
| `ts.resample` | P | `series` | `series: TimeSeries{hz=out_hz}` | `out_hz`, `method(linear\|poly\|decimate)`, `anti_alias` |
| `ts.select_channels` | P | `series` | `series: TimeSeries{shape=[?T,k]}` | `channels` |
| `ts.stats` | P | `series` | `stats: Table` | `features(mean,std,trend_slope,acf_peak,spike_count,...)`, `window_s` |
| `ts.stats_to_text` | P | `stats: Table` | `text: Text{sem=evidence}` | `template`, `precision`, `unit_map` |
| `ts.plot` | P | `series`, `regions`(optional) | `plot: Image{u8,HWC,RGB,0-255,sem=raw_image}` | `size_px`, `dpi`, `channels_per_row`, `mark_regions`, `theme` |
| `ts.detect_spikes` | P | `series` | `regions: Regions{series1d,ts_seconds,sem=rule_hint}` | `z_thresh`, `min_width_s`, `merge_gap_s` |

`ts.plot`은 시계열을 VLM에 넣기 위한 표준 경로다. 렌더 결과가 이미지 타입이 되므로 이후 `adapt.*` 체인과 그대로 결합된다.

## 3.4 Expert Models (Processing, `external_call: true`)

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `expert.propose` | P | `subject: T where T in {Image, TimeSeries}` | `regions: Regions{domain=?D, frame=?F, sem=expert_hint}` | `plugin`(id@ver), `device`, `dtype`, `topk`, `score_threshold`, `seed` |
| `expert.classify` | P | `subject: T` | `labels: TextList{sem=expert_hint}`, `scores: Table` | `plugin`, `topk`, `threshold` |
| `expert.caption` | P | `image: Image` | `text: Text{sem=expert_hint}` | `plugin`, `max_tokens`, `seed` |
| `expert.embed` | P | `subject: T` | `emb: Embedding{f32,shape=[?D]}` | `plugin`, `pooling`, `normalize` |

`?D`, `?F`는 플러그인 매니페스트가 결정하는 타입 변수다(문서 5). `expert.*`는 물질화 경계 앞에 있어야 하며 아니면 G2 실패.

## 3.5 Adapters (Processing) — 암묵 변환 금지의 실행 수단

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `adapt.image_resize` | P | `image: Image{shape=[?H,?W,3]}` | `image: Image{shape=[h,w,3]}` | `size`, `mode(bilinear\|bicubic\|area)`, `keep_aspect`, `pad_value` |
| `adapt.image_dtype` | P | `image: Image{dtype=a}` | `image: Image{dtype=b}` | `to`, `clip`, `round_policy` |
| `adapt.image_layout` | P | `image: Image{layout=a}` | `image: Image{layout=b}` | `to(HWC\|CHW)` |
| `adapt.image_norm` | P | `image: Image{range=0-255,norm=none}` | `image: Image{range=0-1,norm=?}` | `scheme(scale01\|imagenet\|custom)`, `mean`, `std` |
| `adapt.image_denorm` | P | `image: Image{norm=x}` | `image: Image{norm=none}` | — |
| `adapt.colorspace` | P | `image: Image{color=a}` | `image: Image{color=b}` | `to(RGB\|BGR\|GRAY)` |
| `adapt.frame` | P | `regions: Regions{frame=a}` | `regions: Regions{frame=b}` | `to`, `ref_shape`(원본 크기 참조), `clip_to_bounds` |
| `adapt.ts_time_base` | P | `series/regions{ts_index}` | `{ts_seconds}` | `hz`, `t0` |
| `adapt.tokenize` | P | `text: Text` | `tokens: Tokens{int64,shape=[?L]}` | `tokenizer`(백본 어댑터에서 해소), `add_special`, `max_len`, `on_overflow(fail\|truncate)` |
| `adapt.regions_topk` | P | `regions` | `regions{max_n=k}` | `k`, `sort_by`, `min_score` |
| `optional.unwrap_or` | P | `value: T?` | `value: T` | `default` |

`adapt.tokenize`의 `on_overflow` 기본값은 `fail`이다. 조용한 잘림이 우리 도메인에서 가장 비싼 오류이기 때문이다.

## 3.6 Prompt Assembly (Processing) `[신설]`

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `text.template` | P | `slots: 가변(선언한 슬롯 이름마다 포트 생성)` | `text: Text{sem=prompt}` | `template`(Jinja 유사 문법), `slots`(이름→타입), `strip`, `max_chars` |
| `text.join` | P | `parts: TextList` | `text: Text` | `sep`, `order` |
| `text.system_prompt` | P | `text: Text{sem=prompt}` | `text: Text{sem=prompt}` | `system`, `role_format(chatml\|plain)` |
| `prompt.image_slots` | P | `text: Text{sem=prompt}`, `images: ImageList` | `text: Text{sem=prompt, image_slots=n}` | `placeholder`(예: `<image>`), `policy(prepend\|inline)` |
| `prompt.knowledge_inject` | P | `text: Text{sem=prompt}`, `knowledge: Text{sem=domain_knowledge}` | `text: Text{sem=prompt}` | `position(before\|after)`, `header`, `max_chars`, `dedup` |

`prompt.image_slots`가 산출하는 `image_slots=n`은 타입 필드다. Trainer가 기대하는 이미지 개수와 다르면 G2에서 잡힌다.

## 3.7 Answer Design (Processing) `[신설]` — 상세는 문서 6

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `answer.stepwise` | P | `schema: Schema`, `fields: Table`, `evidence: TextList`(optional) | `answer: Text{sem=answer}` | `render(json\|tagged\|markdown)`, `step_order`, `locale` |
| `answer.from_template` | P | `schema`, `fields` | `answer: Text{sem=answer}` | `template` |
| `answer.evidence_rules` | P | `stats: Table`, `regions: Regions`(optional) | `evidence: TextList{sem=evidence}` | `rules`(YAML 규칙 파일), `precision`, `fallback` |
| `answer.validate` | P | `answer: Text{sem=answer}`, `schema: Schema` | `answer: Text{sem=answer,validated=true}`, `report: Report` | `on_violation(quarantine\|drop\|fail)`, `max_tokens` |
| `answer.leakage_guard` | P | `prompt: Text{sem=prompt}`, `answer: Text{sem=answer}` | `prompt: Text{sem=prompt,leak_checked=true}`, `report: Report` | `mode(exact\|normalized\|regex)`, `vocab`, `on_leak(fail\|mask)` |

## 3.8 Data Processing (Processing)

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `sample.assemble` | P | `images: ImageList`(optional), `prompt: Text{sem=prompt}`, `answer: Text{sem=answer,validated=true}`, `meta: Table`(optional) | `sample: Sample{...}` | `id_from`, `keep_meta_cols`, `require_validated(true)` |
| `table.select` | P | `table: Table` | `table: Table` | `columns`, `rename` |
| `table.compute` | P | `table` | `table` | `expressions`(순수 산술/문자열 식만) |
| `table.to_text` | P | `table` | `text: Text` | `template`, `precision` |
| `list.map` | P | `items: List[T]` | `out: List[U]` | `procedure`(서브 Procedure 참조), `max_n` |
| `list.reduce_text` | P | `items: TextList` | `text: Text` | `sep`, `limit` |

## 3.9 Evaluation (Processing)

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `eval.schema_conformance` | P | `answer`, `schema` | `report: Report` | `sample_n`, `fail_ratio_threshold` |
| `eval.dataset_stats` | P | `sample: Sample` | `report: Report` | `bins`, `group_by` |
| `eval.token_budget` | P | `sample: Sample` | `report: Report` | `tokenizer`, `max_len`, `percentile` |
| `eval.split_leakage` | P | `sample: Sample` | `report: Report` | `key`, `group_col`(예: 환자 id) |
| `eval.class_balance` | P | `sample: Sample` | `report: Report` | `label_field`, `min_ratio` |

Evaluation 노드는 Report를 만들 뿐 부작용이 없으므로 Processing이다. Report를 파일로 남기려면 `io.report_save`(Output)로 흘린다.

## 3.10 Visualization (Processing)

Mech-Vision과 동일하게 시각화 결과는 **포트로 나오는 값**이지 종결점이 아니다. `[이미지확인: Visualization Image 출력 포트]`

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `vis.image_grid` | P | `images: ImageList` | `vis: Image{sem=visualization}` | `cols`, `cell_px`, `captions` |
| `vis.prompt_render` | P | `prompt: Text{sem=prompt}`, `images: ImageList`(optional) | `vis: Text{sem=visualization}` | `resolve_slots(true)`, `highlight_slots` |
| `vis.answer_render` | P | `answer: Text{sem=answer}`, `report: Report` | `vis: Text{sem=visualization}` | `show_violations`, `show_step_ids` |
| `vis.timeseries` | P | `series`, `regions`(optional) | `vis: Image{sem=visualization}` | `size_px`, `mark_regions` |

## 3.11 File (Reader and Saver)

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `io.dataset_export` | **O** | `sample: Sample` | — | `format(jsonl\|webdataset\|arrow)`, `out_dir`, `shard_size`, `image_encoding(png\|jpeg:q)`, `split_by` |
| `io.report_save` | **O** | `report: Report` | — | `path`, `format(json\|md)` |
| `debug.data_storage` | **O** | `any: T`(가변 다중 입력) | — | `enabled`, `path`, `what(inputs\|previews\|answers\|all)`, `keep_runs` |

`debug.data_storage`는 Mech-Vision의 Data Storage에 대응한다. `[문서확인: Data Storage 기능 존재]` `[추정: 저장 항목의 세부 구성]`

## 3.12 Training (Output) `[신설 위치, Transmission 자리]`

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `train.vlm_trainer` | **O** | `sample: Sample`, `schema: Schema` | — | 문서 7의 TrainerConfig 전체 |
| `train.resume` | **O** | `sample: Sample` | — | `run_id`, `from_step`, `strict_spec_match(true)` |

`strict_spec_match=true`가 기본이다. 재개할 run의 canonical 스펙 해시가 현재 스펙과 다르면 재개를 거부한다.

## 3.13 System

| 노드 | 태그 | 입력 | 출력 | 주요 파라미터 |
|---|---|---|---|---|
| `sys.procedure` | P | Procedure가 노출한 입력 | Procedure가 노출한 출력 | `ref`(name@semver), 노출 파라미터 |
| `sys.constant` | I | — | `value: T` | `type`, `value` |
| `sys.assert_type` | P | `value: T` | `value: T` | `expect`(타입 표현식) — 문서화용 검문소, 런타임 무비용 |
| `sys.note` | — | — | — | 캔버스 주석. 그래프 의미에 영향 없음 |

## 3.14 검증 케이스(Triad)에 필요한 최소 세트

`source.image`, `source.timeseries`, `source.text_asset`, `source.field`, `schema.define`,
`expert.propose`, `adapt.frame`, `adapt.regions_topk`, `image.crop_by_regions`, `adapt.image_resize`,
`ts.stats`, `ts.stats_to_text`, `answer.evidence_rules`, `answer.stepwise`, `answer.validate`,
`answer.leakage_guard`, `text.template`, `prompt.knowledge_inject`, `prompt.image_slots`,
`sample.assemble`, `io.dataset_export`, `train.vlm_trainer`.

22개. 이 세트가 Phase 2의 구현 범위다(문서 10).
