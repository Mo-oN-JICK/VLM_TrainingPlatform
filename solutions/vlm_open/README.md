# 공개 데이터 부품 분류 (`vlm_open`)

**합성 이미지가 아니라 실물 사진으로 학습 경로 전체를 태워 보기 위한 Solution이다.**
`vlm_parts`는 `tools/make_parts_dataset.py`가 그린 도형을 쓴다 — 파이프라인이 도는지는
보여 주지만 진짜 사진은 아니다. 이쪽은 4090에서 실제로 학습을 돌려 보는 용도다.

## 무엇이 들어 있나

| | |
|---|---|
| 학습용 이미지 | 100장 |
| 학습용 프롬프트 | 100개 (전부 다른 문장) |
| 검증용 이미지 | 10장 |
| 출처 | Wikimedia Commons |
| 라이선스 | CC0 · Public Domain · CC BY (SA/NC/ND 없음) |
| 크기 | 4.8MB (긴 변 448px JPEG) |

정답은 두 항목이고 **둘 다 사진에 대해 실제로 참인 값**이다.
`part_type`은 Commons의 분류에서, `orientation`은 파일의 가로세로에서 온다.

```
부품: 베어링
화면: 가로
```

## 바로 돌려 보기

한 줄이면 끝까지 간다 — 준비 → 굽기 → 학습 → 추론.

```
.venv\Scripts\python.exe -m vlm_trainer.cli.main pipeline solutions\vlm_open\projects\01_open\project.yaml --run-id t1
```

단계는 그래프가 정한다. 이 그래프에는 학습과 추론이 둘 다 있으므로 네 단계를 돈다.
검증 10장의 답은 `runs\t1\infer\answers.jsonl` 에 정답과 나란히 적힌다 —
**채점하지 않는다.** 사람이 읽고 판단할 일이다.

굽기는 `train` split 만 굽는다. 검증 샘플까지 구우면 학습이 그것도 읽어서, 나중에 검증
답을 봐도 이미 외운 것을 다시 물어보는 셈이 된다. 전부 구우려면 `--bake-split all`.

중간에 멈췄다가 같은 `--run-id` 로 다시 부르면 굽다 만 것이 있다고 **물어본다**.
이어 받으려면 `--resume`, 처음부터 구우려면 `--fresh`.

단계를 따로 보고 싶으면 원래 명령이 그대로 있다. `pipeline` 이 부르는 것도 이것들이다.

```
.venv\Scripts\python.exe -m vlm_trainer.cli.main compile solutions\vlm_open\projects\01_open\project.yaml
.venv\Scripts\python.exe -m vlm_trainer.cli.main dryrun  solutions\vlm_open\projects\01_open\project.yaml
.venv\Scripts\python.exe -m vlm_trainer.cli.main budget  solutions\vlm_open\projects\01_open\project.yaml

.venv\Scripts\python.exe -m vlm_trainer.cli.main materialize solutions\vlm_open\projects\01_open\project.yaml --run-id t1 --split train
.venv\Scripts\python.exe -m vlm_trainer.cli.main train       solutions\vlm_open\projects\01_open\project.yaml --run-id t1
.venv\Scripts\python.exe -m vlm_trainer.cli.main run         solutions\vlm_open\projects\01_open\project.yaml --run-id t1 --split val
```

편집기로 그래프를 보려면 `editor.bat solutions\vlm_open\projects\01_open\project.yaml`.
창 하나로 뜨는 네이티브 앱이고, 툴바의 **`실행`** 이 위의 `pipeline` 과 같은
명령을 부른다. 처음이라면 `pip install -e .[app]` 로 PySide6 를 깐다.

`trainer.yaml`의 백본은 `tiny-vlm`이다 — 저장소에 들어 있어 **가중치를 내려받지 않고도**
데이터 적재부터 손실·저장까지 전부 돈다. 실물 2B 백본은 `trainer_qwen2vl.yaml` 쪽이고,
그것은 가중치 4.4GB와 Qwen2-VL 전용 collate가 아직 필요하다(`STATUS.md` 3a).

## 이 데이터의 한계

**라벨에 잡음이 있다.** Commons의 분류는 사람이 주제로 묶은 것이라, "Category:Rivets"
안에 리벳으로 고정한 칼집이 들어 있는 식이다. 수집기가 세 겹으로 거른다 —
라이선스, 제목에 부품 이름이 실제로 있는지, 그리고 픽셀 통계로 사진인지 도면인지.
그래도 열에 두셋은 어긋난다. 파이프라인을 시험하기에는 충분하지만 **정확도를 재는
벤치마크로 쓸 물건은 아니다.**

영어 `gear`가 캠핑 장비를 뜻하는 것처럼 낱말의 중의성도 남아 있다.

## 다시 만들기

```
.venv\Scripts\python.exe tools\fetch_open_dataset.py --train 100 --val 10
```

분류 순서와 제목 정렬이 고정이라 같은 결과가 나온다. 출처 표기는
`data/open_parts/CREDITS.md`가 파일마다 들고 있다 — CC BY의 조건이다.
