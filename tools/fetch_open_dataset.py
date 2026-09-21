"""공개 데이터로 실물 학습 시험용 데이터셋을 만든다.

합성 이미지(`make_parts_dataset.py`)는 파이프라인이 도는지는 보여 주지만 실물 사진이
아니다. 4090에서 실제 백본으로 학습을 태워 보려면 진짜 사진이 필요하다.

**출처는 Wikimedia Commons 하나로 고정한다.** 파일마다 라이선스를 API로 확인할 수 있는
곳이 여기뿐이고, 공개 저장소에 같이 올리려면 그 확인이 선택이 아니다. 허용 목록 밖의
라이선스(ShareAlike/NonCommercial/NoDerivs)는 통과시키지 않는다 — SA는 파생물에까지
조건을 옮기고, NC는 쓰는 쪽을 제약한다.

정답은 지어내지 않는다. `part_type`은 Commons의 분류에서, `orientation`은 파일의
가로세로에서 온다. 둘 다 사진에 대해 실제로 참인 값이다.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

API = "https://commons.wikimedia.org/w/api.php"
UA = "VLM-Trainer-dataset-builder/0.1 (educational; https://github.com/Mo-oN-JICK/VLM_TrainingPlatform)"

# 제목에 이 말이 들어 있어야 받는다. Commons 분류는 **주제**로 묶여 있어서
# "Category:Rivets" 안에 리벳으로 고정한 칼집이, "Category:Ball bearings" 안에
# 베어링을 소재로 한 2차대전 포스터가 들어 있다. 분류만 믿으면 라벨이 거짓이 된다.
TITLE_KEYS: Dict[str, Tuple[str, ...]] = {
    "Category:Gears": ("gear", "cogwheel", "zahnrad"),
    "Category:Ball bearings": ("bearing", "kugellager"),
    "Category:Screws": ("screw", "schraube"),
    "Category:Springs (mechanical)": ("spring", "feder"),
    "Category:Chains": ("chain", "kette"),
    "Category:Nuts (hardware)": ("nut", "mutter"),
    "Category:Washers (hardware)": ("washer", "scheibe"),
    "Category:Pulleys": ("pulley", "sheave"),
    "Category:Rivets": ("rivet", "niet"),
    "Category:Flanges": ("flange", "flansch"),
}

# 사진이 아닌 것들. 도면·포스터·기록물 스캔이 섞이면 모델이 부품 대신 종이를 배운다.
TITLE_DENY: Tuple[str, ...] = (
    # 영어 "gear" 는 캠핑 장비와 착륙 장치까지 뜻한다. 분류가 맞아도 사진은 딴것이다.
    "camping", "landing", "backpack", "outdoor", "climbing", "bridge", "luggage",
    "poster", "diagram", "drawing", "patent", " map", "logo", "chart", "nara",
    "engraving", "illustration", "sketch", "painting", "icon", "symbol",
    "blueprint", "plan ", "sign", "stamp", "coin", "cover", "book", "page",
    "portrait", "museum", "church", "monument", "schema", "graph", "print",
)

# 분류 -> 한국어 부품 이름. 정답 어휘가 되므로 스키마의 values와 같아야 한다.
CATEGORIES: Dict[str, str] = {
    "Category:Gears": "기어",
    "Category:Ball bearings": "베어링",
    "Category:Screws": "나사",
    "Category:Springs (mechanical)": "스프링",
    "Category:Chains": "체인",
    "Category:Nuts (hardware)": "너트",
    "Category:Washers (hardware)": "와셔",
    "Category:Pulleys": "풀리",
    "Category:Rivets": "리벳",
    "Category:Flanges": "플랜지",
}

# 허용 라이선스. 재배포와 축소(파생물)가 조건 없이 또는 출처 표기만으로 가능한 것들.
# 판정은 **슬러그**로만 한다. 자유 텍스트를 짧은 토큰으로 훑으면 "and" 가 "nd" 에 걸리는
# 식으로 멀쩡한 파일이 조용히 빠진다 — 안전한 쪽 실수지만 이유를 알 수 없게 된다.
ALLOWED_EXACT = {"cc0", "pd", "public domain", "cc-pd-mark"}
ALLOWED_PREFIX = ("cc-by-1.0", "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0",
                  "pd-", "cc0-", "cc-zero")
# 이 조각이 슬러그에 있으면 무조건 거른다. 허용 목록보다 이쪽이 먼저다.
DENY_SLUG = ("-sa", "-nc", "-nd", "sharealike", "noncommercial", "noderiv",
             "fairuse", "gfdl", "nonfree")

# 프롬프트 100개. 열 가지 도입 x 다섯 가지 지시 x 두 가지 마무리.
# **정답 어휘를 넣지 않는다** — 부품 이름도, 방향 값(가로/세로/정사각)도.
# 프롬프트에 정답이 섞이면 누설 게이트(answer.leakage_guard)가 그래프를 세운다.
_OPEN = (
    "아래 사진을 보고", "이 이미지를 보고", "제시된 사진을 판독해",
    "사진을 자세히 살펴보고", "첨부된 이미지를 확인해", "다음 사진을 기준으로",
    "주어진 이미지를 분석해", "화면에 보이는 사진에서", "사진 속 대상을 관찰해",
    "이 한 장의 사진으로부터",
)
_TASK = (
    "어떤 기계 요소인지 판정하고 이미지의 방향도 함께 적으시오",
    "대상 부품의 종류와 사진의 방향을 판정하시오",
    "무엇을 찍은 것인지 분류하고 이미지 방향을 밝히시오",
    "해당 기계 부품의 분류와 화면 방향을 결정하시오",
    "물체의 부품 명칭과 이미지의 화면 형태를 판정하시오",
)
_CLOSE = (".", ". 정해진 항목 순서대로 답하시오.")

QUESTIONS = [
    f"{o} {t}{c}" for o in _OPEN for t in _TASK for c in _CLOSE
]


def call(**params: Any) -> Dict[str, Any]:
    params.setdefault("format", "json")
    params.setdefault("action", "query")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    # 공개 API다. 429가 오면 물러섰다 다시 묻는다 — 조르는 쪽이 잘못이다.
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("도달할 수 없음")


def members(category: str, limit: int) -> List[str]:
    """분류에 속한 파일 제목. 이어받기로 limit까지 모은다."""
    out: List[str] = []
    cont: Optional[str] = None
    while len(out) < limit:
        kw: Dict[str, Any] = dict(
            list="categorymembers", cmtitle=category, cmtype="file",
            cmlimit=min(200, limit - len(out)),
        )
        if cont:
            kw["cmcontinue"] = cont
        d = call(**kw)
        out += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
        cont = d.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return out


def title_ok(title: str, category: str) -> bool:
    """제목이 그 부품을 실제로 말하고 있는가. 분류 소속만으로는 라벨이 서지 않는다."""
    low = title.lower()
    if any(bad in low for bad in TITLE_DENY):
        return False
    return any(key in low for key in TITLE_KEYS.get(category, ()))


def license_ok(license_id: str, terms: str) -> bool:
    """슬러그가 없으면 쓰지 않는다 — 모르는 라이선스를 공개 저장소에 올릴 수는 없다."""
    slug = (license_id or "").strip().lower()
    if not slug:
        return False
    if any(bad in slug for bad in DENY_SLUG):
        return False
    return slug in ALLOWED_EXACT or slug.startswith(ALLOWED_PREFIX)


def strip_html(s: str) -> str:
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return " ".join("".join(out).split())


def info(titles: List[str], width: int) -> List[Dict[str, Any]]:
    """제목 묶음의 imageinfo. 한 번에 50개까지가 API 한도다."""
    rows: List[Dict[str, Any]] = []
    for i in range(0, len(titles), 50):
        chunk = titles[i : i + 50]
        d = call(titles="|".join(chunk), prop="imageinfo",
                 iiprop="url|extmetadata|size|mime", iiurlwidth=width)
        for page in d.get("query", {}).get("pages", {}).values():
            ii = (page.get("imageinfo") or [None])[0]
            if not ii:
                continue
            em = ii.get("extmetadata", {})
            rows.append({
                "title": page.get("title", ""),
                "thumb": ii.get("thumburl", ""),
                "descurl": ii.get("descriptionurl", ""),
                "w": ii.get("width", 0),
                "h": ii.get("height", 0),
                "mime": ii.get("mime", ""),
                "license": em.get("License", {}).get("value", ""),
                "license_name": em.get("LicenseShortName", {}).get("value", ""),
                "terms": em.get("UsageTerms", {}).get("value", ""),
                "artist": strip_html(em.get("Artist", {}).get("value", "")) or "(미상)",
            })
        time.sleep(0.2)   # 예의. 공개 API를 두드리는 속도를 스스로 제한한다
    return rows


def orientation(w: int, h: int) -> str:
    if w > h * 1.1:
        return "가로"
    if h > w * 1.1:
        return "세로"
    return "정사각"


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def is_photograph(raw: bytes) -> bool:
    """사진인가, 그려진 것인가. 픽셀로 판정한다.

    제목 필터를 통과해도 도면·CAD 렌더·그래프가 남는다. 그것들은 밝기가 몇 단계에
    몰려 있고 흰 바탕이 넓다 — 사진은 밝기가 고루 퍼진다. 휘도 히스토그램의
    엔트로피와 최빈 밝기 비중, 둘이면 갈린다.

    흑백 사진을 색 가짓수로 재면 도면과 구별되지 않는다. 그래서 휘도로 본다.
    """
    from PIL import Image

    im = Image.open(io.BytesIO(raw)).convert("L")
    im.thumbnail((200, 200))
    hist = im.histogram()
    n = sum(hist) or 1
    entropy = -sum((c / n) * math.log2(c / n) for c in hist if c)
    peak = max(hist) / n
    return entropy >= 4.2 and peak <= 0.45


def save_jpeg(raw: bytes, path: str, side: int) -> Tuple[int, int]:
    """긴 변을 side로 맞춰 JPEG로 저장한다. 저장소에 같이 올릴 것이라 크기를 줄인다."""
    from PIL import Image

    im = Image.open(io.BytesIO(raw))
    im = im.convert("RGB")
    w, h = im.size
    scale = side / float(max(w, h))
    if scale < 1.0:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    im.save(path, "JPEG", quality=88, optimize=True)
    return im.size


def collect(per_cat: int, width: int) -> List[Dict[str, Any]]:
    """분류마다 후보를 모아 라이선스로 거른다. 분류 순서는 고정이라 결과가 재현된다."""
    picked: List[Dict[str, Any]] = []
    for cat in sorted(CATEGORIES):
        # 제목 필터가 열에 일고여덟을 걷어낸다. 깊게 훑어야 분포가 고르다.
        titles = members(cat, max(400, per_cat * 20))
        rows = info(titles, width)
        kept = []
        for r in rows:
            if r["mime"] not in ("image/jpeg", "image/png"):
                continue
            if not r["thumb"] or min(r["w"], r["h"]) < 200:
                continue
            if not license_ok(r["license"], r["terms"]):
                continue
            if not title_ok(r["title"], cat):
                continue
            r["category"] = cat
            r["part_type"] = CATEGORIES[cat]
            kept.append(r)
        kept.sort(key=lambda r: r["title"])   # 결정적으로
        print(f"  {cat:<32} 후보 {len(rows):>3} -> 통과 {len(kept):>3}", flush=True)
        picked += kept[:per_cat]
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description="Wikimedia Commons에서 학습 시험용 부품 사진을 모은다")
    ap.add_argument("--out", default="solutions/vlm_open/data/open_parts")
    ap.add_argument("--train", type=int, default=100)
    ap.add_argument("--val", type=int, default=10)
    ap.add_argument("--side", type=int, default=448, help="긴 변 픽셀")
    a = ap.parse_args()

    need = a.train + a.val
    per_cat = -(-need // len(CATEGORIES)) + 8   # 올림 + 여유
    print(f"Commons에서 모읍니다 (분류 {len(CATEGORIES)}개 x 최대 {per_cat}건)")
    rows = collect(per_cat, a.side)
    if len(rows) < need:
        print(f"  [경고] {len(rows)}건만 모였다. 필요한 것은 {need}건이다.", file=sys.stderr)

    # 분류를 고르게 섞는다 — 한 분류가 val에 몰리면 시험이 되지 않는다
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r["part_type"], []).append(r)
    order: List[Dict[str, Any]] = []
    i = 0
    while len(order) < len(rows):
        for k in sorted(by_cat):
            if i < len(by_cat[k]):
                order.append(by_cat[k][i])
        i += 1
    # 같은 사진이 여러 분류에 걸려 있으면 내려받다 걸러진다. 여분을 두고 시작한다.
    # 사진 판정에서 또 걸러지므로 넉넉히 잡는다.
    rows = order[: need * 2 + 40]

    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    index: List[Dict[str, Any]] = []
    credits: List[str] = []
    seen: set = set()
    n = 0
    dropped = 0   # 사진이 아니라 걸러진 수
    for r in rows:
        try:
            raw = fetch_bytes(r["thumb"])
        except Exception as exc:
            print(f"  건너뜀 {r['title'][:50]}: {exc}", file=sys.stderr)
            continue
        digest = hashlib.sha256(raw).hexdigest()[:16]
        if digest in seen:
            continue     # 같은 사진이 여러 분류에 걸려 있는 경우
        seen.add(digest)
        if not is_photograph(raw):
            dropped += 1
            continue

        split = "val" if n >= a.train else "train"
        sid = f"o{n:04d}"
        rel = f"images/{sid}.jpg"
        w, h = save_jpeg(raw, os.path.join(img_dir, sid + ".jpg"), a.side)
        index.append({
            "sample_id": sid,
            "source_id": digest,
            "image_path": rel,
            "question": QUESTIONS[n % len(QUESTIONS)],
            "part_type": r["part_type"],
            "orientation": orientation(w, h),
            "license": r["license_name"],
            "split_hint": split,
        })
        credits.append(
            f"| {sid} | [{r['title'].replace('File:', '')}]({r['descurl']}) "
            f"| {r['artist']} | {r['license_name']} |"
        )
        n += 1
        if n % 10 == 0:
            print(f"  {n}/{need}", flush=True)
        if n >= need:
            break

    with io.open(os.path.join(a.out, "index.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for row in index:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with io.open(os.path.join(a.out, "CREDITS.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# 출처와 라이선스\n\n")
        fh.write("모든 사진은 [Wikimedia Commons](https://commons.wikimedia.org)에서 왔고,\n")
        fh.write("재배포와 축소가 허용된 라이선스만 담았다. 원본을 긴 변 "
                 f"{a.side}px로 줄여 JPEG로 저장했다.\n\n")
        fh.write("CC BY 계열은 출처 표기가 조건이다. 아래 표가 그 표기다.\n\n")
        fh.write("| 파일 | 원본 | 작성자 | 라이선스 |\n|---|---|---|---|\n")
        fh.write("\n".join(credits) + "\n")

    counts: Dict[str, int] = {}
    for row in index:
        counts[row["part_type"]] = counts.get(row["part_type"], 0) + 1
    print()
    print(f"  사진이 아니라 걸러짐: {dropped}장")
    print(f"{a.out}/index.jsonl: {len(index)}건 "
          f"(train {sum(1 for r in index if r['split_hint'] == 'train')} · "
          f"val {sum(1 for r in index if r['split_hint'] == 'val')})")
    print("  부품 분포:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    lic: Dict[str, int] = {}
    for row in index:
        lic[row["license"]] = lic.get(row["license"], 0) + 1
    print("  라이선스:", ", ".join(f"{k} {v}" for k, v in sorted(lic.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
