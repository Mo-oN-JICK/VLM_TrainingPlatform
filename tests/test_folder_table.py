"""사진 폴더 → 표(CSV). 훑기 규칙과 맞추기 규칙."""

from __future__ import annotations

import pytest

from vlm_trainer.core.errors import SpecError
from vlm_trainer.engine import folder_table as F
from vlm_trainer.engine import samples


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def _rows(path):
    import csv

    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


def test_scan_reads_subfolders_in_name_order_and_joins_folder_names_with_double_underscore(tmp_path):
    for rel in ("b.png", "a.JPG", "검정/c01.jpeg", "검정/깊이/d.bmp", "note.txt", "검정/thumbs.db"):
        _touch(tmp_path / rel)
    items, skipped = F.scan(str(tmp_path))
    assert [k for k, _ in items] == ["a", "b", "검정__c01", "검정__깊이__d"]
    assert items[2][1] == "검정/c01.jpeg" and skipped == 2


def test_the_same_name_with_two_extensions_is_refused(tmp_path):
    _touch(tmp_path / "a.jpg")
    _touch(tmp_path / "a.png")
    with pytest.raises(SpecError, match="두 사진에 겹칩니다"):
        F.scan(str(tmp_path))


def test_sync_creates_the_table_then_appends_new_photos_and_marks_missing_ones_without_deleting(tmp_path):
    pics = tmp_path / "pics"
    _touch(pics / "a.jpg")
    _touch(pics / "흰색/b.jpg")
    csvp = tmp_path / "proj" / "data.csv"
    csvp.parent.mkdir()
    got = F.sync(str(pics), str(csvp), ["색상"])
    assert got == {"total": 2, "added": 2, "excluded": 0, "skipped": 0}
    rows = _rows(csvp)
    assert rows[0] == ["건", "사진", "색상", "제외"]
    assert rows[2][:2] == ["흰색__b", "../pics/흰색/b.jpg"]
    assert csvp.read_bytes().startswith(b"\xef\xbb\xbf")

    rows[1][2] = "검정색"                                     # 사람이 정답을 적었다
    F._write(str(csvp), rows)
    (pics / "a.jpg").unlink()
    _touch(pics / "c.png")
    got = F.sync(str(pics), str(csvp), ["색상", "크기"])
    assert (got["added"], got["excluded"]) == (1, 1)
    rows = _rows(csvp)
    assert rows[0] == ["건", "사진", "색상", "제외", "크기"]
    assert rows[1] == ["a", "../pics/a.jpg", "검정색", "x", ""]     # 지우지 않고 x, 정답은 남는다
    assert [r[0] for r in rows[1:]] == ["a", "흰색__b", "c"]

    before = csvp.read_bytes()
    assert F.sync(str(pics), str(csvp), ["색상"])["added"] == 0
    assert csvp.read_bytes() == before                          # 바뀐 것이 없으면 쓰지 않는다

    space = samples.load({"index": str(csvp), "key": "건", "splits": {"strategy": "random", "ratios": {"train": 1}}},
                         str(tmp_path))
    assert sorted(space.keys()) == ["c", "흰색__b"]
    assert space.excluded == 1


def test_a_missing_folder_is_refused(tmp_path):
    with pytest.raises(SpecError, match="사진 폴더가 없습니다"):
        F.sync(str(tmp_path / "없음"), str(tmp_path / "data.csv"), [])
