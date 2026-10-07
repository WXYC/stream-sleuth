"""The leg runner's CLI end to end: real ffmpeg cuts, a localhost Shazam, a fake Olaf.

Hours are 20 s of sine, so each holds one 12 s clip and one 6 s clip (at 0 s). The Olaf
binary is never run here: ``OlafRecognizer`` is replaced, and its own end-to-end test is in
``test_olaf_snapshot.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from evaluation import run as run_mod
from evaluation.olaf_snapshot import RESULTS, SnapshotError, build_snapshot, snapshot_lock
from evaluation.pool import open_pool_db
from evaluation.run import main
from stream_sleuth.paths import CHECKOUT, DataPathError
from tests.audio import render
from tests.characterization.shazam_responses import NO_MATCH
from tests.shazam_fake import FakeShazam, json_response

pytestmark = pytest.mark.ffmpeg

HOUR = "2026/08/12/202608121600.mp3"
OTHER = "2026/08/12/202608121700.mp3"
MATCH = {
    "artist": "Jessica Pratt",
    "song": "Back, Baby",
    "album": "On Your Own Love Again",
    "label": "",
    "source": "local",
    "confidence": 31.0,
    "query_offset_s": 2.0,
    "ref_start_s": 50.0,
    "ref_key": "cd" * 20,
}


class FakeOlaf:
    def __init__(self, home: Path, min_match_count: int, lookup: object) -> None:
        self.home = home

    def recognize(self, wav_path: str) -> dict:
        assert wav_path.endswith(".wav")  # Olaf legs cut the decoded capture, as the live loop does
        return MATCH


@pytest.fixture(autouse=True)
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", "0")

    class Plenty:  # never the host's disk; the free-space test sets its own
        free = 1 << 40

    monkeypatch.setattr(run_mod.shutil, "disk_usage", lambda p: Plenty)
    monkeypatch.setattr(run_mod, "OlafRecognizer", FakeOlaf)
    for key in (HOUR, OTHER):
        path = tmp_path / "data" / "archive" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        render(path, 20, args=["-c:a", "libmp3lame", "-b:a", "128k"])
    labels = {
        HOUR: {"group": "contrast", "band": "evening", "subset": True},
        OTHER: {"group": "contrast", "band": "evening", "subset": False},
    }
    (tmp_path / "data" / "selection.json").write_text(json.dumps({"hours": labels}))
    return tmp_path / "data"


@pytest.fixture
def fake() -> Iterator[FakeShazam]:
    fake = FakeShazam([json_response(200, NO_MATCH)] * 3)
    yield fake
    fake.close()


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_shazam_legs_read_selection_json_and_share_one_throttle(
    data: Path, fake: FakeShazam
) -> None:
    assert main(["--only", "shazam", "--legs", "12s", "6s-subset", "--base-url", fake.url]) == 0
    # The 12 s leg takes both hours; the 6 s subset leg only the hour with subset: true.
    assert [r["address"] for r in _records(data / "shazam" / "results.jsonl")] == [
        f"{HOUR}#0+12@128k",
        f"{OTHER}#0+12@128k",
        f"{HOUR}#0+6@128k",
    ]
    assert json.loads((data / "shazam" / "throttle.json").read_text())["count"] == 3
    assert list((data / "clips").iterdir()) == []


def test_an_olaf_leg_files_its_results_in_the_snapshot(data: Path) -> None:
    home = data / "olaf" / "rotation"
    open_pool_db(home / "pool.db").close()
    assert main(["--only", "olaf", "--snapshot", "rotation", "--legs", "12s"]) == 0
    records = _records(home / RESULTS)
    assert [r["address"] for r in records] == [f"{HOUR}#0+12@128k", f"{OTHER}#0+12@128k"]
    assert {r["recognizer"] for r in records} == {run_mod.olaf_identity("rotation")}
    assert {r["ref_key"] for r in records} == {"cd" * 20}
    assert not (data / "shazam" / "throttle.json").exists()  # an Olaf-only run spends no budget
    assert list((data / "clips").iterdir()) == []


def test_a_run_with_no_shazam_leg_touches_nothing_of_shazams(
    data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object) -> None:
        raise AssertionError("an Olaf-only run must not reach Shazam's pin or budget")

    monkeypatch.setattr(run_mod, "require_pinned_shazamio", refuse)
    monkeypatch.setattr(run_mod, "budget_from_env", refuse)
    monkeypatch.setattr(run_mod, "Throttle", refuse)
    open_pool_db(data / "olaf" / "rotation" / "pool.db").close()
    for argv in (["--only", "olaf", "--legs", "12s"], ["--legs", "6s", "20s"]):
        assert main([*argv, "--snapshot", "rotation"]) == 0
    assert not (data / "shazam").exists()  # no state file, no lock, not even its directory


def test_olaf_legs_need_a_built_snapshot_and_a_name(data: Path) -> None:
    with pytest.raises(SystemExit, match="no snapshot"):
        main(["--only", "olaf", "--snapshot", "rotation"])
    with pytest.raises(SystemExit):
        main(["--only", "olaf"])  # the parser's own error: --snapshot is required
    assert not (data / "olaf").exists()


def test_a_snapshot_being_built_is_refused(data: Path) -> None:
    home = data / "olaf" / "rotation"
    open_pool_db(home / "pool.db").close()
    with snapshot_lock(home), pytest.raises(SnapshotError, match="another"):
        main(["--only", "olaf", "--snapshot", "rotation"])
    assert not (home / RESULTS).exists()


def test_a_build_is_refused_while_a_run_queries_the_snapshot(
    data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refusals: list[str] = []

    def build_mid_query(self: FakeOlaf, wav_path: str) -> dict:
        with pytest.raises(SnapshotError, match="another") as refusal:
            build_snapshot("rotation", [])
        refusals.append(str(refusal.value))
        return MATCH

    monkeypatch.setattr(FakeOlaf, "recognize", build_mid_query)
    open_pool_db(data / "olaf" / "rotation" / "pool.db").close()
    assert main(["--only", "olaf", "--snapshot", "rotation", "--legs", "12s"]) == 0
    assert len(refusals) == 2  # one per query
    with snapshot_lock(data / "olaf" / "rotation"):  # the run released its lock
        pass


def test_a_snapshot_named_by_case_alone_is_refused_by_a_run(data: Path) -> None:
    open_pool_db(data / "olaf" / "rotation" / "pool.db").close()
    with pytest.raises(SnapshotError, match="case"):
        main(["--only", "olaf", "--snapshot", "Rotation"])


def test_a_run_below_five_gib_free_refuses_before_any_request(
    data: Path, fake: FakeShazam, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Usage:
        free = (5 << 30) - 1

    monkeypatch.setattr(run_mod.shutil, "disk_usage", lambda p: Usage)
    with pytest.raises(SystemExit, match="free"):
        main(["--only", "shazam", "--base-url", fake.url])
    assert fake.requests == []
    assert not (data / "shazam" / "throttle.json.lock").exists()  # refused before the lock


@pytest.mark.parametrize("flag", ["--store", "--state", "--work-dir", "--archive-dir"])
def test_a_path_in_the_checkout_is_refused_before_any_request(
    data: Path, fake: FakeShazam, flag: str
) -> None:
    with pytest.raises(DataPathError):
        main(["--only", "shazam", "--base-url", fake.url, flag, str(CHECKOUT / "run-guard" / "x")])
    assert fake.requests == []
    assert not (CHECKOUT / "run-guard").exists()
