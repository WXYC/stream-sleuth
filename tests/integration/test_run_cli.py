"""The leg runner's CLI end to end: real ffmpeg cuts against a localhost Shazam.

Hours are 20 s of sine, so each holds one 12 s clip and one 6 s clip (at 0 s).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from evaluation import run as run_mod
from evaluation.run import main
from stream_sleuth.paths import CHECKOUT, DataPathError
from tests.characterization.shazam_responses import NO_MATCH
from tests.shazam_fake import FakeShazam, json_response

pytestmark = pytest.mark.ffmpeg

HOUR = "2026/08/12/202608121600.mp3"
OTHER = "2026/08/12/202608121700.mp3"


@pytest.fixture(autouse=True)
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", "0")
    for key in (HOUR, OTHER):
        path = tmp_path / "data" / "archive" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
             "-c:a", "libmp3lame", "-b:a", "128k", str(path)],
            check=True,
        )  # fmt: skip
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


def test_legs_read_selection_json_and_share_one_throttle(data: Path, fake: FakeShazam) -> None:
    assert main(["--legs", "12s", "6s-subset", "--base-url", fake.url]) == 0
    # The 12 s leg takes both hours; the 6 s subset leg only the hour with subset: true.
    records = [
        json.loads(line) for line in (data / "shazam" / "results.jsonl").read_text().splitlines()
    ]
    assert [r["address"] for r in records] == [
        f"{HOUR}#0+12@128k",
        f"{OTHER}#0+12@128k",
        f"{HOUR}#0+6@128k",
    ]
    assert json.loads((data / "shazam" / "throttle.json").read_text())["count"] == 3
    assert list((data / "clips").iterdir()) == []


def test_a_run_below_five_gib_free_refuses_before_any_request(
    data: Path, fake: FakeShazam, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Usage:
        free = (5 << 30) - 1

    monkeypatch.setattr(run_mod.shutil, "disk_usage", lambda p: Usage)
    with pytest.raises(SystemExit, match="free"):
        main(["--base-url", fake.url])
    assert fake.requests == []
    assert not (data / "shazam" / "throttle.json.lock").exists()  # refused before the lock


@pytest.mark.parametrize("flag", ["--store", "--state", "--work-dir", "--archive-dir"])
def test_a_path_in_the_checkout_is_refused_before_any_request(
    data: Path, fake: FakeShazam, flag: str
) -> None:
    with pytest.raises(DataPathError):
        main(["--base-url", fake.url, flag, str(CHECKOUT / "run-guard" / "x")])
    assert fake.requests == []
    assert not (CHECKOUT / "run-guard").exists()
