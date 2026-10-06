"""The Shazam CLI end to end against a localhost server, cutting real clips with ffmpeg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evaluation.shazam_eval import main
from tests.characterization.shazam_responses import JESSICA_PRATT
from tests.unit.test_shazam_eval import HTML_429, FakeShazam, _json

pytestmark = pytest.mark.ffmpeg

HOUR = "2026/08/12/202608121600.mp3"


def test_the_cli_stores_one_outcome_per_clip_and_stops_on_a_429(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", "0")
    archive = tmp_path / "archive"
    (archive / HOUR).parent.mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=40",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(archive / HOUR),
        ],
        check=True,
    )
    (tmp_path / "hours.txt").write_text(HOUR + "\n")
    (tmp_path / "work").mkdir()
    fake = FakeShazam([_json(200, JESSICA_PRATT), HTML_429])
    try:
        assert (
            main(
                [
                    "--hours",
                    str(tmp_path / "hours.txt"),
                    "--archive-dir",
                    str(archive),
                    "--work-dir",
                    str(tmp_path / "work"),
                    "--store",
                    str(tmp_path / "shazam.jsonl"),
                    "--state",
                    str(tmp_path / "throttle.json"),
                    "--base-url",
                    fake.url,
                ]
            )
            == 0
        )
    finally:
        fake.close()
    records = [json.loads(line) for line in (tmp_path / "shazam.jsonl").read_text().splitlines()]
    assert [(r["address"], r["kind"]) for r in records] == [
        (f"{HOUR}#0+12@128k", "matched"),
        (f"{HOUR}#15+12@128k", "rate_limited"),
    ]
    assert records[0]["artist"] == "Jessica Pratt"
    assert json.loads((tmp_path / "throttle.json").read_text())["stopped"] is True
    assert list((tmp_path / "work").iterdir()) == []
