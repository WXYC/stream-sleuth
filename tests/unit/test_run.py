"""The leg runner: the legs table, Shazam legs under one throttle, and the free-space preflight.

No test contacts Shazam (a localhost ``FakeShazam``); ``hour_addresses`` and ``cut`` are
replaced, so no test needs ffmpeg either.
"""

from __future__ import annotations

import json
from collections import namedtuple
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from evaluation import run as run_mod
from evaluation.clips import ClipAddress
from evaluation.run import LEGS, preflight, run_legs, selected_hours
from evaluation.shazam_eval import (
    CountingClient,
    ResultStore,
    Throttle,
    budget_from_env,
    recognizer_identity,
)
from tests.characterization.shazam_responses import NO_MATCH
from tests.shazam_fake import HTML_429, FakeShazam, clock_at, json_response, write_tone

HOUR = "2026/08/12/202608121600.mp3"
OTHER = "2026/08/12/202608121700.mp3"
HOURS = {"all": [HOUR, OTHER], "subset": [HOUR]}

Setup = namedtuple("Setup", "store client fake")


@pytest.fixture(autouse=True)
def fake_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two clips per hour, 0 s and 15 s, each 'cut' to a synthetic tone."""
    tone = tmp_path / "tone.wav"
    write_tone(tone)

    def addresses(keys, archive_dir, length_s, profile):  # noqa: ANN001, ANN202
        return [ClipAddress(k, 15 * i, length_s, profile) for k in keys for i in range(2)]

    @contextmanager
    def cut(address, hour_path, work_dir) -> Iterator[Path]:  # noqa: ANN001
        yield tone

    monkeypatch.setattr(run_mod, "hour_addresses", addresses)
    monkeypatch.setattr(run_mod, "cut", cut)


@pytest.fixture
def shazam(tmp_path: Path) -> Iterator[Callable[[list[tuple]], Setup]]:
    fakes: list[FakeShazam] = []

    def make(responses: list[tuple]) -> Setup:
        fake = FakeShazam(responses)
        fakes.append(fake)
        clock = clock_at("2026-10-06T20:00:00")
        throttle = Throttle(tmp_path / "throttle.json", 500, 0.0, clock=clock, sleep=clock.sleep)
        client = CountingClient(throttle, base_url=fake.url)
        return Setup(ResultStore(tmp_path / "shazam.jsonl"), client, fake)

    yield make
    for fake in fakes:
        fake.close()


def legs(*names: str):
    return [leg for leg in LEGS if leg.name in names]


def test_the_legs_table_is_the_shazam_query_budget() -> None:
    assert [(leg.length_s, leg.profile, leg.hours) for leg in LEGS] == [
        (12, "128k", "all"),
        (6, "128k", "subset"),
        (20, "128k", "subset"),
        (12, "320k", "subset"),
    ]
    assert len({leg.name for leg in LEGS}) == len(LEGS)


def test_every_leg_reads_the_hours_of_selection_json(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    hours = {
        HOUR: {"group": "contrast", "band": "evening", "subset": True},
        OTHER: {"group": "contrast", "band": "evening", "subset": False},
    }
    path.write_text(json.dumps({"hours": hours}))
    assert selected_hours(path, subset_only=False) == [HOUR, OTHER]
    assert selected_hours(path, subset_only=True) == [HOUR]


def test_two_shazam_legs_share_one_throttle(tmp_path: Path, shazam) -> None:
    setup = shazam([json_response(200, NO_MATCH)] * 6)
    report = run_legs(legs("12s", "6s-subset"), HOURS, tmp_path, tmp_path, *setup[:2])
    assert report == {"12s": "done", "6s-subset": "done"}
    assert len(setup.fake.requests) == 6  # 2 hours x 2 clips, then 1 hour x 2 clips
    assert json.loads((tmp_path / "throttle.json").read_text())["count"] == 6
    assert {r["recognizer"] for r in setup.store.records()} == {
        recognizer_identity(12),
        recognizer_identity(6),
    }


def test_a_leg_that_is_stopped_stops_the_next_leg_before_it_sends_anything(
    tmp_path: Path, shazam, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = shazam([HTML_429, json_response(200, NO_MATCH)])
    grids: list[int] = []
    real = run_mod.hour_addresses

    def spy(keys, archive_dir, length_s, profile):  # noqa: ANN001, ANN202
        grids.append(length_s)
        return real(keys, archive_dir, length_s, profile)

    monkeypatch.setattr(run_mod, "hour_addresses", spy)
    report = run_legs(
        legs("12s", "6s-subset", "12s-320k-subset"), HOURS, tmp_path, tmp_path, *setup[:2]
    )
    assert report == {
        "12s": "rate_limited",
        "6s-subset": "not started",
        "12s-320k-subset": "not started",
    }
    assert len(setup.fake.requests) == 1
    assert grids == [12]  # a leg that is not started does not even decode its hours


def test_a_daily_cap_also_stops_the_later_legs(tmp_path: Path, shazam) -> None:
    setup = shazam([json_response(200, NO_MATCH)] * 4)
    clock = clock_at("2026-10-06T20:00:00")
    setup.client.throttle.close()
    with Throttle(tmp_path / "throttle.json", 3, 0.0, clock=clock, sleep=clock.sleep) as throttle:
        client = CountingClient(throttle, base_url=setup.fake.url)
        report = run_legs(legs("12s", "6s-subset"), HOURS, tmp_path, tmp_path, setup.store, client)
    assert report == {"12s": "daily_cap", "6s-subset": "not started"}
    assert len(setup.fake.requests) == 3


def test_preflight_refuses_below_five_gib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(run_mod.shutil, "disk_usage", lambda p: usage(0, 0, (5 << 30) - 1))
    with pytest.raises(SystemExit, match="free"):
        preflight(tmp_path)
    monkeypatch.setattr(run_mod.shutil, "disk_usage", lambda p: usage(0, 0, 5 << 30))
    preflight(tmp_path)


def test_the_budget_variables_are_read_in_one_place(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STREAM_SLEUTH_SHAZAM_RATE_PER_DAY", raising=False)
    monkeypatch.delenv("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", raising=False)
    assert budget_from_env() == (500, 20.0)
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_RATE_PER_DAY", "7")
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", "1.5")
    assert budget_from_env() == (7, 1.5)
