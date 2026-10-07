"""The leg runner: the legs table, Shazam legs under one throttle, Olaf legs that resume, and the
free-space preflight.

No test contacts Shazam (a localhost ``FakeShazam``) or needs Olaf (a fake recognizer);
``hour_addresses`` and ``cut`` are replaced, so no test needs ffmpeg either.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import namedtuple
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from evaluation import run as run_mod
from evaluation.clips import ClipAddress
from evaluation.run import LEGS, main, preflight, read_hours, run_legs, run_olaf
from evaluation.shazam_eval import (
    MAX_RETRIES,
    CountingClient,
    ResultStore,
    ShazamOutcome,
    Throttle,
    budget_from_env,
    recognizer_identity,
)
from stream_sleuth.recognizers.base import EvalIdentification
from stream_sleuth.recognizers.olaf import OlafError
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity
from tests.characterization.shazam_responses import NO_MATCH
from tests.shazam_fake import HTML_429, FakeShazam, clock_at, json_response, write_tone

HOUR = "2026/08/12/202608121600.mp3"
OTHER = "2026/08/12/202608121700.mp3"
HOURS = {"all": [HOUR, OTHER], "subset": [HOUR]}

MATCH: EvalIdentification = {
    "artist": "Juana Molina",
    "song": "la paradoja",
    "album": "DOGA",
    "label": "",
    "source": "local",
    "confidence": 42.0,
    "query_offset_s": 3.5,
    "ref_start_s": 61.0,
    "ref_key": "ab" * 20,
}

Setup = namedtuple("Setup", "store client fake")


class FakeOlaf:
    """Answers every clip with ``match`` (None for no match) and counts the queries."""

    def __init__(
        self, match: EvalIdentification | None = MATCH, fail_on: int | None = None
    ) -> None:
        self.match, self.fail_on, self.calls = match, fail_on, 0

    def recognize(self, wav_path: str) -> EvalIdentification | None:
        self.calls += 1
        if self.calls == self.fail_on:
            raise KeyboardInterrupt  # stands in for the process dying here
        return self.match


Usage = namedtuple("Usage", "total used free")
FREE = Usage(0, 0, 1 << 40)


@pytest.fixture(autouse=True)
def fake_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two clips per hour, 0 s and 15 s, each 'cut' to a synthetic tone."""
    tone = tmp_path / "tone.wav"
    write_tone(tone)

    def addresses(keys, archive_dir, length_s, profile):  # noqa: ANN001, ANN202
        return [ClipAddress(k, 15 * i, length_s, profile) for k in keys for i in range(2)]

    @contextmanager
    def cut(address, hour_path, work_dir, *, wav=False) -> Iterator[Path]:  # noqa: ANN001
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


def test_the_legs_table_is_the_study_budget() -> None:
    assert len({leg.name for leg in LEGS}) == len(LEGS)
    shazam = [(leg.length_s, leg.profile, leg.hours) for leg in LEGS if "shazam" in leg.recognizers]
    assert shazam == [
        (12, "128k", "all"),
        (6, "128k", "subset"),
        (20, "128k", "subset"),
        (12, "320k", "subset"),
    ]
    olaf = [(leg.length_s, leg.profile, leg.hours) for leg in LEGS if "olaf" in leg.recognizers]
    assert sorted(
        olaf
    ) == [  # plan 5.2: every length on the full corpus, and 12 s at 320k on the subset
        (6, "128k", "all"),
        (12, "128k", "all"),
        (12, "320k", "subset"),
        (20, "128k", "all"),
    ]


def test_every_leg_reads_the_hours_of_selection_json(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    hours = {
        HOUR: {"group": "contrast", "band": "evening", "subset": True},
        OTHER: {"group": "contrast", "band": "evening", "subset": False},
    }
    path.write_text(json.dumps({"hours": hours}))
    assert read_hours(path) == {"all": [HOUR, OTHER], "subset": [HOUR]}


@pytest.mark.parametrize(
    ("hours", "names"),
    [
        ({HOUR: {"subset": True}, OTHER: {"subset": "false"}}, OTHER),  # a hand edit, truthy
        ({HOUR: {"subset": 1}}, HOUR),
        ({HOUR: {"group": "contrast"}}, HOUR),  # no subset label at all
        ({HOUR: True}, HOUR),
        ([HOUR, OTHER], "hours"),  # a list of keys, not an object keyed by hour
    ],
)
def test_a_selection_without_a_boolean_subset_per_hour_is_refused_naming_it(
    tmp_path: Path, hours: object, names: str
) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"hours": hours}))
    with pytest.raises(SystemExit, match=re.escape(names)) as refusal:
        read_hours(path)
    assert str(path) in str(refusal.value)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (None, "cannot read"),  # no file at all
        (f"{HOUR}\n{OTHER}\n", "not JSON"),  # hours.txt passed by mistake
        (json.dumps([{"hours": {}}]), "not an object"),
        (json.dumps({"hours": {}}), "no hours"),  # the subset legs would be done with 0 requests
        (json.dumps({"hours": {HOUR: {"subset": False}}}), "no hour with subset: true"),
    ],
)
def test_a_selection_that_cannot_drive_the_legs_is_refused_in_one_line(
    tmp_path: Path, text: str | None, problem: str
) -> None:
    path = tmp_path / "selection.json"
    if text is not None:
        path.write_text(text)
    with pytest.raises(SystemExit) as refusal:
        read_hours(path)
    message = str(refusal.value)
    assert message.startswith(f"{path}: ") and problem in message and "\n" not in message


def test_two_shazam_legs_share_one_throttle(tmp_path: Path, shazam) -> None:
    setup = shazam([json_response(200, NO_MATCH)] * 6)
    report = run_legs(
        legs("12s", "6s-subset"), HOURS, tmp_path, tmp_path, shazam=(setup.store, setup.client)
    )
    assert report == {"12s/shazam": "done", "6s-subset/shazam": "done"}
    assert len(setup.fake.requests) == 6  # 2 hours x 2 clips, then 1 hour x 2 clips
    assert json.loads((tmp_path / "throttle.json").read_text())["count"] == 6
    assert {r["recognizer"] for r in setup.store.records()} == {
        recognizer_identity(12),
        recognizer_identity(6),
    }


def test_the_320k_leg_queries_the_hours_the_128k_leg_already_scored(tmp_path: Path, shazam) -> None:
    setup = shazam([json_response(200, NO_MATCH)] * 4)
    one_hour = {"all": [HOUR], "subset": [HOUR]}
    report = run_legs(
        legs("12s", "12s-320k-subset"),
        one_hour,
        tmp_path,
        tmp_path,
        shazam=(setup.store, setup.client),
    )
    assert report == {"12s/shazam": "done", "12s-320k-subset/shazam": "done"}
    # One recognizer identity for both; only the address's profile keeps them apart.
    assert [r["address"] for r in setup.store.records()] == [
        f"{HOUR}#0+12@128k",
        f"{HOUR}#15+12@128k",
        f"{HOUR}#0+12@320k",
        f"{HOUR}#15+12@320k",
    ]
    assert {r["recognizer"] for r in setup.store.records()} == {recognizer_identity(12)}


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
        legs("12s", "6s-subset", "12s-320k-subset"),
        HOURS,
        tmp_path,
        tmp_path,
        shazam=(setup.store, setup.client),
    )
    assert report == {
        "12s/shazam": "rate_limited",
        "6s-subset/shazam": "not started",
        "12s-320k-subset/shazam": "not started",
    }
    assert len(setup.fake.requests) == 1
    assert grids == [12]  # a leg that is not started does not even decode its hours


def test_a_daily_cap_also_stops_the_later_legs(tmp_path: Path, shazam) -> None:
    setup = shazam([json_response(200, NO_MATCH)] * 4)
    clock = clock_at("2026-10-06T20:00:00")
    setup.client.throttle.close()
    with Throttle(tmp_path / "throttle.json", 3, 0.0, clock=clock, sleep=clock.sleep) as throttle:
        client = CountingClient(throttle, base_url=setup.fake.url)
        report = run_legs(
            legs("12s", "6s-subset"), HOURS, tmp_path, tmp_path, shazam=(setup.store, client)
        )
    assert report == {"12s/shazam": "daily_cap", "6s-subset/shazam": "not started"}
    assert len(setup.fake.requests) == 3


def test_a_future_dated_throttle_state_is_reported_as_the_leg_refused(
    tmp_path: Path, shazam
) -> None:
    setup = shazam([json_response(200, NO_MATCH)])
    ahead = clock_at("2026-10-06T21:00:00").now  # an hour ahead of the throttle's clock
    state = {"day": "2026-10-06", "count": 0, "last": ahead, "stopped": False}
    (tmp_path / "throttle.json").write_text(json.dumps(state))
    report = run_legs(
        legs("12s", "6s-subset"), HOURS, tmp_path, tmp_path, shazam=(setup.store, setup.client)
    )
    assert report == {"12s/shazam": "refused", "6s-subset/shazam": "not started"}
    assert setup.fake.requests == []


def test_a_stopped_shazam_day_does_not_stop_the_olaf_legs(tmp_path: Path, shazam) -> None:
    setup = shazam([HTML_429])
    olaf = FakeOlaf()
    report = run_legs(
        legs("12s"),
        HOURS,
        tmp_path,
        tmp_path,
        shazam=(setup.store, setup.client),
        olaf=(ResultStore(tmp_path / "results.jsonl"), olaf.recognize, "olaf@x"),
    )
    assert (report["12s/shazam"], report["12s/olaf"]) == ("rate_limited", "done")
    assert olaf.calls == 4


def test_a_refused_shazam_leg_leaves_the_olaf_legs_to_run(tmp_path: Path, shazam) -> None:
    setup = shazam([])
    ahead = clock_at("2026-10-06T21:00:00").now  # an hour ahead of the throttle's clock
    state = {"day": "2026-10-06", "count": 0, "last": ahead, "stopped": False}
    (tmp_path / "throttle.json").write_text(json.dumps(state))
    olaf = FakeOlaf()
    report = run_legs(
        legs("12s", "6s", "6s-subset"),
        HOURS,
        tmp_path,
        tmp_path,
        shazam=(setup.store, setup.client),
        olaf=(ResultStore(tmp_path / "results.jsonl"), olaf.recognize, "olaf@x"),
    )
    assert report == {
        "12s/shazam": "refused",
        "12s/olaf": "done",
        "6s/olaf": "done",
        "6s-subset/shazam": "not started",
    }
    assert olaf.calls == 8
    assert setup.fake.requests == []


def olaf_store(tmp_path: Path) -> ResultStore:
    return ResultStore(tmp_path / "results.jsonl")


def run_one_olaf(store: ResultStore, olaf: FakeOlaf, identity: str, n: int = 4) -> str:
    addresses = [ClipAddress(HOUR, 15 * i, 12, "128k") for i in range(n)]
    return run_olaf(
        addresses, lambda a: run_mod.cut(a, Path(), Path()), store, olaf.recognize, identity
    )


def test_an_olaf_leg_resumes_without_requerying_a_stored_address(tmp_path: Path) -> None:
    store, identity = olaf_store(tmp_path), "olaf@x"
    with pytest.raises(KeyboardInterrupt):
        run_one_olaf(store, FakeOlaf(fail_on=3), identity)
    assert len(store.records()) == 2
    again = FakeOlaf(match=None)
    assert run_one_olaf(store, again, identity) == "done"
    assert again.calls == 2  # only the two addresses that had no record
    assert [r["kind"] for r in store.records()] == ["matched", "matched", "no_match", "no_match"]
    third = FakeOlaf()
    assert run_one_olaf(store, third, identity) == "done"
    assert third.calls == 0
    assert len(store.records()) == 4


def test_an_olaf_error_aborts_the_run_and_stores_nothing_for_that_address(tmp_path: Path) -> None:
    class Broken:
        def recognize(self, wav_path: str) -> None:
            raise OlafError("no Olaf index")

    store = olaf_store(tmp_path)
    with pytest.raises(OlafError):
        run_one_olaf(store, Broken(), "olaf@x")  # type: ignore[arg-type]
    assert store.records() == []


def test_an_olaf_record_is_the_shazam_shape_plus_confidence_and_ref_key(tmp_path: Path) -> None:
    store = olaf_store(tmp_path)
    run_one_olaf(store, FakeOlaf(), "olaf@x", n=1)
    record = store.records()[0]
    assert record.keys() == {
        "address", "recognizer", "recorded_at", "status", "kind", "artist", "song", "album",
        "label", "offset_s", "confidence", "ref_key", "query_offset_s", "ref_start_s",
    }  # fmt: skip
    assert (record["status"], record["offset_s"]) == (0, None)  # 0: the subprocess answered
    assert (record["kind"], record["confidence"], record["ref_key"]) == ("matched", 42.0, "ab" * 20)
    assert store.history()[0] == {(f"{HOUR}#0+12@128k", "olaf@x")}


def test_a_result_filed_under_one_floor_is_not_reused_for_another(tmp_path: Path) -> None:
    store = olaf_store(tmp_path)
    loose, strict = olaf_identity("rotation", 5), olaf_identity("rotation", 12)
    run_one_olaf(store, FakeOlaf(), loose, n=2)
    again = FakeOlaf(match=None)
    run_one_olaf(store, again, strict, n=2)
    assert again.calls == 2  # the loose floor's results never stand in for the strict one's


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data directory whose selection.json labels HOUR subset and OTHER not."""
    data = tmp_path / "data"
    (data / "shazam").mkdir(parents=True)
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(data))
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", "0")
    monkeypatch.setattr(run_mod.shutil, "disk_usage", lambda p: FREE)  # never the host's disk
    labels = {HOUR: {"subset": True}, OTHER: {"subset": False}}
    (data / "selection.json").write_text(json.dumps({"hours": labels}))
    return data


def test_a_refusal_after_a_finished_leg_keeps_its_done_and_logs_the_summary(
    data: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = ResultStore(data / "shazam" / "results.jsonl")
    tired = f"{HOUR}#0+6@128k"
    for _ in range(MAX_RETRIES + 1):
        store.append(tired, recognizer_identity(6), ShazamOutcome(503, "server_error"))
    real = run_mod.hour_addresses

    def future_dated_after_the_first_leg(keys, archive_dir, length_s, profile):  # noqa: ANN001, ANN202
        if length_s == 6:  # the second leg: the clock now runs an hour behind the state file
            path = data / "shazam" / "throttle.json"
            path.write_text(
                json.dumps({**json.loads(path.read_text()), "last": time.time() + 3600})
            )
        return real(keys, archive_dir, length_s, profile)

    monkeypatch.setattr(run_mod, "hour_addresses", future_dated_after_the_first_leg)
    fake = FakeShazam([json_response(200, NO_MATCH)] * 4)
    try:
        with caplog.at_level(logging.INFO):
            code = main(
                [
                    "--only",
                    "shazam",
                    "--legs",
                    "12s",
                    "6s-subset",
                    "20s-subset",
                    "--base-url",
                    fake.url,
                ]
            )
    finally:
        fake.close()
    assert code == 1
    assert len(fake.requests) == 4  # the first leg's 2 hours x 2 clips, then nothing
    for line in (
        "12s/shazam: done",
        "6s-subset/shazam: refused",
        "20s-subset/shazam: not started",
        "4 requests",
    ):
        assert line in caplog.text
    assert "out of retries and not queried" in caplog.text
    assert tired in caplog.text


def test_the_cli_reports_every_leg_and_exits_non_zero_on_a_refused_state(
    data: Path, caplog: pytest.LogCaptureFixture
) -> None:
    state = {"day": "2026-10-06", "count": 0, "last": time.time() + 3600, "stopped": False}
    (data / "shazam" / "throttle.json").write_text(json.dumps(state))
    fake = FakeShazam([])
    try:
        with caplog.at_level(logging.INFO):
            assert (
                main(["--only", "shazam", "--legs", "12s", "6s-subset", "--base-url", fake.url])
                == 1
            )
    finally:
        fake.close()
    assert "is in the future" in caplog.text
    assert "12s/shazam: refused" in caplog.text
    assert "6s-subset/shazam: not started" in caplog.text
    assert fake.requests == []


def test_the_cli_takes_its_daily_cap_from_the_environment(
    data: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("STREAM_SLEUTH_SHAZAM_RATE_PER_DAY", "1")
    fake = FakeShazam([json_response(200, NO_MATCH)] * 2)
    try:
        with caplog.at_level(logging.INFO):
            assert (
                main(["--only", "shazam", "--legs", "12s", "6s-subset", "--base-url", fake.url])
                == 0
            )
    finally:
        fake.close()
    assert len(fake.requests) == 1
    assert "12s/shazam: daily_cap" in caplog.text
    assert "6s-subset/shazam: not started" in caplog.text
    assert "1 requests" in caplog.text


def test_the_cli_ends_with_the_out_of_retries_summary(
    data: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = ResultStore(data / "shazam" / "results.jsonl")
    tired = f"{HOUR}#0+6@128k"
    for _ in range(MAX_RETRIES + 1):
        store.append(tired, recognizer_identity(6), ShazamOutcome(503, "server_error"))
    fake = FakeShazam([json_response(200, NO_MATCH)])
    try:
        with caplog.at_level(logging.INFO):
            assert main(["--only", "shazam", "--legs", "6s-subset", "--base-url", fake.url]) == 0
    finally:
        fake.close()
    assert len(fake.requests) == 1  # the other clip; the tired one is not queried
    assert "out of retries and not queried" in caplog.text
    assert tired in caplog.text


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
