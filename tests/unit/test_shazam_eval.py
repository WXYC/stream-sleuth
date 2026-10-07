"""Shazam evaluation adapter: outcomes, throttle, day-stop, and store, against a localhost server.

Every test points the client at a localhost ``http.server`` through the required
``base_url``; no test contacts Shazam. Signatures are computed offline by
``shazamio-core`` from a synthetic two-tone WAV, so ``Shazam.recognize`` runs its
real path up to the HTTP request.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pytest

from evaluation import run as run_mod
from evaluation import shazam_eval
from evaluation.clips import ClipAddress, ClipError
from evaluation.shazam_eval import (
    MAX_FAILURE_STREAK,
    MAX_RETRIES,
    SHAZAMIO_VERSION,
    CountingClient,
    FutureStateError,
    ResultStore,
    ShazamOutcome,
    Throttle,
    ThrottleBusyError,
    outcome_from,
    recognizer_identity,
    require_pinned_shazamio,
    run,
)
from stream_sleuth.paths import CHECKOUT, DataPathError
from tests.characterization.shazam_responses import JUANA_MOLINA, NO_MATCH
from tests.shazam_fake import (
    HTML_429,
    FakeClock,
    FakeShazam,
    clock_at,
    json_response,
    write_tone,
)

HOUR = "2026/08/12/202608121600.mp3"


class SimulatedCrashError(Exception):
    """Stands in for the process dying at the point it is raised."""


@pytest.fixture
def server() -> Iterator[list[FakeShazam]]:
    made: list[FakeShazam] = []
    yield made
    for s in made:
        s.close()


@pytest.fixture(scope="module")
def tone(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "tone.wav"
    write_tone(path)
    return path


def _addresses(n: int) -> list[ClipAddress]:
    return [ClipAddress(HOUR, 15 * i, 12) for i in range(n)]


def _run(
    tmp_path: Path,
    tone: Path,
    fake: FakeShazam,
    n: int,
    *,
    clock: FakeClock | None = None,
    rate: int = 500,
    interval: float = 20.0,
) -> tuple[ResultStore, Throttle, CountingClient, str]:
    clock = clock or clock_at("2026-10-06T20:00:00")
    store = ResultStore(tmp_path / "shazam.jsonl")

    @contextmanager
    def open_clip(address: ClipAddress) -> Iterator[Path]:
        yield tone

    # A fresh Throttle per call is a restarted process: only the state file carries over.
    with Throttle(
        tmp_path / "throttle.json", rate, interval, clock=clock, sleep=clock.sleep
    ) as throttle:
        client = CountingClient(throttle, base_url=fake.url)
        stop = asyncio.run(run(_addresses(n), open_clip, store, client))
    return store, throttle, client, stop


@pytest.mark.parametrize(
    ("status", "body", "kind", "artist", "offset"),
    [
        (200, JUANA_MOLINA, "matched", "Juana Molina", 41.2),
        (200, NO_MATCH, "no_match", "", None),
        (429, {"error": "rate"}, "rate_limited", "", None),
        (503, {"error": "busy"}, "server_error", "", None),
        (200, None, "decode_error", "", None),
        (429, None, "rate_limited", "", None),
        (200, {"track": "Juana Molina"}, "decode_error", "", None),
        (200, {"track": {"subtitle": "Juana Molina"}, "matches": ["x"]}, "decode_error", "", None),
        (200, {"track": {"sections": [None]}}, "decode_error", "", None),
        (200, {"track": {"title": "x"}, "matches": [{"offset": "soon"}]}, "decode_error", "", None),
    ],
)
def test_outcome_from_classifies_status_and_body(
    status: int, body: dict | None, kind: str, artist: str, offset: float | None
) -> None:
    got = outcome_from(status, body)
    assert (got.status, got.kind, got.artist, got.offset_s) == (status, kind, artist, offset)


def test_a_match_carries_the_four_wire_fields() -> None:
    got = outcome_from(200, JUANA_MOLINA)
    assert (got.artist, got.song, got.album, got.label) == (
        "Juana Molina",
        "la paradoja",
        "DOGA",
        "Sonamos",
    )


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (json_response(200, JUANA_MOLINA), "matched"),
        (json_response(200, NO_MATCH), "no_match"),
        (json_response(429, {"error": "rate limited"}), "rate_limited"),
        (HTML_429, "rate_limited"),
        ((429, b'{"error": "rate', "application/json", 100), "rate_limited"),  # body cut short
        (json_response(503, {"error": "busy"}), "server_error"),
        ((200, b"<html>maintenance</html>", "text/html"), "decode_error"),
        (json_response(200, {"track": "Juana Molina"}), "decode_error"),
    ],
)
def test_each_response_is_one_request_and_one_stored_outcome(
    tmp_path: Path, tone: Path, server: list[FakeShazam], response: tuple, kind: str
) -> None:
    fake = FakeShazam([response, json_response(200, NO_MATCH)])
    server.append(fake)
    clock = clock_at("2026-10-06T20:00:00")
    store, _, client, stop = _run(tmp_path, tone, fake, 2, clock=clock)
    stops = kind == "rate_limited"  # only a 429 ends the day; every other kind runs on
    assert stop == ("rate_limited" if stops else "done")
    assert len(fake.requests) == client.requests == (1 if stops else 2)
    assert [r["kind"] for r in store.records()] == ([kind] if stops else [kind, "no_match"])
    record = store.records()[0]
    assert record["address"] == str(_addresses(1)[0])
    assert record["recognizer"] == recognizer_identity(12)
    assert clock.slept == ([] if stops else [20.0])
    assert json.loads((tmp_path / "throttle.json").read_text())["stopped"] == (
        "rate_limited" if stops else False
    )


def test_matched_offset_is_extracted_from_the_wire(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    fake = FakeShazam([json_response(200, JUANA_MOLINA)])
    server.append(fake)
    store, *_ = _run(tmp_path, tone, fake, 1)
    assert store.records()[0]["offset_s"] == 41.2


def test_requests_are_spaced_by_the_minimum_interval(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    fake = FakeShazam([json_response(200, NO_MATCH)] * 3)
    server.append(fake)
    clock = clock_at("2026-10-06T20:00:00")
    _run(tmp_path, tone, fake, 3, clock=clock, interval=20.0)
    assert clock.slept == [20.0, 20.0]
    assert len(fake.requests) == 3


@pytest.mark.parametrize("response", [json_response(429, {"error": "rate limited"}), HTML_429])
def test_a_429_stops_the_day_and_survives_a_restart(
    tmp_path: Path, tone: Path, server: list[FakeShazam], response: tuple
) -> None:
    fake = FakeShazam([json_response(200, NO_MATCH), response, json_response(200, NO_MATCH)])
    server.append(fake)
    clock = clock_at("2026-10-06T20:00:00")
    store, _, _, stop = _run(tmp_path, tone, fake, 3, clock=clock)
    assert stop == "rate_limited"
    assert len(fake.requests) == 2
    assert [r["kind"] for r in store.records()] == ["no_match", "rate_limited"]

    # Same UTC day, new process: nothing is sent.
    _, _, client, stop = _run(tmp_path, tone, fake, 3, clock=clock)
    assert (stop, client.requests, len(fake.requests)) == ("rate_limited", 0, 2)


def test_a_429_stop_is_persisted_before_its_record(
    tmp_path: Path, tone: Path, server: list[FakeShazam], monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(*args: object) -> None:
        raise SimulatedCrashError

    fake = FakeShazam([HTML_429])
    server.append(fake)
    monkeypatch.setattr(ResultStore, "append", crash)
    with pytest.raises(SimulatedCrashError):
        _run(tmp_path, tone, fake, 1)
    assert json.loads((tmp_path / "throttle.json").read_text())["stopped"] == "rate_limited"


def test_the_daily_cap_counts_requests_across_restarts(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    fake = FakeShazam([json_response(200, NO_MATCH)] * 3)
    server.append(fake)
    clock = clock_at("2026-10-06T20:00:00")
    _, _, _, stop = _run(tmp_path, tone, fake, 2, clock=clock, rate=3)
    assert stop == "done"
    store, _, client, stop = _run(tmp_path, tone, fake, 5, clock=clock, rate=3)
    assert (stop, client.requests, len(fake.requests)) == ("daily_cap", 1, 3)
    assert len(store.records()) == 3


def test_a_new_utc_day_resets_the_cap_and_the_stop(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    fake = FakeShazam([HTML_429, json_response(200, NO_MATCH)])
    server.append(fake)
    clock = clock_at("2026-10-06T23:59:00")
    _run(tmp_path, tone, fake, 1, clock=clock)
    clock.now += 120  # 00:01 UTC on 2026-10-07
    store, _, client, stop = _run(tmp_path, tone, fake, 1, clock=clock)
    assert (stop, client.requests) == ("done", 1)
    assert [r["kind"] for r in store.records()] == ["rate_limited", "no_match"]


@pytest.mark.parametrize(
    ("first_at", "first_n", "restart_at", "rate", "slept"),
    [
        # Same UTC day: requests at :00 and :20, restart at :25. One more fits the cap of 3.
        ("2026-10-06T20:00:00", 2, "2026-10-06T20:00:25", 3, [15.0]),
        # Across UTC midnight: the cap of 1 resets, but the interval since 23:59:55 holds.
        ("2026-10-06T23:59:55", 1, "2026-10-07T00:00:05", 1, [10.0]),
    ],
)
def test_a_restart_honors_the_persisted_interval_and_count(
    tmp_path: Path,
    tone: Path,
    server: list[FakeShazam],
    first_at: str,
    first_n: int,
    restart_at: str,
    rate: int,
    slept: list[float],
) -> None:
    fake = FakeShazam([json_response(200, NO_MATCH)] * (first_n + 1))
    server.append(fake)
    _run(tmp_path, tone, fake, first_n, clock=clock_at(first_at), rate=rate)
    restarted = clock_at(restart_at)  # a new process: a new clock, nothing in memory
    _, _, client, stop = _run(tmp_path, tone, fake, first_n + 3, clock=restarted, rate=rate)
    assert restarted.slept == slept
    assert (stop, client.requests, len(fake.requests)) == ("daily_cap", 1, first_n + 1)


def test_the_interval_is_checked_again_after_every_sleep(tmp_path: Path) -> None:
    clock = clock_at("2026-10-06T20:00:00")
    sent: list[float] = []

    async def one(throttle: Throttle) -> None:
        await throttle.acquire()
        sent.append(clock())

    async def two_at_once(throttle: Throttle) -> None:
        await asyncio.gather(one(throttle), one(throttle))

    with Throttle(tmp_path / "throttle.json", 500, 20.0, clock=clock, sleep=clock.sleep) as t:
        asyncio.run(one(t))
        clock.now += 5
        asyncio.run(two_at_once(t))  # both wake from the same wait; the second must wait again
    assert [b - a for a, b in zip(sent, sent[1:], strict=False)] == [20.0, 20.0]


def test_a_clock_stepped_back_across_midnight_keeps_the_later_days_stop(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    fake = FakeShazam([HTML_429, json_response(200, NO_MATCH)])
    server.append(fake)
    _run(tmp_path, tone, fake, 1, clock=clock_at("2026-10-07T00:10:00"))
    stepped_back = clock_at("2026-10-06T23:58:00")
    _, _, client, stop = _run(tmp_path, tone, fake, 1, clock=stepped_back)
    assert (stop, client.requests, len(fake.requests)) == ("rate_limited", 0, 1)
    assert stepped_back.slept == []  # refused at once, not after sleeping into the later day


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


@pytest.mark.parametrize("ahead", [21.0, 3600.0, 90 * 86400.0])
def test_a_future_dated_last_is_refused_with_one_error_and_no_sleep(
    tmp_path: Path, ahead: float
) -> None:
    state = tmp_path / "throttle.json"
    clock = clock_at("2026-10-06T20:00:00")
    with Throttle(state, 500, 20.0, clock=clock, sleep=clock.sleep) as throttle:
        asyncio.run(throttle.acquire())
    last = clock()
    clock.now -= ahead  # a clock stepped back, or a state file written by one running ahead
    with Throttle(state, 500, 20.0, clock=clock, sleep=clock.sleep) as throttle:
        with pytest.raises(FutureStateError, match="future") as refused:
            asyncio.run(throttle.acquire())
    for expected in (str(state), _iso(last), _iso(clock())):
        assert expected in str(refused.value)
    assert clock.slept == []
    assert json.loads(state.read_text())["count"] == 1  # nothing was counted


@pytest.mark.parametrize(
    ("elapsed", "slept", "logged"),
    [
        (5.0, 15.0, False),  # an ordinary restart: the rest of the interval
        (0.0, 20.0, False),  # exactly the interval is not long
        (-5.0, 25.0, True),  # last is slightly ahead: longer than the interval
        (-20.0, 40.0, True),  # the most that is not refused
    ],
)
def test_a_wait_longer_than_the_interval_is_logged_before_sleeping(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, elapsed: float, slept: float, logged: bool
) -> None:
    state = tmp_path / "throttle.json"
    clock = clock_at("2026-10-06T20:00:00")
    with Throttle(state, 500, 20.0, clock=clock, sleep=clock.sleep) as throttle:
        asyncio.run(throttle.acquire())
        clock.slept.clear()
        clock.now += elapsed
        with caplog.at_level("WARNING"):
            asyncio.run(throttle.acquire())
    assert clock.slept == [slept]
    assert bool(caplog.records) is logged
    assert (str(state) in caplog.text) is logged


def test_resume_skips_scoring_outcomes_and_retries_errors(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    fake = FakeShazam(
        [
            json_response(200, JUANA_MOLINA),
            json_response(503, {}),
            json_response(200, NO_MATCH),
            json_response(200, NO_MATCH),
        ]
    )
    server.append(fake)
    clock = clock_at("2026-10-06T20:00:00")  # shared: a restart's clock never runs behind the file
    _run(tmp_path, tone, fake, 3, clock=clock)
    store, _, client, _ = _run(tmp_path, tone, fake, 3, clock=clock)
    assert client.requests == 1  # only the 503 address is retried
    kinds = [(r["address"], r["kind"]) for r in store.records()]
    second = str(_addresses(3)[1])
    assert kinds[1] == (second, "server_error") and kinds[3] == (second, "no_match")


def test_a_clip_error_skips_that_address_without_a_query_or_a_record(
    tmp_path: Path, tone: Path, server: list[FakeShazam], caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeShazam([json_response(200, NO_MATCH)] * 2)
    server.append(fake)
    clock = clock_at("2026-10-06T20:00:00")
    throttle = Throttle(tmp_path / "throttle.json", 500, 20.0, clock=clock, sleep=clock.sleep)
    client = CountingClient(throttle, base_url=fake.url)
    store = ResultStore(tmp_path / "shazam.jsonl")
    bad = _addresses(3)[1]

    @contextmanager
    def open_clip(address: ClipAddress) -> Iterator[Path]:
        if address == bad:
            raise ClipError(f"{address} runs past the end")
        yield tone

    with caplog.at_level("WARNING"):
        stop = asyncio.run(run(_addresses(3), open_clip, store, client))
    assert (stop, client.requests, len(fake.requests)) == ("done", 2, 2)
    assert [r["address"] for r in store.records()] == [
        str(a) for a in (_addresses(3)[0], _addresses(3)[2])
    ]
    assert str(bad) in caplog.text
    assert clock.slept == [20.0]  # the skipped address spent neither a query nor an interval


def test_the_writers_refuse_a_path_in_the_checkout(tmp_path: Path) -> None:
    with pytest.raises(DataPathError):
        ResultStore(CHECKOUT / "results.jsonl")
    with pytest.raises(DataPathError):
        Throttle(CHECKOUT / "throttle.json", 500, 20.0)


def test_the_store_only_appends(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "shazam.jsonl")
    store.append("a#0+12@128k", recognizer_identity(12), ShazamOutcome(200, "no_match"))
    before = (tmp_path / "shazam.jsonl").read_text()
    ResultStore(tmp_path / "shazam.jsonl").append(
        "b#0+12@128k", recognizer_identity(12), ShazamOutcome(200, "no_match")
    )
    assert (tmp_path / "shazam.jsonl").read_text().startswith(before)
    assert len(ResultStore(tmp_path / "shazam.jsonl").records()) == 2


def test_a_torn_line_is_skipped_and_the_next_append_starts_a_new_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "shazam.jsonl"
    store = ResultStore(path)
    store.append("a#0+12@128k", recognizer_identity(12), ShazamOutcome(200, "no_match"))
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"address": "b#0+12@128k", "rec')  # a crash or a full disk mid-write
    torn = path.read_text()
    with caplog.at_level("WARNING"):
        assert [r["address"] for r in store.records()] == ["a#0+12@128k"]
    assert str(path) in caplog.text
    store.append("c#0+12@128k", recognizer_identity(12), ShazamOutcome(200, "no_match"))
    assert path.read_text().startswith(torn + "\n")  # the torn bytes are kept, never rewritten
    assert [r["address"] for r in store.records()] == ["a#0+12@128k", "c#0+12@128k"]


def test_a_line_torn_inside_a_multibyte_character_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "shazam.jsonl"
    store = ResultStore(path)
    store.append("a#0+12@128k", recognizer_identity(12), ShazamOutcome(200, "no_match"))
    line = json.dumps(
        {"address": "b#0+12@128k", "artist": "Hermanos Gutiérrez"}, ensure_ascii=False
    )
    raw = line.encode("utf-8")
    cut = raw.index("é".encode()) + 1  # between the two bytes of é
    with open(path, "ab") as f:
        f.write(raw[:cut])
    with caplog.at_level("WARNING"):
        assert [r["address"] for r in store.records()] == ["a#0+12@128k"]
    assert str(path) in caplog.text
    store.append("c#0+12@128k", recognizer_identity(12), ShazamOutcome(200, "no_match"))
    assert [r["address"] for r in store.records()] == ["a#0+12@128k", "c#0+12@128k"]


def test_a_streak_of_non_scoring_outcomes_stops_the_day(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    n = MAX_FAILURE_STREAK + 5
    fake = FakeShazam([json_response(503, {"error": "busy"})] * n)
    server.append(fake)
    store, throttle, client, stop = _run(tmp_path, tone, fake, n)
    assert (stop, client.requests) == ("failure_streak", MAX_FAILURE_STREAK)
    # The stop persists with its own reason, so a rerun the same day sends nothing.
    _, _, rerun, again = _run(tmp_path, tone, fake, n)
    assert (again, rerun.requests) == ("failure_streak", 0)


def test_never_tried_addresses_go_before_retries_so_a_bad_stretch_cannot_stall_the_leg(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    n = MAX_FAILURE_STREAK + 3
    fake = FakeShazam(
        [json_response(503, {})] * MAX_FAILURE_STREAK + [json_response(200, NO_MATCH)] * n
    )
    server.append(fake)
    _run(tmp_path, tone, fake, n)  # day one: the first 20 fail and the streak stops the day
    store, _, _, stop = _run(tmp_path, tone, fake, n, clock=clock_at("2026-10-07T20:00:00"))
    day_two = [r["address"] for r in store.records()][MAX_FAILURE_STREAK:]
    expected = _addresses(n)[MAX_FAILURE_STREAK:] + _addresses(n)[:MAX_FAILURE_STREAK]
    assert (stop, day_two) == ("done", [str(a) for a in expected])


def _later(clock: FakeClock, days: int = 1) -> FakeClock:
    clock.now += days * 86400  # the next UTC day, so a stop or the cap does not carry over
    return clock


def test_an_address_is_retried_at_most_max_retries_times_then_not_queried(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    attempts = 1 + MAX_RETRIES
    fake = FakeShazam([json_response(503, {})] * attempts + [json_response(200, NO_MATCH)])
    server.append(fake)
    clock = clock_at("2026-10-06T20:00:00")
    for _ in range(attempts):
        store, _, client, stop = _run(tmp_path, tone, fake, 1, clock=clock)
        assert (stop, client.requests) == ("done", 1)
    assert store.history()[2] == {(str(_addresses(1)[0]), recognizer_identity(12))}
    _, _, client, stop = _run(tmp_path, tone, fake, 1, clock=clock)
    assert (stop, client.requests, len(fake.requests)) == ("done", 0, attempts)


def test_an_exhausted_address_does_not_block_the_others(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    fake = FakeShazam(
        [json_response(503, {})] * (1 + MAX_RETRIES) + [json_response(200, NO_MATCH)] * 2
    )
    server.append(fake)
    clock = clock_at("2026-10-06T20:00:00")
    for _ in range(1 + MAX_RETRIES):
        _run(tmp_path, tone, fake, 1, clock=clock)
    store, _, client, stop = _run(tmp_path, tone, fake, 3, clock=clock)
    assert (stop, client.requests) == ("done", 2)
    assert [r["address"] for r in store.records()][-2:] == [str(a) for a in _addresses(3)[1:]]


@pytest.mark.parametrize("status", [429, 403])
def test_a_stop_status_is_the_load_or_the_policy_not_the_addresss_fault(
    tmp_path: Path, tone: Path, server: list[FakeShazam], status: int
) -> None:
    fake = FakeShazam(
        [json_response(status, {})] * (1 + MAX_RETRIES) + [json_response(200, NO_MATCH)]
    )
    server.append(fake)
    clock = clock_at("2026-10-06T20:00:00")
    for _ in range(1 + MAX_RETRIES):
        _run(tmp_path, tone, fake, 1, clock=_later(clock))
    store, _, client, stop = _run(tmp_path, tone, fake, 1, clock=_later(clock))
    assert (stop, client.requests) == ("done", 1)
    assert store.history()[2] == set()


def test_the_history_is_the_one_source_of_scored_last_status_and_exhausted(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "shazam.jsonl")
    who = recognizer_identity(12)
    fail = [(503, "server_error"), (200, "decode_error")]
    many = 2 * (1 + MAX_RETRIES)
    rows = [
        ("scored", 200, "matched"),
        ("rescored", *fail[0]),  # failed once, then scored: never exhausted
        ("rescored", 200, "no_match"),
        *[("flaky", *fail[i % 2]) for i in range(MAX_RETRIES)],  # one short of exhausted
        *[("dead", *fail[i % 2]) for i in range(1 + MAX_RETRIES)],  # first failure + the retries
        *[("loaded", 429, "rate_limited")] * many,  # a stop status is not the address's fault
        *[("denied", 403, "server_error")] * many,
    ]
    for address, status, kind in rows:
        store.append(address, who, ShazamOutcome(status, kind))
    scored, last, exhausted = store.history()
    assert scored == {("scored", who), ("rescored", who)}
    assert last == {
        (a, who): s
        for a, s in (
            ("scored", 200),
            ("rescored", 200),
            ("flaky", fail[(MAX_RETRIES - 1) % 2][0]),
            ("dead", fail[MAX_RETRIES % 2][0]),
            ("loaded", 429),
            ("denied", 403),
        )
    }
    assert exhausted == {("dead", who)}


def test_a_403_stops_the_day_as_forbidden_after_one_request(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    fake = FakeShazam(
        [json_response(403, {"error": "forbidden"})] + [json_response(200, NO_MATCH)] * 2
    )
    server.append(fake)
    clock = clock_at("2026-10-06T20:00:00")
    store, throttle, client, stop = _run(tmp_path, tone, fake, 3, clock=clock)
    assert (stop, client.requests, len(fake.requests)) == ("forbidden", 1, 1)
    assert [(r["status"], r["kind"]) for r in store.records()] == [(403, "server_error")]
    assert json.loads((tmp_path / "throttle.json").read_text())["stopped"] == "forbidden"
    # The stop persists with its own reason: a rerun the same day sends nothing.
    _, _, rerun, again = _run(tmp_path, tone, fake, 3, clock=clock)
    assert (again, rerun.requests, len(fake.requests)) == ("forbidden", 0, 1)


def test_a_scoring_outcome_resets_the_failure_streak(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    almost = [json_response(503, {})] * (MAX_FAILURE_STREAK - 1)
    fake = FakeShazam([*almost, json_response(200, NO_MATCH), *almost])
    server.append(fake)
    _, _, client, stop = _run(tmp_path, tone, fake, 2 * MAX_FAILURE_STREAK - 1)
    assert (stop, client.requests) == ("done", 2 * MAX_FAILURE_STREAK - 1)


def test_the_client_requires_an_explicit_base_url(tmp_path: Path) -> None:
    with Throttle(tmp_path / "t.json", 500, 20.0) as throttle, pytest.raises(TypeError):
        CountingClient(throttle)  # type: ignore[call-arg]


_OPEN_IN_A_CHILD = (
    "import sys; from pathlib import Path; from evaluation.shazam_eval import Throttle; "
    "Throttle(Path(sys.argv[1]), 500, 20.0)"
)


def test_one_process_holds_a_state_file_and_a_second_opener_is_refused(tmp_path: Path) -> None:
    state = tmp_path / "throttle.json"
    first = Throttle(state, 500, 20.0)
    with pytest.raises(ThrottleBusyError, match="another run holds it") as refused:
        Throttle(state, 500, 20.0)
    assert str(state) in str(refused.value)
    child = subprocess.run(
        [sys.executable, "-c", _OPEN_IN_A_CHILD, str(state)],
        cwd=CHECKOUT,
        capture_output=True,
        text=True,
    )
    assert child.returncode != 0
    assert "ThrottleBusyError" in child.stderr and str(state) in child.stderr
    first.close()
    with pytest.raises(ValueError, match="closed"):
        asyncio.run(first.acquire())
    with Throttle(state, 500, 20.0):  # released: the next run opens it
        pass
    assert (
        subprocess.run(
            [sys.executable, "-c", _OPEN_IN_A_CHILD, str(state)], cwd=CHECKOUT
        ).returncode
        == 0
    )


def test_a_request_is_counted_before_it_is_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(*args: object, **kwargs: object) -> None:
        raise SimulatedCrashError

    monkeypatch.setattr(aiohttp.ClientSession, "request", crash)
    clock = clock_at("2026-10-06T20:00:00")
    with Throttle(tmp_path / "throttle.json", 500, 20.0, clock=clock, sleep=clock.sleep) as t:
        with pytest.raises(SimulatedCrashError):
            asyncio.run(CountingClient(t, base_url="http://127.0.0.1:9").request("POST", "/x"))
    clock.now += 5
    with Throttle(tmp_path / "throttle.json", 500, 20.0, clock=clock, sleep=clock.sleep) as t:
        asyncio.run(t.acquire())
    assert clock.slept == [15.0]  # the unsent request was counted and its time kept
    assert json.loads((tmp_path / "throttle.json").read_text())["count"] == 2


def _go(
    tmp_path: Path,
    tone: Path,
    fake: FakeShazam,
    addresses: list[ClipAddress],
    store: ResultStore | None = None,
    *,
    clock: FakeClock | None = None,
) -> tuple[ResultStore, CountingClient, str]:
    """``run`` over exactly ``addresses`` (``_run`` always builds a prefix of the grid)."""
    clock = clock or clock_at("2026-10-06T20:00:00")
    store = store or ResultStore(tmp_path / "shazam.jsonl")

    @contextmanager
    def open_clip(address: ClipAddress) -> Iterator[Path]:
        yield tone

    with Throttle(
        tmp_path / "throttle.json", 500, 20.0, clock=clock, sleep=clock.sleep
    ) as throttle:
        client = CountingClient(throttle, base_url=fake.url)
        return store, client, asyncio.run(run(addresses, open_clip, store, client))


def test_a_repeated_address_is_queried_and_stored_once(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    a, b, _ = _addresses(3)
    fake = FakeShazam(
        [
            json_response(200, NO_MATCH),
            json_response(200, JUANA_MOLINA),
            json_response(200, NO_MATCH),
        ]
    )
    server.append(fake)
    store, client, stop = _go(tmp_path, tone, fake, [a, b, a])
    assert (stop, client.requests, len(fake.requests)) == ("done", 2, 2)
    assert [r["address"] for r in store.records()] == [str(a), str(b)]  # first position kept


class _NoUnionSet(set):
    """A set that fails the test if ``run`` builds ``scored | exhausted`` once per address."""

    def __or__(self, other: object) -> set:
        raise AssertionError("the pending filter rebuilds a union per address")


def test_the_pending_filter_builds_no_union_per_address(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    class Store(ResultStore):
        def history(self):
            scored, last, exhausted = super().history()
            return _NoUnionSet(scored), last, _NoUnionSet(exhausted)

    fake = FakeShazam([json_response(200, NO_MATCH)] * 2)
    server.append(fake)
    _, client, stop = _go(tmp_path, tone, fake, _addresses(2), Store(tmp_path / "shazam.jsonl"))
    assert (stop, client.requests) == ("done", 2)


def test_an_address_whose_last_answer_was_a_stop_status_goes_after_the_other_retries(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    denied, flaky, slow = _addresses(3)  # grid order puts the denied address first
    who = recognizer_identity(12)
    store = ResultStore(tmp_path / "shazam.jsonl")
    store.append(str(flaky), who, ShazamOutcome(503, "server_error"))
    store.append(str(denied), who, ShazamOutcome(403, "server_error"))  # the day's first request
    store.append(str(slow), who, ShazamOutcome(503, "server_error"))
    fake = FakeShazam([json_response(200, NO_MATCH)] * 3)
    server.append(fake)
    store, _, stop = _go(tmp_path, tone, fake, [denied, flaky, slow], store)
    assert stop == "done"
    assert [r["address"] for r in store.records()][3:] == [str(a) for a in (flaky, slow, denied)]


def test_stop_status_addresses_go_least_recently_tried_first(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    # An address that is denied every day must not lead the stop-status tier every day.
    always, once = _addresses(2)  # grid order puts the always-denied address first
    who = recognizer_identity(12)
    store = ResultStore(tmp_path / "shazam.jsonl")
    store.append(str(always), who, ShazamOutcome(403, "server_error"))
    store.append(str(once), who, ShazamOutcome(403, "server_error"))
    store.append(str(always), who, ShazamOutcome(403, "server_error"))  # denied again, later
    fake = FakeShazam([json_response(200, NO_MATCH)] * 2)
    server.append(fake)
    store, _, stop = _go(tmp_path, tone, fake, [always, once], store)
    assert stop == "done"
    assert [r["address"] for r in store.records()][3:] == [str(once), str(always)]


def test_a_stop_status_that_was_later_answered_otherwise_no_longer_trails(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    first, second = _addresses(2)
    who = recognizer_identity(12)
    store = ResultStore(tmp_path / "shazam.jsonl")
    store.append(str(first), who, ShazamOutcome(403, "server_error"))
    store.append(str(first), who, ShazamOutcome(503, "server_error"))  # the last word
    store.append(str(second), who, ShazamOutcome(503, "server_error"))
    fake = FakeShazam([json_response(200, NO_MATCH)] * 2)
    server.append(fake)
    store, _, _ = _go(tmp_path, tone, fake, [first, second], store)
    assert [r["address"] for r in store.records()][3:] == [str(first), str(second)]


def test_a_refused_state_logs_the_summary_and_exits_non_zero_with_one_message(
    tmp_path: Path,
    tone: Path,
    server: list[FakeShazam],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeShazam([])
    server.append(fake)
    addresses = _addresses(2)
    monkeypatch.setattr(shazam_eval, "hour_addresses", lambda *a: addresses)

    @contextmanager
    def cut(address: ClipAddress, hour_path: Path, work_dir: Path) -> Iterator[Path]:
        yield tone

    monkeypatch.setattr(shazam_eval, "cut", cut)
    who = recognizer_identity(12)
    store, state = tmp_path / "shazam.jsonl", tmp_path / "throttle.json"
    results = ResultStore(store)
    for _ in range(1 + MAX_RETRIES):  # one address is already out of retries
        results.append(str(addresses[1]), who, ShazamOutcome(503, "server_error"))
    now = datetime.now(timezone.utc)
    state.write_text(
        json.dumps(
            {
                "day": now.date().isoformat(),
                "count": 1,
                "last": now.timestamp() + 3600,
                "stopped": False,
            }
        )
    )
    hours = tmp_path / "hours.txt"
    hours.write_text(HOUR + "\n")
    argv = [
        *("--hours", str(hours), "--archive-dir", str(tmp_path), "--base-url", fake.url),
        *("--work-dir", str(tmp_path / "work"), "--store", str(store), "--state", str(state)),
    ]
    with caplog.at_level("INFO"):
        assert shazam_eval.main(argv) == 1
    assert fake.requests == []
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "stopped: refused after 0 requests" in text
    assert "out of retries and not queried" in text and str(addresses[1]) in text
    refusals = [r.getMessage() for r in caplog.records if "future" in r.getMessage()]
    assert len(refusals) == 1 and "\n" not in refusals[0] and str(state) in refusals[0]
    # What to do: wait or fix the clock; deleting the state file is not the fix.
    assert "wait" in refusals[0] and "clock" in refusals[0]
    assert "deleting" in refusals[0] and "resets" in refusals[0]
    assert json.loads(state.read_text())["count"] == 1


def test_the_stored_identity_is_frozen_byte_for_byte() -> None:
    # A live leg filed hundreds of answers under this string; changing it re-queries every address.
    assert recognizer_identity(12) == "shazam@0.8.1, segment=12"
    assert recognizer_identity(6) == "shazam@0.8.1, segment=6"


def test_the_pinned_version_is_the_locked_and_the_installed_one() -> None:
    lock = (CHECKOUT / "uv.lock").read_text()  # no tomllib: the floor is 3.10
    locked = re.search(r'\[\[package\]\]\nname = "shazamio"\nversion = "([^"]+)"', lock)
    assert locked is not None
    assert SHAZAMIO_VERSION == locked.group(1)
    assert SHAZAMIO_VERSION == shazam_eval.version("shazamio")
    require_pinned_shazamio()  # the installed version agrees, so it returns


def test_the_identity_is_built_from_the_constant_not_the_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shazam_eval, "version", lambda name: "9.9.9")
    assert recognizer_identity(12) == "shazam@0.8.1, segment=12"


def test_another_installed_version_is_refused_with_what_it_would_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shazam_eval, "version", lambda name: "9.9.9")
    with pytest.raises(SystemExit) as refusal:
        require_pinned_shazamio()
    message = str(refusal.value)
    assert "\n" not in message
    for part in ("9.9.9", SHAZAMIO_VERSION, "new identity", "re-queries every address", "go-ahead"):
        assert part in message, part


@pytest.mark.parametrize("entry", [shazam_eval.main, run_mod.main], ids=["shazam_eval", "run"])
def test_a_cli_on_another_shazamio_version_refuses_before_the_lock_or_any_request(
    entry: Callable[[list[str]], int],
    tmp_path: Path,
    server: list[FakeShazam],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeShazam([])
    server.append(fake)
    monkeypatch.setattr(shazam_eval, "version", lambda name: "9.9.9")
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(tmp_path / "data"))
    state = tmp_path / "throttle.json"
    hours = tmp_path / "hours.txt"
    hours.write_text(HOUR + "\n")
    argv = ["--state", str(state), "--base-url", fake.url]
    if entry is shazam_eval.main:
        argv += ["--hours", str(hours), "--archive-dir", str(tmp_path)]
    with pytest.raises(SystemExit) as refusal:
        entry(argv)
    assert refusal.value.code not in (0, None)
    assert "9.9.9" in str(refusal.value)  # the version guard refused, not a later check
    assert fake.requests == []
    assert list(tmp_path.glob("throttle.json*")) == []  # no state file and no <state>.lock
    assert not (tmp_path / "data").exists()


def test_external_api_is_declared_excluded_and_opted_out_of_ci_sync() -> None:
    text = (CHECKOUT / "pyproject.toml").read_text()  # no tomllib: the floor is 3.10
    assert '"external_api: ' in text
    assert "not external_api" in text
    assert (
        "# ci-sync-skip: external_api reason: unofficial Shazam client, never called from CI"
        in text
    )
