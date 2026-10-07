"""Shazam evaluation adapter: outcomes, throttle, day-stop, and store, against a localhost server.

Every test points the client at a localhost ``http.server`` through the required
``base_url``; no test contacts Shazam. Signatures are computed offline by
``shazamio-core`` from a synthetic two-tone WAV, so ``Shazam.recognize`` runs its
real path up to the HTTP request.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import subprocess
import sys
import threading
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import aiohttp
import pytest

from evaluation.clips import ClipAddress, ClipError
from evaluation.shazam_eval import (
    MAX_FAILURE_STREAK,
    CountingClient,
    ResultStore,
    ShazamOutcome,
    Throttle,
    ThrottleBusyError,
    outcome_from,
    recognizer_identity,
    run,
)
from stream_sleuth.paths import CHECKOUT, DataPathError
from tests.characterization.shazam_responses import JUANA_MOLINA, NO_MATCH

HOUR = "2026/08/12/202608121600.mp3"


class FakeShazam:
    """A localhost server that answers each POST with the next scripted (status, body, type).

    A fourth element overrides the declared Content-Length, so a body can end short.
    """

    def __init__(self, responses: list[tuple]) -> None:
        self.responses = list(responses)
        self.requests: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append(self.path)
                status, body, ctype, *length = outer.responses.pop(0)
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(length[0] if length else len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _json(status: int, body: object) -> tuple[int, bytes, str]:
    return status, json.dumps(body).encode(), "application/json"


HTML_429 = (429, b"<html><body>Too Many Requests</body></html>", "text/html")


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
    rate = 16000
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack(
                    "<h",
                    int(
                        8000 * math.sin(2 * math.pi * 440 * i / rate)
                        + 3000 * math.sin(2 * math.pi * 1234 * i / rate)
                    ),
                )
                for i in range(rate * 12)
            )
        )
    return path


class FakeClock:
    """Wall clock plus a recording async sleep that advances it."""

    def __init__(self, start: float) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        await asyncio.sleep(0)  # yield, as a real sleep would, before time moves on
        self.now += seconds


def _clock_at(iso: str) -> FakeClock:
    return FakeClock(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp())


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
    clock = clock or _clock_at("2026-10-06T20:00:00")
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
        (_json(200, JUANA_MOLINA), "matched"),
        (_json(200, NO_MATCH), "no_match"),
        (_json(429, {"error": "rate limited"}), "rate_limited"),
        (HTML_429, "rate_limited"),
        ((429, b'{"error": "rate', "application/json", 100), "rate_limited"),  # body cut short
        (_json(503, {"error": "busy"}), "server_error"),
        ((200, b"<html>maintenance</html>", "text/html"), "decode_error"),
        (_json(200, {"track": "Juana Molina"}), "decode_error"),
    ],
)
def test_each_response_is_one_request_and_one_stored_outcome(
    tmp_path: Path, tone: Path, server: list[FakeShazam], response: tuple, kind: str
) -> None:
    fake = FakeShazam([response, _json(200, NO_MATCH)])
    server.append(fake)
    clock = _clock_at("2026-10-06T20:00:00")
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
    fake = FakeShazam([_json(200, JUANA_MOLINA)])
    server.append(fake)
    store, *_ = _run(tmp_path, tone, fake, 1)
    assert store.records()[0]["offset_s"] == 41.2


def test_requests_are_spaced_by_the_minimum_interval(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    fake = FakeShazam([_json(200, NO_MATCH)] * 3)
    server.append(fake)
    clock = _clock_at("2026-10-06T20:00:00")
    _run(tmp_path, tone, fake, 3, clock=clock, interval=20.0)
    assert clock.slept == [20.0, 20.0]
    assert len(fake.requests) == 3


@pytest.mark.parametrize("response", [_json(429, {"error": "rate limited"}), HTML_429])
def test_a_429_stops_the_day_and_survives_a_restart(
    tmp_path: Path, tone: Path, server: list[FakeShazam], response: tuple
) -> None:
    fake = FakeShazam([_json(200, NO_MATCH), response, _json(200, NO_MATCH)])
    server.append(fake)
    clock = _clock_at("2026-10-06T20:00:00")
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
    fake = FakeShazam([_json(200, NO_MATCH)] * 3)
    server.append(fake)
    clock = _clock_at("2026-10-06T20:00:00")
    _, _, _, stop = _run(tmp_path, tone, fake, 2, clock=clock, rate=3)
    assert stop == "done"
    store, _, client, stop = _run(tmp_path, tone, fake, 5, clock=clock, rate=3)
    assert (stop, client.requests, len(fake.requests)) == ("daily_cap", 1, 3)
    assert len(store.records()) == 3


def test_a_new_utc_day_resets_the_cap_and_the_stop(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    fake = FakeShazam([HTML_429, _json(200, NO_MATCH)])
    server.append(fake)
    clock = _clock_at("2026-10-06T23:59:00")
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
    fake = FakeShazam([_json(200, NO_MATCH)] * (first_n + 1))
    server.append(fake)
    _run(tmp_path, tone, fake, first_n, clock=_clock_at(first_at), rate=rate)
    restarted = _clock_at(restart_at)  # a new process: a new clock, nothing in memory
    _, _, client, stop = _run(tmp_path, tone, fake, first_n + 3, clock=restarted, rate=rate)
    assert restarted.slept == slept
    assert (stop, client.requests, len(fake.requests)) == ("daily_cap", 1, first_n + 1)


def test_the_interval_is_checked_again_after_every_sleep(tmp_path: Path) -> None:
    clock = _clock_at("2026-10-06T20:00:00")
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
    fake = FakeShazam([HTML_429, _json(200, NO_MATCH)])
    server.append(fake)
    _run(tmp_path, tone, fake, 1, clock=_clock_at("2026-10-07T00:10:00"))
    stepped_back = _clock_at("2026-10-06T23:58:00")
    _, _, client, stop = _run(tmp_path, tone, fake, 1, clock=stepped_back)
    assert (stop, client.requests, len(fake.requests)) == ("rate_limited", 0, 1)
    assert stepped_back.slept == []  # refused at once, not after sleeping into the later day


def test_resume_skips_scoring_outcomes_and_retries_errors(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    fake = FakeShazam(
        [_json(200, JUANA_MOLINA), _json(503, {}), _json(200, NO_MATCH), _json(200, NO_MATCH)]
    )
    server.append(fake)
    _run(tmp_path, tone, fake, 3)
    store, _, client, _ = _run(tmp_path, tone, fake, 3)
    assert client.requests == 1  # only the 503 address is retried
    kinds = [(r["address"], r["kind"]) for r in store.records()]
    second = str(_addresses(3)[1])
    assert kinds[1] == (second, "server_error") and kinds[3] == (second, "no_match")


def test_a_clip_error_skips_that_address_without_a_query_or_a_record(
    tmp_path: Path, tone: Path, server: list[FakeShazam], caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeShazam([_json(200, NO_MATCH)] * 2)
    server.append(fake)
    clock = _clock_at("2026-10-06T20:00:00")
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
    fake = FakeShazam([_json(503, {"error": "busy"})] * n)
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
    fake = FakeShazam([_json(503, {})] * MAX_FAILURE_STREAK + [_json(200, NO_MATCH)] * n)
    server.append(fake)
    _run(tmp_path, tone, fake, n)  # day one: the first 20 fail and the streak stops the day
    store, _, _, stop = _run(tmp_path, tone, fake, n, clock=_clock_at("2026-10-07T20:00:00"))
    day_two = [r["address"] for r in store.records()][MAX_FAILURE_STREAK:]
    expected = _addresses(n)[MAX_FAILURE_STREAK:] + _addresses(n)[:MAX_FAILURE_STREAK]
    assert (stop, day_two) == ("done", [str(a) for a in expected])


def test_a_scoring_outcome_resets_the_failure_streak(
    tmp_path: Path, tone: Path, server: list[FakeShazam]
) -> None:
    almost = [_json(503, {})] * (MAX_FAILURE_STREAK - 1)
    fake = FakeShazam([*almost, _json(200, NO_MATCH), *almost])
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
    clock = _clock_at("2026-10-06T20:00:00")
    with Throttle(tmp_path / "throttle.json", 500, 20.0, clock=clock, sleep=clock.sleep) as t:
        with pytest.raises(SimulatedCrashError):
            asyncio.run(CountingClient(t, base_url="http://127.0.0.1:9").request("POST", "/x"))
    clock.now += 5
    with Throttle(tmp_path / "throttle.json", 500, 20.0, clock=clock, sleep=clock.sleep) as t:
        asyncio.run(t.acquire())
    assert clock.slept == [15.0]  # the unsent request was counted and its time kept
    assert json.loads((tmp_path / "throttle.json").read_text())["count"] == 2


def test_external_api_is_declared_excluded_and_opted_out_of_ci_sync() -> None:
    text = (CHECKOUT / "pyproject.toml").read_text()  # no tomllib: the floor is 3.10
    assert '"external_api: ' in text
    assert "not external_api" in text
    assert (
        "# ci-sync-skip: external_api reason: unofficial Shazam client, never called from CI"
        in text
    )
