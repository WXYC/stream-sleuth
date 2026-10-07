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
import threading
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from evaluation.clips import ClipAddress, ClipError
from evaluation.shazam_eval import (
    CountingClient,
    ResultStore,
    ShazamOutcome,
    Throttle,
    main,
    outcome_from,
    recognizer_identity,
    run,
)
from stream_sleuth.paths import CHECKOUT, DataPathError
from tests.characterization.shazam_responses import JUANA_MOLINA, NO_MATCH

HOUR = "2026/08/12/202608121600.mp3"


class FakeShazam:
    """A localhost server that answers each POST with the next scripted (status, body, type)."""

    def __init__(self, responses: list[tuple[int, bytes, str]]) -> None:
        self.responses = list(responses)
        self.requests: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append(self.path)
                status, body, ctype = outer.responses.pop(0)
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
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
    throttle = Throttle(tmp_path / "throttle.json", rate, interval, clock=clock, sleep=clock.sleep)
    client = CountingClient(throttle, base_url=fake.url)
    store = ResultStore(tmp_path / "shazam.jsonl")

    @contextmanager
    def open_clip(address: ClipAddress) -> Iterator[Path]:
        yield tone

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
        (_json(503, {"error": "busy"}), "server_error"),
        ((200, b"<html>maintenance</html>", "text/html"), "decode_error"),
    ],
)
def test_each_response_is_one_request_and_one_stored_outcome(
    tmp_path: Path, tone: Path, server: list[FakeShazam], response: tuple, kind: str
) -> None:
    fake = FakeShazam([response])
    server.append(fake)
    store, _, client, _ = _run(tmp_path, tone, fake, 1)
    assert len(fake.requests) == 1 == client.requests
    [record] = store.records()
    assert record["kind"] == kind
    assert record["address"] == str(_addresses(1)[0])
    assert record["recognizer"] == recognizer_identity(12)


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


@pytest.mark.parametrize("flag", ["--store", "--state", "--work-dir"])
@pytest.mark.parametrize("where", ["inside", "relative"])
def test_the_cli_refuses_a_path_in_the_checkout_before_any_request(
    tmp_path: Path, server: list[FakeShazam], flag: str, where: str
) -> None:
    fake = FakeShazam([_json(200, NO_MATCH)])
    server.append(fake)
    bad = CHECKOUT / "shazam-guard-test" / "x" if where == "inside" else Path("x")
    args = {
        "--store": str(tmp_path / "shazam.jsonl"),
        "--state": str(tmp_path / "throttle.json"),
        "--work-dir": str(tmp_path / "work"),
    } | {flag: str(bad)}
    (tmp_path / "hours.txt").write_text(HOUR + "\n")
    argv = ["--hours", str(tmp_path / "hours.txt"), "--archive-dir", str(tmp_path)]
    argv += ["--base-url", fake.url]
    argv += [part for pair in args.items() for part in pair]
    with pytest.raises(DataPathError):
        main(argv)
    assert fake.requests == []
    assert not (CHECKOUT / "shazam-guard-test").exists()
    assert not (tmp_path / "work").exists()  # nothing was created for the other paths either


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


def test_the_client_requires_an_explicit_base_url(tmp_path: Path) -> None:
    throttle = Throttle(tmp_path / "t.json", 500, 20.0)
    with pytest.raises(TypeError):
        CountingClient(throttle)  # type: ignore[call-arg]


def test_external_api_is_declared_excluded_and_opted_out_of_ci_sync() -> None:
    text = (CHECKOUT / "pyproject.toml").read_text()  # no tomllib: the floor is 3.10
    assert '"external_api: ' in text
    assert "not external_api" in text
    assert (
        "# ci-sync-skip: external_api reason: unofficial Shazam client, never called from CI"
        in text
    )
