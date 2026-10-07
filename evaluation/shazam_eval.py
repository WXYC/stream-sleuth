"""Shazam over the clip grid: one attempt per request, a persisted daily throttle, and an append-only store.

Station-neutral. ``shazamio``'s own client retries and hides status codes, so
``CountingClient`` implements its ``HTTPClientInterface`` instead: exactly one
``aiohttp`` attempt per request, every request counted against a ``Throttle``
before it is sent, and the status and parsed body kept for ``outcome_from``.
A 429 stops the run for the rest of the UTC day, and the stop is persisted, so a
restart that day sends nothing. One process holds a throttle state file at a
time, enforced with a lock. Only ``matched`` and ``no_match`` are scoring
outcomes; ``server_error`` and ``decode_error`` are stored too and retried by a
later run. The store is a JSONL file the harness only ever appends to.

The result store, the throttle state, and the clip work directory are research
data: each is checked with ``stream_sleuth.paths.require_outside_checkout`` before
anything is created, and each defaults to a path under ``data_dir()``. This module
imports nothing else from ``stream_sleuth``; ``outcome_from`` repeats
``recognizer.parse()``'s field extraction and adds Shazam's match offset.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from shazamio import Shazam
from shazamio.interfaces.client import HTTPClientInterface

from evaluation.clips import ClipAddress, ClipError, cut, grid, hour_duration
from stream_sleuth.paths import data_dir, require_outside_checkout

log = logging.getLogger(__name__)

SCORING_KINDS = frozenset({"matched", "no_match"})
# Consecutive non-scoring outcomes that stop the day: a systemic failure other than a
# 429 (a 403, a changed response shape) would otherwise spend the whole daily budget.
MAX_FAILURE_STREAK = 20


def recognizer_identity(segment_s: int) -> str:
    """The store's recognizer key: the shazamio version and the fingerprinted length."""
    return f"shazam@{version('shazamio')}, segment={segment_s}"


@dataclass(frozen=True)
class ShazamOutcome:
    """One Shazam answer: HTTP status, kind, the four wire fields, and the match offset."""

    status: int
    kind: str
    artist: str = ""
    song: str = ""
    album: str = ""
    label: str = ""
    offset_s: float | None = None


def outcome_from(status: int, body: dict[str, Any] | None) -> ShazamOutcome:
    """Classify a response; ``body`` is None when it was not a JSON object.

    Any non-2xx status but 429 is a ``server_error`` (retried later); status 0
    means the request never got a response. A 2xx body that is not JSON, or is
    JSON of an unexpected shape, is a ``decode_error``.
    """
    if status == 429:
        return ShazamOutcome(status, "rate_limited")
    if not 200 <= status < 300:
        return ShazamOutcome(status, "server_error")
    if body is None:
        return ShazamOutcome(status, "decode_error")
    try:
        return _match(status, body)
    except (AttributeError, LookupError, TypeError, ValueError):
        return ShazamOutcome(status, "decode_error")


def _match(status: int, body: dict[str, Any]) -> ShazamOutcome:
    track = body.get("track")
    if not track:
        return ShazamOutcome(status, "no_match")
    fields = {"album": "", "label": ""}
    for section in track.get("sections", []) or []:
        for md in section.get("metadata", []) or []:
            key = (md.get("title") or "").strip().lower()
            if key in fields and not fields[key]:
                fields[key] = md.get("text", "") or ""
    matches = body.get("matches") or [{}]
    offset = matches[0].get("offset")
    return ShazamOutcome(
        status,
        "matched",
        artist=track.get("subtitle", "") or "",
        song=track.get("title", "") or "",
        offset_s=float(offset) if offset is not None else None,
        **fields,
    )


class DayStoppedError(Exception):
    """The throttle refuses any more requests today; ``args[0]`` says why."""


class ThrottleBusyError(Exception):
    """Another process holds the throttle state file's lock."""


class Throttle:
    """At most ``rate_per_day`` requests per UTC day, ``min_interval_s`` apart, persisted to ``state_path``.

    The state file records the UTC day, the requests sent that day, the time of
    the last one, and whether a 429 stopped the day, so a restarted process
    honors all four. ``state_path`` is refused if it is relative or inside the
    checkout (``DataPathError``); its directory must exist. One process per state
    file: opening takes a non-blocking exclusive ``flock`` on ``<state_path>.lock``
    and holds it until ``close()`` (or process exit), and a second opener gets
    ``ThrottleBusyError``. The budget is the account's, so every leg shares one
    file and runs in turn.
    """

    def __init__(
        self,
        state_path: Path,
        rate_per_day: int,
        min_interval_s: float,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.state_path = require_outside_checkout(state_path)
        self.rate_per_day = rate_per_day
        self.min_interval_s = min_interval_s
        self.clock = clock
        self.sleep = sleep
        lock_path = self.state_path.with_name(self.state_path.name + ".lock")
        self._lock: int | None = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._lock)
            raise ThrottleBusyError(
                f"{self.state_path}: another run holds it ({lock_path}); one process per state file"
            ) from None

    def close(self) -> None:
        """Release the lock; the throttle refuses to read or write state after this."""
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def __enter__(self) -> Throttle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _state(self) -> dict[str, Any]:
        if self._lock is None:
            raise ValueError(f"throttle on {self.state_path} is closed")
        today = datetime.fromtimestamp(self.clock(), timezone.utc).date().isoformat()
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if today > state.get("day", ""):  # a clock stepped back keeps the later day's state
            state = {"day": today, "count": 0, "last": state.get("last"), "stopped": False}
        return state

    def _save(self, state: dict[str, Any]) -> None:
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, self.state_path)

    async def acquire(self) -> None:
        """Wait out the interval and count one request, or raise ``DayStoppedError``.

        The stop, the cap, and the interval are checked again after every sleep,
        and the count and time are saved before the caller sends, so a crash
        after this returns over-counts rather than under-counts.
        """
        while True:
            state = self._state()
            if state["stopped"]:
                raise DayStoppedError("rate_limited")
            if state["count"] >= self.rate_per_day:
                raise DayStoppedError("daily_cap")
            last = state["last"]
            wait = 0.0 if last is None else last + self.min_interval_s - self.clock()
            if wait <= 0:
                break
            await self.sleep(wait)
        self._save({**state, "count": state["count"] + 1, "last": self.clock()})

    def stop_for_day(self) -> None:
        self._save({**self._state(), "stopped": True})


class CountingClient(HTTPClientInterface):
    """``shazamio`` HTTP client: one attempt per request, throttled, status kept in ``last``.

    ``base_url`` is required so no caller reaches Shazam by default: pass None
    for the real host, or a test server's URL, whose scheme and host replace the
    request URL's.
    """

    def __init__(self, throttle: Throttle, *, base_url: str | None) -> None:
        self.throttle = throttle
        self.base_url = base_url
        self.requests = 0
        self.last: tuple[int, dict[str, Any] | None] = (0, None)

    async def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        await self.throttle.acquire()
        if self.base_url is not None:
            base = urlsplit(self.base_url)
            url = urlunsplit(urlsplit(url)._replace(scheme=base.scheme, netloc=base.netloc))
        self.requests += 1
        self.last = (0, None)
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, url, **kwargs) as resp:
                self.last = (resp.status, None)  # kept if reading the body fails
                try:
                    body = await resp.json(content_type=None)
                except ValueError:
                    body = None
                self.last = (resp.status, body if isinstance(body, dict) else None)
        return self.last[1] or {}


class ResultStore:
    """Append-only JSONL of Shazam outcomes keyed by clip address and recognizer identity."""

    def __init__(self, path: Path) -> None:
        self.path = require_outside_checkout(path)

    def records(self) -> list[dict[str, Any]]:
        """Every whole record; a line torn by a crash is logged and skipped, never rewritten."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        # Decoded per line, so a crash inside a multibyte character tears only that line.
        for line in filter(None, self.path.read_bytes().split(b"\n")):
            try:
                records.append(json.loads(line.decode("utf-8")))
            except ValueError:  # includes UnicodeDecodeError
                log.warning("skipped a torn line in %s: %.60r", self.path, line)
        return records

    def _ends_mid_line(self) -> bool:
        try:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                return f.read(1) != b"\n"
        except OSError:  # missing or empty
            return False

    def scored(self) -> set[tuple[str, str]]:
        """Addresses with a scoring outcome; anything else is retried."""
        return {
            (r["address"], r["recognizer"]) for r in self.records() if r["kind"] in SCORING_KINDS
        }

    def append(self, address: str, recognizer: str, outcome: ShazamOutcome) -> None:
        record = {
            "address": address,
            "recognizer": recognizer,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **asdict(outcome),
        }
        lead = "\n" if self._ends_mid_line() else ""  # never glue onto a torn line
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(lead + json.dumps(record, ensure_ascii=False) + "\n")


async def run(
    addresses: Iterable[ClipAddress],
    open_clip: Callable[[ClipAddress], AbstractContextManager[Path]],
    store: ResultStore,
    client: CountingClient,
) -> str:
    """Query every unscored address in order.

    Returns ``done``, ``rate_limited``, ``daily_cap``, or ``failure_streak`` (after
    ``MAX_FAILURE_STREAK`` non-scoring outcomes in a row, which stops the day like a 429).
    """
    scored = store.scored()
    streak = 0
    for address in addresses:
        identity = recognizer_identity(address.length_s)
        if (str(address), identity) in scored:
            continue
        shazam = Shazam(http_client=client, segment_duration_seconds=address.length_s)
        try:
            with open_clip(address) as clip:
                await shazam.recognize(str(clip))
            outcome = outcome_from(*client.last)
        except DayStoppedError as stop:
            return str(stop.args[0])
        except ClipError as exc:  # no clip, so no query and nothing stored
            log.warning("skipped %s: %s", address, exc)
            continue
        except (aiohttp.ClientError, asyncio.TimeoutError):
            status = client.last[0]  # a 429 whose body failed to arrive still stops the day
            outcome = ShazamOutcome(status, "rate_limited" if status == 429 else "server_error")
        if outcome.kind == "rate_limited":
            client.throttle.stop_for_day()  # before the record, so a crash between keeps the stop
            store.append(str(address), identity, outcome)
            return "rate_limited"
        store.append(str(address), identity, outcome)
        streak = 0 if outcome.kind in SCORING_KINDS else streak + 1
        if streak >= MAX_FAILURE_STREAK:
            log.warning("%d non-scoring outcomes in a row; stopping for the day", streak)
            client.throttle.stop_for_day()
            return "failure_streak"
    return "done"


def hour_addresses(
    keys: Iterable[str], archive_dir: Path, length_s: int, profile: str
) -> list[ClipAddress]:
    """Every clip address that fits each hour's decoded length, in ``keys`` order.

    An hour that is missing or unreadable is logged and contributes nothing, so
    one bad hour never stops the run; a short hour yields only the clips that fit.
    """
    addresses: list[ClipAddress] = []
    for key in keys:
        try:
            hour_s = hour_duration(archive_dir / key)
        except ClipError as exc:
            log.warning("skipped hour %s: %s", key, exc)
            continue
        addresses += grid(key, length_s, profile, hour_s=hour_s)
    return addresses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=Path, required=True, help="file of hour keys, one per line")
    parser.add_argument("--archive-dir", type=Path, required=True, help="directory of hour files")
    parser.add_argument("--work-dir", type=Path, help="where clips are cut; default: <data>/clips")
    parser.add_argument(
        "--store", type=Path, help="append-only results JSONL; default: <data>/shazam/results.jsonl"
    )
    parser.add_argument(
        "--state", type=Path, help="throttle state JSON; default: <data>/shazam/throttle.json"
    )
    parser.add_argument("--length", type=int, default=12, help="capture length in seconds")
    parser.add_argument("--profile", default="128k")
    parser.add_argument("--base-url", default=None, help=argparse.SUPPRESS)  # tests only
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Every path this run writes is checked before a request, a mkdir, or a decode.
    work_dir = require_outside_checkout(args.work_dir or data_dir() / "clips")
    store = ResultStore(args.store or data_dir() / "shazam" / "results.jsonl")
    state_path = require_outside_checkout(args.state or data_dir() / "shazam" / "throttle.json")
    for directory in (work_dir, store.path.parent, state_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    # Held until the run ends, before any decode: a second run on this state is refused.
    with Throttle(
        state_path,
        int(os.environ.get("STREAM_SLEUTH_SHAZAM_RATE_PER_DAY", "500")),
        float(os.environ.get("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", "20")),
    ) as throttle:
        client = CountingClient(throttle, base_url=args.base_url)
        addresses = hour_addresses(
            args.hours.read_text().split(), args.archive_dir, args.length, args.profile
        )
        stop = asyncio.run(
            run(
                addresses,
                lambda a: cut(a, args.archive_dir / a.hour_key, work_dir),
                store,
                client,
            )
        )
    log.info("stopped: %s after %d requests", stop, client.requests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
