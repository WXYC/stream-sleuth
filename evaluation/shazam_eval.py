"""Shazam over the clip grid: one attempt per request, a persisted daily throttle, and an append-only store.

Station-neutral. ``shazamio``'s own client retries and hides status codes, so
``CountingClient`` implements its ``HTTPClientInterface`` instead: exactly one
``aiohttp`` attempt per request, every request counted against a ``Throttle``
before it is sent, and the status and parsed body kept for ``outcome_from``.
A 429 (``rate_limited``) or a 403 (``forbidden``) stops the run for the rest of the UTC
day, and the stop is persisted, so a restart that day sends nothing. A state file whose
last request is more than the interval in the future is refused, never slept on. One
process holds a throttle state file at a time, enforced with a lock. Only ``matched``
and ``no_match`` are scoring outcomes; ``server_error`` and ``decode_error`` are stored
too and retried by a later run, at most ``MAX_RETRIES`` (in ``evaluation.results``) times per address, after which
the address is reported and not queried. The store is a JSONL file the harness only
ever appends to.

The result store, the throttle state, and the clip work directory are research
data: each is checked with ``stream_sleuth.paths.require_outside_checkout`` before
anything is created, and each defaults to a path under ``data_dir()``. This module
imports only ``stream_sleuth.paths`` and ``stream_sleuth.recognizers.shazam.parse`` from
``stream_sleuth``: ``outcome_from`` takes the four wire fields from the live recognizer's
``parse`` (one extraction, so the study scores what WXDU posts) and adds Shazam's match offset.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from shazamio import Shazam
from shazamio.interfaces.client import HTTPClientInterface

from evaluation.clips import ClipAddress, ClipError, cut, hour_addresses
from evaluation.envvars import env_number
from evaluation.results import (
    SCORING_KINDS,
    SHAZAMIO_VERSION,
    STOP_REASONS,
    Key,
    ResultStore,
    recognizer_identity,
)
from stream_sleuth.paths import data_dir, require_outside_checkout
from stream_sleuth.recognizers.shazam import parse

log = logging.getLogger(__name__)

# Consecutive non-scoring outcomes that stop the day: a systemic failure other than a
# 429 or 403 (a changed response shape, a flapping 5xx) would otherwise spend the whole budget.
MAX_FAILURE_STREAK = 20


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def require_pinned_shazamio() -> None:
    """Exit before any lock, decode, or request unless the installed shazamio is the pinned one."""
    if (installed := version("shazamio")) != SHAZAMIO_VERSION:
        raise SystemExit(
            f"shazamio {installed} is installed but the identity is pinned to {SHAZAMIO_VERSION}: a new "
            "version is a new identity that re-queries every address; change SHAZAMIO_VERSION "
            "deliberately, with the owner's go-ahead"
        )


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
    fields = parse(body)  # the live recognizer's extraction, so the study scores what it posts
    if fields is None:
        return ShazamOutcome(status, "no_match")
    offset = (body.get("matches") or [{}])[0].get("offset")
    return ShazamOutcome(
        status, "matched", offset_s=float(offset) if offset is not None else None, **fields
    )


class DayStoppedError(Exception):
    """The throttle refuses any more requests today; ``args[0]`` says why."""


class FutureStateError(ValueError):
    """The throttle state's last request is further ahead of the clock than one interval."""


class ThrottleBusyError(Exception):
    """Another process holds the throttle state file's lock."""


class Throttle:
    """At most ``rate_per_day`` requests per UTC day, ``min_interval_s`` apart, persisted to ``state_path``.

    The state file records the UTC day, the requests sent that day, the time of
    the last one, and the reason a stop gave for the day (``stop_for_day``: a 429
    or 403 status, or the failure streak), so a restarted process honors all four.
    ``acquire()`` raises ``DayStoppedError`` once the day is stopped or at its cap,
    and ``FutureStateError`` for a last request more than one interval ahead of the
    clock, which it never sleeps on and which deleting the file would only hide.
    ``state_path`` is refused if it is relative or inside the
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
        today = _utc(self.clock())[:10]
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
            if state["stopped"]:  # the reason, or True from a state file written before reasons
                stopped = state["stopped"]
                raise DayStoppedError(stopped if isinstance(stopped, str) else "rate_limited")
            if state["count"] >= self.rate_per_day:
                raise DayStoppedError("daily_cap")
            last, now = state["last"], self.clock()
            wait = 0.0 if last is None else last + self.min_interval_s - now
            if wait > 2 * self.min_interval_s:  # last is more than an interval ahead of now
                raise FutureStateError(
                    f"{self.state_path}: last {_utc(last)} is in the future of now {_utc(now)}; "
                    "wait until the clock passes it, or correct the clock; deleting the state "
                    "file resets the day's count and clears any stop, so it is not the fix"
                )
            if wait <= 0:
                break
            if wait > self.min_interval_s:
                log.warning("%s: waiting %.0fs, longer than the interval", self.state_path, wait)
            await self.sleep(wait)
        self._save({**state, "count": state["count"] + 1, "last": self.clock()})

    def stop_for_day(self, reason: str = "rate_limited") -> None:
        """Refuse every request until the next UTC day; ``acquire()`` raises with ``reason``."""
        self._save({**self._state(), "stopped": reason})


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


async def run(
    addresses: Iterable[ClipAddress],
    open_clip: Callable[[ClipAddress], AbstractContextManager[Path]],
    store: ResultStore,
    client: CountingClient,
) -> str:
    """Query every unscored address once, in order, except those out of retries (not queried).

    Returns ``done``, ``rate_limited``, ``forbidden`` (a 403), ``daily_cap``, or
    ``failure_streak`` (after ``MAX_FAILURE_STREAK`` non-scoring outcomes in a row);
    all but ``done`` and ``daily_cap`` stop the day like a 429.
    """
    scored, last, exhausted = store.history()

    def key(address: ClipAddress) -> Key:
        return (str(address), recognizer_identity(address.length_s))

    by_key = {key(a): a for a in addresses}  # a repeated address is queried once, where first seen
    pending = [a for k, a in by_key.items() if k not in scored and k not in exhausted]
    # Never-tried addresses first, so a stretch that always fails is retried only after them
    # and cannot trip the failure streak at the same place every day; then retries, and last
    # those whose latest answer stopped the day, least recently tried first, so one address
    # cannot lead every day's queue.
    recency = {k: i for i, k in enumerate(last)}

    def rank(a: ClipAddress) -> tuple[bool, bool, int]:
        k = key(a)
        stopped = last.get(k) in STOP_REASONS
        return (k in last, stopped, recency[k] if stopped else 0)

    pending.sort(key=rank)  # stable: grid order within the never-tried and retry tiers
    streak = 0
    for address in pending:
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
        if reason := STOP_REASONS.get(outcome.status):
            client.throttle.stop_for_day(reason)  # before the record, so a crash between keeps it
            store.append(*key(address), outcome)
            return reason
        store.append(*key(address), outcome)
        streak = 0 if outcome.kind in SCORING_KINDS else streak + 1
        if streak >= MAX_FAILURE_STREAK:
            log.warning("%d non-scoring outcomes in a row; stopping for the day", streak)
            client.throttle.stop_for_day("failure_streak")
            return "failure_streak"
    return "done"


def budget_from_env() -> tuple[int, float]:
    """The account's daily request budget and request spacing: the one place both settings are read.

    The rate must be a positive integer and the interval a finite number of seconds, zero
    allowed (no spacing, the daily cap still binds); otherwise the run is refused by name.
    """
    return (
        env_number("STREAM_SLEUTH_SHAZAM_RATE_PER_DAY", "500", int, 1, "a positive integer"),
        env_number("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", "20", float, 0.0, "a finite number >= 0"),
    )


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
    require_pinned_shazamio()
    budget = budget_from_env()  # refused here, before any path is created
    # Every path this run writes is checked before a request, a mkdir, or a decode.
    work_dir = require_outside_checkout(args.work_dir or data_dir() / "clips")
    store = ResultStore(args.store or data_dir() / "shazam" / "results.jsonl")
    state_path = require_outside_checkout(args.state or data_dir() / "shazam" / "throttle.json")
    for directory in (work_dir, store.path.parent, state_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    # Held until the run ends, before any decode: a second run on this state is refused.
    with Throttle(state_path, *budget) as throttle:
        client = CountingClient(throttle, base_url=args.base_url)
        addresses = hour_addresses(
            args.hours.read_text().split(), args.archive_dir, args.length, args.profile
        )
        try:
            stop = asyncio.run(
                run(
                    addresses,
                    lambda a: cut(a, args.archive_dir / a.hour_key, work_dir),
                    store,
                    client,
                )
            )
        except FutureStateError as refusal:  # still log the summary below
            log.error("%s", refusal)
            stop = "refused"
    log.info("stopped: %s after %d requests", stop, client.requests)
    if exhausted := sorted(store.history()[2]):
        log.warning("out of retries and not queried: %s", exhausted)
    return int(stop == "refused")


if __name__ == "__main__":
    sys.exit(main())
