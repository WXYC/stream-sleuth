"""Shazam over the clip grid: one attempt per request, a persisted daily throttle, and an append-only store.

Station-neutral. ``shazamio``'s own client retries and hides status codes, so
``CountingClient`` implements its ``HTTPClientInterface`` instead: exactly one
``aiohttp`` attempt per request, every request counted against a ``Throttle``
before it is sent, and the status and parsed body kept for ``outcome_from``.
A 429 stops the run for the rest of the UTC day, and the stop is persisted, so a
restart that day sends nothing. Only ``matched`` and ``no_match`` are scoring
outcomes; ``server_error`` and ``decode_error`` are stored too and retried by a
later run. The store is a JSONL file the harness only ever appends to.

This module imports nothing from ``stream_sleuth``; ``outcome_from`` repeats
``recognizer.parse()``'s field extraction and adds Shazam's match offset.
"""

from __future__ import annotations

import argparse
import asyncio
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

from evaluation.clips import ClipAddress, cut, grid

log = logging.getLogger(__name__)

SCORING_KINDS = frozenset({"matched", "no_match"})


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
    means the request never got a response.
    """
    if status == 429:
        return ShazamOutcome(status, "rate_limited")
    if not 200 <= status < 300:
        return ShazamOutcome(status, "server_error")
    if body is None:
        return ShazamOutcome(status, "decode_error")
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


class Throttle:
    """At most ``rate_per_day`` requests per UTC day, ``min_interval_s`` apart, persisted to ``state_path``.

    The state file records the UTC day, the requests sent that day, the time of
    the last one, and whether a 429 stopped the day, so a restarted process
    honors all four.
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
        self.state_path = state_path
        self.rate_per_day = rate_per_day
        self.min_interval_s = min_interval_s
        self.clock = clock
        self.sleep = sleep

    def _state(self) -> dict[str, Any]:
        today = datetime.fromtimestamp(self.clock(), timezone.utc).date().isoformat()
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if state.get("day") != today:
            state = {"day": today, "count": 0, "last": state.get("last"), "stopped": False}
        return state

    def _save(self, state: dict[str, Any]) -> None:
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, self.state_path)

    async def acquire(self) -> None:
        """Wait out the interval and count one request, or raise ``DayStoppedError``."""
        state = self._state()
        if state["last"] is not None:
            wait = state["last"] + self.min_interval_s - self.clock()
            if wait > 0:
                await self.sleep(wait)
                state = self._state()
        if state["stopped"]:
            raise DayStoppedError("rate_limited")
        if state["count"] >= self.rate_per_day:
            raise DayStoppedError("daily_cap")
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
                try:
                    body = await resp.json(content_type=None)
                except ValueError:
                    body = None
                self.last = (resp.status, body if isinstance(body, dict) else None)
        return self.last[1] or {}


class ResultStore:
    """Append-only JSONL of Shazam outcomes keyed by clip address and recognizer identity."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

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
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run(
    addresses: Iterable[ClipAddress],
    open_clip: Callable[[ClipAddress], AbstractContextManager[Path]],
    store: ResultStore,
    client: CountingClient,
) -> str:
    """Query every unscored address in order; return ``done``, ``rate_limited``, or ``daily_cap``."""
    scored = store.scored()
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
        except (aiohttp.ClientError, asyncio.TimeoutError):
            outcome = ShazamOutcome(0, "server_error")
        store.append(str(address), identity, outcome)
        if outcome.kind == "rate_limited":
            client.throttle.stop_for_day()
            return "rate_limited"
    return "done"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=Path, required=True, help="file of hour keys, one per line")
    parser.add_argument("--archive-dir", type=Path, required=True, help="directory of hour files")
    parser.add_argument(
        "--work-dir", type=Path, required=True, help="where ephemeral clips are cut"
    )
    parser.add_argument("--store", type=Path, required=True, help="append-only results JSONL")
    parser.add_argument("--state", type=Path, required=True, help="throttle state JSON")
    parser.add_argument("--length", type=int, default=12, help="capture length in seconds")
    parser.add_argument("--profile", default="128k")
    parser.add_argument("--base-url", default=None, help=argparse.SUPPRESS)  # tests only
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    throttle = Throttle(
        args.state,
        int(os.environ.get("STREAM_SLEUTH_SHAZAM_RATE_PER_DAY", "500")),
        float(os.environ.get("STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S", "20")),
    )
    client = CountingClient(throttle, base_url=args.base_url)
    addresses = [
        a for key in args.hours.read_text().split() for a in grid(key, args.length, args.profile)
    ]
    stop = asyncio.run(
        run(
            addresses,
            lambda a: cut(a, args.archive_dir / a.hour_key, args.work_dir),
            ResultStore(args.store),
            client,
        )
    )
    log.info("stopped: %s after %d requests", stop, client.requests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
