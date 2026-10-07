"""Run every leg of the study over both recognizers, one after another.

Station-neutral. A leg is one capture length and codec profile over one hour set, run by
the recognizers it names; every leg's hours come from ``selection.json`` (written by
``corpus select``), never from this module. Each Shazam leg goes through
:func:`evaluation.shazam_eval.run` inside one ``Throttle``, one ``CountingClient``, and one
state file, so all legs share one daily budget; a leg that stops the day (anything but
``done``) starts no later Shazam leg, and the report says so. Resume, retries, and stop
reasons are ``shazam_eval``'s; this module adds none of its own. The Olaf legs query a
snapshot and file their results beside it in ``results.jsonl``, under an identity that
carries the commit, snapshot, and match floor. Olaf is free and deterministic, so its
results are never retried, and a day that stops Shazam does not stop it.

Every path a run writes is checked with ``require_outside_checkout`` and defaults from
``data_dir()``; the run refuses to start with less than 5 GiB free.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, ExitStack, closing
from dataclasses import dataclass
from pathlib import Path

from evaluation.clips import ClipAddress, ClipError, cut, hour_addresses
from evaluation.olaf_snapshot import (
    RESULTS,
    SnapshotError,
    checked_snapshot_dir,
    require_built,
    snapshot_lock,
)
from evaluation.pool import open_pool_db, tag_lookup
from evaluation.results import LEGS, SOURCES, Leg, ResultStore, require_snapshot, select_legs
from evaluation.selection import read_hours
from evaluation.shazam_eval import (
    CountingClient,
    FutureStateError,
    ShazamOutcome,
    Throttle,
    budget_from_env,
    require_pinned_shazamio,
)
from evaluation.shazam_eval import run as run_shazam
from stream_sleuth.paths import data_dir, require_outside_checkout
from stream_sleuth.recognizers.base import EvalIdentification
from stream_sleuth.recognizers.olaf import DEFAULT_MIN_MATCH_COUNT, OlafRecognizer
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity

log = logging.getLogger(__name__)

MIN_FREE_BYTES = 5 << 30

OpenClip = Callable[[ClipAddress], AbstractContextManager[Path]]
Recognize = Callable[[str], EvalIdentification | None]


@dataclass(frozen=True)
class OlafOutcome(ShazamOutcome):
    """An Olaf answer in the Shazam store's record shape, plus what only Olaf reports.

    The inherited ``status`` is 0, meaning the subprocess answered, and ``offset_s`` is always
    null (Olaf reports ``query_offset_s`` and ``ref_start_s``); neither says anything about HTTP.
    """

    confidence: float | None = None
    ref_key: str = ""
    query_offset_s: float | None = None
    ref_start_s: float | None = None


def _olaf_outcome(found: EvalIdentification | None) -> OlafOutcome:
    if found is None:
        return OlafOutcome(0, "no_match")
    return OlafOutcome(
        0, "matched", found["artist"], found["song"], found["album"], found["label"],
        confidence=found["confidence"], ref_key=found["ref_key"],
        query_offset_s=found["query_offset_s"], ref_start_s=found["ref_start_s"],
    )  # fmt: skip


def run_olaf(
    addresses: Iterable[ClipAddress],
    open_clip: OpenClip,
    store: ResultStore,
    recognize: Recognize,
    identity: str,
) -> str:
    """Query every address with no record under ``identity``; a stored answer is never retried.

    An ``OlafError`` propagates: a broken index fails every query, so the run aborts, and
    the failing address has no record.
    """
    _, latest, _ = store.history()
    for address in addresses:
        if (str(address), identity) in latest:
            continue
        try:
            with open_clip(address) as clip:
                found = recognize(str(clip))
        except ClipError as exc:  # no clip, so no query and nothing stored
            log.warning("skipped %s: %s", address, exc)
            continue
        store.append(str(address), identity, _olaf_outcome(found))
    return "done"


def run_legs(
    legs: Iterable[Leg],
    hours: dict[str, list[str]],
    archive_dir: Path,
    work_dir: Path,
    *,
    shazam: tuple[ResultStore, CountingClient] | None = None,
    olaf: tuple[ResultStore, Recognize, str] | None = None,
) -> dict[str, str]:
    """Run each leg's recognizers in turn; the report maps ``<leg>/<recognizer>`` to how it ended.

    A Shazam leg ends ``done`` or with the reason it stopped, or ``refused`` when the throttle
    refused a future-dated state (``FutureStateError``, logged). A Shazam leg after one that
    stopped is ``not started``: it decodes and sends nothing. Olaf legs are not stopped by any
    of that, since they spend no budget: they run, in leg order, regardless.
    """

    def inputs(leg: Leg, wav: bool) -> tuple[list[ClipAddress], OpenClip]:
        addresses = hour_addresses(hours[leg.hours], archive_dir, leg.length_s, leg.profile)
        return addresses, lambda a: cut(a, archive_dir / a.hour_key, work_dir, wav=wav)

    report: dict[str, str] = {}
    stopped = False  # one Shazam leg that ends the day ends the day for the rest
    for leg in legs:
        if shazam and "shazam" in leg.recognizers:
            tag = f"{leg.name}/shazam"
            if stopped:
                report[tag] = "not started"
            else:
                addresses, open_clip = inputs(leg, wav=False)
                try:
                    report[tag] = asyncio.run(run_shazam(addresses, open_clip, *shazam))
                except FutureStateError as refusal:
                    log.error("%s", refusal)
                    report[tag] = "refused"
                stopped = report[tag] != "done"
        if olaf and "olaf" in leg.recognizers:
            addresses, open_clip = inputs(leg, wav=True)
            report[f"{leg.name}/olaf"] = run_olaf(addresses, open_clip, *olaf)
    return report


def preflight(path: Path) -> None:
    """Refuse to start with less than 5 GiB free where ``path`` lives."""
    free = shutil.disk_usage(path).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"{path}: {free / 2**30:.1f} GiB free; refusing to start below 5 GiB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, help="default: <data>/selection.json")
    parser.add_argument("--archive-dir", type=Path, help="default: <data>/archive")
    parser.add_argument("--work-dir", type=Path, help="where clips are cut; default: <data>/clips")
    parser.add_argument("--store", type=Path, help="default: <data>/shazam/results.jsonl")
    parser.add_argument("--state", type=Path, help="default: <data>/shazam/throttle.json")
    parser.add_argument("--snapshot", help="the Olaf snapshot to query; required for Olaf legs")
    parser.add_argument("--min-match-count", type=int, default=DEFAULT_MIN_MATCH_COUNT)
    parser.add_argument("--only", choices=sorted(SOURCES), help="run one recognizer's legs")
    parser.add_argument(
        "--legs", nargs="+", choices=[leg.name for leg in LEGS], help="default: all"
    )
    parser.add_argument("--base-url", default=None, help=argparse.SUPPRESS)  # tests only
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    legs, use_shazam, use_olaf = select_legs(parser, args.only, args.legs)
    # A run with no Shazam leg touches nothing of Shazam's: no pin check, budget, state, or lock.
    if use_shazam:
        require_pinned_shazamio()
        budget = budget_from_env()  # refused here, before any path is created
    require_snapshot(parser, use_olaf, args.snapshot)
    data = data_dir()
    work_dir = require_outside_checkout(args.work_dir or data / "clips")
    archive_dir = require_outside_checkout(args.archive_dir or data / "archive")
    directories = [work_dir]
    if use_shazam:
        store = ResultStore(args.store or data / "shazam" / "results.jsonl")
        state = require_outside_checkout(args.state or data / "shazam" / "throttle.json")
        directories += [store.path.parent, state.parent]
    hours = read_hours(args.selection or data / "selection.json")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    preflight(work_dir)
    report: dict[str, str] = {}
    with ExitStack() as stack:  # the snapshot's lock and pool.db, held through every leg
        olaf = None
        if use_olaf:  # refused before the Shazam lock is taken, in one line
            try:
                home = checked_snapshot_dir(args.snapshot)
                if not (home / "pool.db").is_file():
                    raise SystemExit(f"{home}: no snapshot here; build it first")
                stack.enter_context(snapshot_lock(home))
                require_built(home)  # an unfinished build must not have its misses stored
            except SnapshotError as refusal:
                raise SystemExit(str(refusal)) from None
            db = stack.enter_context(closing(open_pool_db(home / "pool.db")))
            recognizer = OlafRecognizer(
                home, min_match_count=args.min_match_count, lookup=tag_lookup(db)
            )
            identity = olaf_identity(args.snapshot, args.min_match_count)
            olaf = (ResultStore(home / RESULTS), recognizer.recognize, identity)
        if use_shazam:  # the throttle's lock is released as soon as the Shazam legs end
            with Throttle(state, *budget) as throttle:
                client = CountingClient(throttle, base_url=args.base_url)
                report = run_legs(legs, hours, archive_dir, work_dir, shazam=(store, client))
            # Logged before any Olaf leg, so an Olaf exception cannot drop the Shazam outcome.
            for name, reason in report.items():
                log.info("%s: %s", name, reason)
            log.info("%d requests", client.requests)
            if exhausted := sorted(store.history()[2]):
                log.warning("out of retries and not queried: %s", exhausted)
        if olaf:
            olaf_report = run_legs(legs, hours, archive_dir, work_dir, olaf=olaf)
            for name, reason in olaf_report.items():
                log.info("%s: %s", name, reason)
            report |= olaf_report
    return int("refused" in report.values())


if __name__ == "__main__":
    sys.exit(main())
