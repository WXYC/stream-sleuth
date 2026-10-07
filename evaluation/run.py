"""Run every leg of the study, one after another, under one daily Shazam budget.

Station-neutral. A leg is one capture length and codec profile over one hour set; every
leg's hours come from ``selection.json`` (written by ``corpus select``), never from this
module. Each Shazam leg goes through :func:`evaluation.shazam_eval.run` inside one
``Throttle``, one ``CountingClient``, and one state file, so all legs share one daily
budget; a leg that stops the day (anything but ``done``) starts no later Shazam leg, and
the report says so. Resume, retries, and stop reasons are ``shazam_eval``'s; this module
adds none of its own.

Every path a run writes is checked with ``require_outside_checkout`` and defaults from
``data_dir()``; the run refuses to start with less than 5 GiB free.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from collections import Counter
from collections.abc import Collection, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from evaluation.clips import CAPTURE_LENGTHS_S, ClipAddress, cut, hour_addresses
from evaluation.shazam_eval import (
    CountingClient,
    FutureStateError,
    Key,
    ResultStore,
    Throttle,
    budget_from_env,
    require_pinned_shazamio,
    recognizer_identity,
)
from evaluation.shazam_eval import run as run_shazam
from stream_sleuth.paths import data_dir, require_outside_checkout
from stream_sleuth.recognizers.base import EvalIdentification, IdentificationSource
from stream_sleuth.recognizers.olaf import DEFAULT_MIN_MATCH_COUNT
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity

log = logging.getLogger(__name__)

MIN_FREE_BYTES = 5 << 30
# The recognizer identity's prefix -> the emission's source.
SOURCES: dict[str, IdentificationSource] = {"shazam": "shazam", "olaf": "local"}

Emission = tuple[str, EvalIdentification]  # the hour key, as in plays.jsonl, and the emission


@dataclass(frozen=True)
class Leg:
    """One capture length and codec profile over the ``all`` or ``subset`` hours."""

    name: str
    length_s: int
    profile: str
    hours: str


# The study's Shazam query budget, in the order the legs run: 12 s on the full corpus, then
# 6 s, 20 s, and 12 s @320k on the four-hour subset. The two 12 s legs file results under one
# recognizer identity; only the address's @profile tells them apart, so a reader keys on it.
LEGS = (
    Leg("12s", 12, "128k", "all"),
    Leg("6s-subset", 6, "128k", "subset"),
    Leg("20s-subset", 20, "128k", "subset"),
    Leg("12s-320k-subset", 12, "320k", "subset"),
)


def run_legs(
    legs: Iterable[Leg],
    hours: dict[str, list[str]],
    archive_dir: Path,
    work_dir: Path,
    store: ResultStore,
    client: CountingClient,
) -> dict[str, str]:
    """Run each leg in turn; the report maps its name to ``done`` or the reason it stopped.

    ``refused`` means the throttle refused a future-dated state (``FutureStateError``,
    logged). A leg after one that stopped is ``not started``: it decodes and sends nothing.
    """

    def open_clip(a: ClipAddress) -> AbstractContextManager[Path]:
        return cut(a, archive_dir / a.hour_key, work_dir)

    report: dict[str, str] = {}
    stopped = False  # one leg that ends the day ends it for the rest
    for leg in legs:
        if stopped:
            report[leg.name] = "not started"
            continue
        addresses = hour_addresses(hours[leg.hours], archive_dir, leg.length_s, leg.profile)
        try:
            report[leg.name] = asyncio.run(run_shazam(addresses, open_clip, store, client))
        except FutureStateError as refusal:
            log.error("%s", refusal)
            report[leg.name] = "refused"
        stopped = report[leg.name] != "done"
    return report


def read_hours(selection: Path) -> dict[str, list[str]]:
    """The hour keys of ``selection.json``, read once: ``all`` of them, and the ``subset``.

    The file must be a JSON object whose ``hours`` is a non-empty object keyed by hour (so
    every key is a string), each hour's ``subset`` a JSON boolean, and at least one hour in
    the subset, or the subset legs would report ``done`` having sent nothing. Anything else,
    such as a hand-edited ``"false"``, which would be truthy, or ``hours.txt`` passed by
    mistake, refuses the run with one line naming the file (and the hour, when one is at fault).
    """

    def fail(problem: str) -> NoReturn:
        raise SystemExit(f"{selection}: {problem}")

    try:
        record = json.loads(selection.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        fail(f"cannot read: {e}")
    except json.JSONDecodeError as e:
        fail(f"not JSON ({e}); pass the selection.json that `select` wrote")
    if not isinstance(record, dict):
        fail("not an object")
    hours = record.get("hours")
    if not isinstance(hours, dict):
        fail("hours must be an object keyed by hour")
    if not hours:
        fail("no hours")
    for key, label in hours.items():
        subset = label.get("subset") if isinstance(label, dict) else None
        if not isinstance(subset, bool):
            fail(f"hour {key}: subset must be true or false")
    if not (chosen := [key for key, label in hours.items() if label["subset"]]):
        fail("no hour with subset: true")
    return {"all": list(hours), "subset": chosen}


def to_identification(record: dict[str, Any]) -> Emission | None:
    """A matched record as ``(hour key, EvalIdentification)``; None for every other kind.

    ``at`` is the address's grid offset. A Shazam answer sets ``query_offset_s = 0`` and
    ``ref_start_s`` from its stored offset, or neither when it stored none (the two are set
    together, and that answer is left out of lag estimation); an Olaf answer carries its own.
    """
    if record["kind"] != "matched":
        return None
    address = ClipAddress.parse(record["address"])
    source = SOURCES[record["recognizer"].partition("@")[0]]
    found: EvalIdentification = {
        "artist": record["artist"],
        "song": record["song"],
        "album": record["album"],
        "label": record["label"],
        "at": float(address.offset_s),
        "source": source,
    }
    if source == "local":
        found["confidence"] = record["confidence"]
        found["query_offset_s"] = record["query_offset_s"]
        found["ref_start_s"] = record["ref_start_s"]
        found["ref_key"] = record["ref_key"]
    elif record["offset_s"] is not None:
        found["query_offset_s"] = 0.0
        found["ref_start_s"] = record["offset_s"]
    return address.hour_key, found


def study_identities(snapshot: str, min_match_count: int = DEFAULT_MIN_MATCH_COUNT) -> set[str]:
    """Every recognizer identity one study scores: each Shazam segment length and the Olaf snapshot.

    The Shazam identities name the *installed* shazamio version, so a lock change makes them
    stop matching the records already stored; :func:`read_results` warns when that happens.
    """
    return {
        *(recognizer_identity(n) for n in CAPTURE_LENGTHS_S),
        olaf_identity(snapshot, min_match_count),
    }


def read_results(
    stores: Iterable[ResultStore], identities: Collection[str]
) -> tuple[list[Emission], dict[Key, str]]:
    """The emissions and the uncovered keys in the union of ``stores``, for ``identities`` only.

    A key is uncovered when it has records but none scoring (a 429, a 5xx, a decode error); the
    value is its latest kind. The scorer reports it as missing data, never as a miss. Which
    addresses were never tried at all needs the expected grid, which is the scorer's.

    A record filed under another identity, such as another match floor, is not returned, and
    a warning counts those per identity, so a version mismatch never reads as zero coverage.
    """
    emissions: list[Emission] = []
    uncovered: dict[Key, str] = {}
    outside: Counter[str] = Counter()
    for store in stores:
        records = store.records()  # one read, so the kinds and the emissions agree
        scored, last, _ = store.history()
        kinds = {(r["address"], r["recognizer"]): r["kind"] for r in records}  # latest wins
        uncovered |= {
            k: kinds[k] for k in last.keys() - scored if k[1] in identities and k in kinds
        }
        outside.update(r["recognizer"] for r in records if r["recognizer"] not in identities)
        emissions += [
            e for r in records if r["recognizer"] in identities and (e := to_identification(r))
        ]
    if outside:
        log.warning(
            "%d record(s) under identities outside the requested set were not read: %s",
            sum(outside.values()),
            dict(outside),
        )
    return emissions, uncovered


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
    parser.add_argument(
        "--legs", nargs="+", choices=[leg.name for leg in LEGS], help="default: all"
    )
    parser.add_argument("--base-url", default=None, help=argparse.SUPPRESS)  # tests only
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    require_pinned_shazamio()
    data = data_dir()
    work_dir = require_outside_checkout(args.work_dir or data / "clips")
    archive_dir = require_outside_checkout(args.archive_dir or data / "archive")
    store = ResultStore(args.store or data / "shazam" / "results.jsonl")
    state = require_outside_checkout(args.state or data / "shazam" / "throttle.json")
    hours = read_hours(args.selection or data / "selection.json")
    for directory in (work_dir, store.path.parent, state.parent):
        directory.mkdir(parents=True, exist_ok=True)
    preflight(work_dir)
    with Throttle(state, *budget_from_env()) as throttle:  # held until every leg has run
        client = CountingClient(throttle, base_url=args.base_url)
        legs = [leg for leg in LEGS if not args.legs or leg.name in args.legs]
        report = run_legs(legs, hours, archive_dir, work_dir, store, client)
    for name, reason in report.items():
        log.info("%s: %s", name, reason)
    log.info("%d requests", client.requests)
    if exhausted := sorted(store.history()[2]):
        log.warning("out of retries and not queried: %s", exhausted)
    return int("refused" in report.values())


if __name__ == "__main__":
    sys.exit(main())
