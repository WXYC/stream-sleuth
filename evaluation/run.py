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
import json
import logging
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from collections.abc import Set as AbstractSet
from contextlib import AbstractContextManager, ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

from evaluation.clips import CAPTURE_LENGTHS_S, ClipAddress, ClipError, cut, hour_addresses
from evaluation.olaf_snapshot import RESULTS, checked_snapshot_dir, snapshot_lock
from evaluation.pool import open_pool_db, tag_lookup
from evaluation.shazam_eval import (
    SCORING_KINDS,
    CountingClient,
    FutureStateError,
    Key,
    ResultStore,
    ShazamOutcome,
    Throttle,
    budget_from_env,
    recognizer_identity,
    require_pinned_shazamio,
)
from evaluation.shazam_eval import run as run_shazam
from stream_sleuth.paths import data_dir, require_outside_checkout
from stream_sleuth.recognizers.base import EvalIdentification, IdentificationSource
from stream_sleuth.recognizers.olaf import DEFAULT_MIN_MATCH_COUNT, OlafRecognizer
from stream_sleuth.recognizers.olaf import recognizer_identity as olaf_identity

log = logging.getLogger(__name__)

MIN_FREE_BYTES = 5 << 30
# The recognizer identity's prefix -> the emission's source.
SOURCES: dict[str, IdentificationSource] = {"shazam": "shazam", "olaf": "local"}

OpenClip = Callable[[ClipAddress], AbstractContextManager[Path]]
Recognize = Callable[[str], EvalIdentification | None]


class Emission(NamedTuple):
    """One matched record: its store key (address and recognizer identity, so legs at one
    offset stay distinct), the parsed address (its ``hour_key`` is plays.jsonl's), and the answer."""

    key: Key
    address: ClipAddress
    found: EvalIdentification


class Results(NamedTuple):
    """What :func:`read_results` found: the matched answers, the keys with a scoring record
    (``matched`` or ``no_match``), and the tried-but-unscored keys with their latest kind."""

    emissions: list[Emission]
    scored: set[Key]
    uncovered: dict[Key, str]


class _Snapshot(ResultStore):
    """A store's records read once, so ``history()`` and the emissions share one snapshot."""

    def __init__(self, store: ResultStore, records: list[dict[str, Any]]) -> None:
        self.path, self._records = store.path, records

    def records(self) -> list[dict[str, Any]]:
        return self._records


@dataclass(frozen=True)
class Leg:
    """One capture length and codec profile over the ``all`` or ``subset`` hours, per recognizer."""

    name: str
    length_s: int
    profile: str
    hours: str
    recognizers: tuple[str, ...]


# The study's query budget. Shazam runs 12 s on the full corpus, then 6 s, 20 s, and 12 s @320k
# on the four-hour subset, in that order. Olaf is free, so it runs every length on the full
# corpus and 12 s @320k on the subset (plan 5.2). The two 12 s Shazam legs file results under
# one recognizer identity; only the address's @profile tells them apart, so a reader keys on it.
LEGS = (
    Leg("12s", 12, "128k", "all", ("shazam", "olaf")),
    Leg("6s", 6, "128k", "all", ("olaf",)),
    Leg("20s", 20, "128k", "all", ("olaf",)),
    Leg("6s-subset", 6, "128k", "subset", ("shazam",)),
    Leg("20s-subset", 20, "128k", "subset", ("shazam",)),
    Leg("12s-320k-subset", 12, "320k", "subset", ("shazam", "olaf")),
)


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
    """A matched record as an :class:`Emission`; None for every other kind.

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
    return Emission((record["address"], record["recognizer"]), address, found)


def study_identities(snapshot: str, min_match_count: int = DEFAULT_MIN_MATCH_COUNT) -> set[str]:
    """Every recognizer identity one study scores: each Shazam segment length and the Olaf snapshot.

    The Shazam identities name the pinned shazamio version (``SHAZAMIO_VERSION``), so they match
    the stored records; :func:`read_results` still warns about records under any other identity.
    """
    return {
        *(recognizer_identity(n) for n in CAPTURE_LENGTHS_S),
        olaf_identity(snapshot, min_match_count),
    }


def _line_of(path: Path, address: str) -> int:
    """The first line of ``path`` whose record has this ``address`` (0 if none), for an error."""
    for n, line in enumerate(path.read_bytes().split(b"\n"), 1):
        try:
            if json.loads(line.decode("utf-8")).get("address") == address:
                return n
        except (ValueError, AttributeError):
            continue
    return 0


def read_results(stores: Iterable[ResultStore], identities: AbstractSet[str]) -> Results:
    """The matched answers, scored keys, and uncovered keys in the union of ``stores``.

    Only records under ``identities`` are read. Per key the first ``matched`` or ``no_match``
    record is the answer, in store order; a later scoring record for it is logged and ignored.
    Scored keys include ``no_match`` ones, so a miss and an address never tried differ. A key is
    uncovered when it has records but none scoring (a 429, a 5xx, a decode error), and its value
    is its latest kind: missing data, never a miss. Which addresses were never tried needs the
    expected grid, which is the scorer's. Each store is read once, so a live run appending
    meanwhile cannot make its tallies and its emissions disagree.

    A record filed under another identity, such as another match floor or shazamio version, is
    not read, and a warning counts those per identity, so it never reads as zero coverage. A
    record whose address does not parse raises ``ValueError`` naming the store and line.

    ``identities`` is a set, never a bare ``str``: ``in`` on a string is a substring test, and
    Olaf identities nest (``min=1`` is inside ``min=12``), so a string raises ``TypeError``.
    """
    if isinstance(identities, str):
        raise TypeError(f"identities must be a set of identities, not the string {identities!r}")
    emissions: list[Emission] = []
    scored: set[Key] = set()
    tried: dict[Key, str] = {}  # key -> latest kind
    answered: set[Key] = set()
    outside: Counter[str] = Counter()
    for store in stores:
        records = store.records()
        store_scored, _, _ = _Snapshot(store, records).history()
        scored |= {k for k in store_scored if k[1] in identities}
        for r in records:
            key = (r["address"], r["recognizer"])
            if key[1] not in identities:
                outside[key[1]] += 1
                continue
            tried[key] = r["kind"]
            try:
                ClipAddress.parse(key[0])
            except ValueError as e:
                raise ValueError(f"{store.path}:{_line_of(store.path, key[0])}: {e}") from e
            if r["kind"] not in SCORING_KINDS:
                continue
            if key in answered:
                log.warning("ignored a later scoring record for %s under %s", *key)
            else:
                answered.add(key)
                emission = to_identification(r)
                if emission:
                    emissions.append(emission)
    uncovered = {k: kind for k, kind in tried.items() if k not in scored}
    if outside:
        log.warning(
            "%d record(s) under identities outside the requested set were not read: %s",
            sum(outside.values()),
            dict(outside),
        )
    return Results(emissions, scored, uncovered)


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
    require_pinned_shazamio()
    budget = budget_from_env()  # refused here, before any path is created
    use = {args.only} if args.only else set(SOURCES)
    if "olaf" in use and not args.snapshot:
        parser.error("--snapshot is required for Olaf legs; pass --only shazam to skip them")
    data = data_dir()
    work_dir = require_outside_checkout(args.work_dir or data / "clips")
    archive_dir = require_outside_checkout(args.archive_dir or data / "archive")
    store = ResultStore(args.store or data / "shazam" / "results.jsonl")
    state = require_outside_checkout(args.state or data / "shazam" / "throttle.json")
    hours = read_hours(args.selection or data / "selection.json")
    for directory in (work_dir, store.path.parent, state.parent):
        directory.mkdir(parents=True, exist_ok=True)
    preflight(work_dir)
    with ExitStack() as stack:  # held until every leg has run
        shazam = olaf = client = None
        if "shazam" in use:
            throttle = stack.enter_context(Throttle(state, *budget))
            client = CountingClient(throttle, base_url=args.base_url)
            shazam = (store, client)
        if "olaf" in use:
            home = checked_snapshot_dir(args.snapshot)
            if not (home / "pool.db").is_file():
                raise SystemExit(f"{home}: no snapshot here; build it first")
            stack.enter_context(snapshot_lock(home))
            db = stack.enter_context(closing(open_pool_db(home / "pool.db")))
            recognizer = OlafRecognizer(
                home, min_match_count=args.min_match_count, lookup=tag_lookup(db)
            )
            identity = olaf_identity(args.snapshot, args.min_match_count)
            olaf = (ResultStore(home / RESULTS), recognizer.recognize, identity)
        legs = [leg for leg in LEGS if not args.legs or leg.name in args.legs]
        report = run_legs(legs, hours, archive_dir, work_dir, shazam=shazam, olaf=olaf)
    for name, reason in report.items():
        log.info("%s: %s", name, reason)
    if client:
        log.info("%d requests", client.requests)
        if exhausted := sorted(store.history()[2]):
            log.warning("out of retries and not queried: %s", exhausted)
    return int("refused" in report.values())


if __name__ == "__main__":
    sys.exit(main())
