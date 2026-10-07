"""The review queues: wrong emissions a person judges, and the verdicts read back.

Station-neutral, like :mod:`evaluation.score`, which scores and calls into this module; it never
imports :mod:`evaluation.corpus` or :mod:`evaluation.archive` (the station import scan enforces
it). :func:`near_miss_runs` and :func:`false_positive_runs` choose runs of wrong emissions (an
emission no play claims) for the two CSV queues, :func:`write_queue` and :func:`settle_queue`
write them without ever overwriting a person's verdicts, and :func:`read_queue` reads those
verdicts back. The scoring types it works on (:class:`evaluation.score.Play`, ``Verdict``,
``LegScore``) are imported for annotations only, so this module has no import cycle with
:mod:`evaluation.score`, which imports it; the three emission helpers both modules need
(:func:`_artists`, :func:`_song`, :func:`_song_start`) live here for that reason.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evaluation import names
from evaluation.clips import GRID_S
from evaluation.results import Emission
from stream_sleuth.recognizers.base import EvalIdentification

if TYPE_CHECKING:
    from evaluation.score import LegScore, Play, Verdict

log = logging.getLogger(__name__)


NEAR_MISS_SIMILARITY = 0.8
RUN_COLUMNS = (
    "leg",
    "recognizer",
    "address",
    "last_address",
    "emissions",
    "hour_key",
    "start",
    "end",
    "source",
    "artist",
    "song",
    "album",
    "reference_artists",
)
NEAR_COLUMNS = (
    *RUN_COLUMNS,
    "play_id",
    "play_carryover",
    "play_artist",
    "play_title",
    "play_album",
    "matched_on",
    "verdict",
)
FP_COLUMNS = (*RUN_COLUMNS, "preflag", "verdict")
STEADY_START_S = GRID_S / 2  # a start estimate that holds to half a grid step is steady
LIKELY_UNLOGGED = "likely-unlogged-correct"


def _artists(
    f: EvalIdentification, references: Mapping[str, tuple[str, ...]] | None
) -> tuple[str, ...]:
    """The artist names an emission names: its own, plus for a local match every name its reference carries.

    The emission's own name stays, so a mapping built from another ``pool.db`` than the one that
    tagged the emission can never score a match worse than without it."""
    if f["source"] == "local" and references and f.get("ref_key", "") in references:
        return (f["artist"], *references[f["ref_key"]])
    return (f["artist"],)


def _song(v: Verdict) -> tuple[str, str, str]:
    f = v.emission.found
    return v.emission.address.hour_key, f["artist"].lower(), f["song"].lower()


def _song_start(f: EvalIdentification) -> float | None:
    """``at + query_offset_s - ref_start_s``; None for an answer without offsets (Shazam's null)."""
    if f.get("query_offset_s") is None or f.get("ref_start_s") is None:
        return None
    return f["at"] + f["query_offset_s"] - f["ref_start_s"]


def wrong_runs(verdicts: Sequence[Verdict]) -> list[list[Verdict]]:
    """Runs of consecutive wrong emissions of one song (``distinct_precision``'s song key).

    A run also ends at a correct emission, which ``distinct_precision``'s runs do not: a song run
    it counts once can contribute more than one wrong run here.
    """
    groups = groupby(verdicts, lambda v: (v.play is None, _song(v)))
    return [list(run) for (wrong, _), run in groups if wrong]


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 3600}:{int(seconds) // 60 % 60:02d}:{int(seconds) % 60:02d}"


def run_row(tag: str, run: Sequence[Verdict]) -> dict[str, Any]:
    """A queue row for a run: leg, the identity it was scored under, first and last clip address,
    count, and hour-file position, with the first emission's metadata."""
    first, last = run[0].emission, run[-1].emission
    f = first.found
    return {
        "leg": tag,
        "recognizer": first.key[1],
        "address": first.address.key,
        "last_address": last.address.key,
        "emissions": len(run),
        "hour_key": first.address.hour_key,
        "start": _clock(f["at"]),
        "end": _clock(last.found["at"] + last.address.length_s),
        "source": f["source"],
        "artist": f["artist"],
        "song": f["song"],
        "album": f["album"],
    }


def _tags(emission: Emission, references: Mapping[str, tuple[str, ...]] | None) -> str:
    """The artist tags of an Olaf emission's reference file, ``" | "``-joined (empty otherwise)."""
    tags = references.get(emission.found.get("ref_key", ""), ()) if references else ()
    return " | ".join(tags)


def _same(a: str | None, b: str | None) -> bool:
    key = names.fuzzy(a)
    return bool(key) and key == names.fuzzy(b)


def _tokens(artist: str | None, title: str | None) -> set[str]:
    """The fuzzy tokens of each field, keyed alone so a credit in one never reaches the other."""
    return set(names.fuzzy(artist).split()) | set(names.fuzzy(title).split())


def _nearness(p: Play, f: EvalIdentification, artists: Sequence[str]) -> tuple[list[str], float]:
    """The fields of ``p`` equal to the emission's under the fuzzy key, and the best Jaccard."""
    fields = [
        field
        for field, hit in (
            ("artist", any(_same(p.artist, a) for a in artists)),
            ("title", _same(p.title, f["song"])),
        )
        if hit
    ]
    theirs = _tokens(p.artist, p.title)
    jaccards = [
        len(theirs & mine) / len(theirs | mine) for a in artists if (mine := _tokens(a, f["song"]))
    ]
    return fields, max(jaccards, default=0.0)


def near_miss(
    run: Sequence[Verdict],
    by_hour: Mapping[str, Sequence[Play]],
    references: Mapping[str, tuple[str, ...]] | None,
) -> tuple[Play, str, Emission] | None:
    """The play nearest a run of wrong emissions, why, and the emission that found it.

    Every emission of the run is judged.

    A candidate is a play whose window holds an emission's ``at`` and that shares ``artist`` or
    ``title`` with it (both can hold, as for a version mismatch) under the fuzzy key, else
    ``similarity``: a Jaccard of the token sets of at least 0.8. The best candidate has the most
    shared fields, then the highest similarity, and the earliest emission wins a tie.
    """
    near = []
    for v in run:
        f = v.emission.found
        artists = _artists(f, references)
        for p in by_hour[v.emission.address.hour_key]:
            if p.window_start_s <= f["at"] <= p.window_end_s:
                fields, similarity = _nearness(p, f, artists)
                if fields or similarity >= NEAR_MISS_SIMILARITY:
                    label = "+".join(fields) or "similarity"
                    near.append(((len(fields), similarity), p, label, v.emission))
    return max(near, key=lambda n: n[0])[1:] if near else None


def near_miss_runs(
    plays: Sequence[Play],
    legs: Mapping[str, LegScore],
    references: Mapping[str, tuple[str, ...]] | None,
) -> list[tuple[dict[str, Any], list[Verdict]]]:
    """The near-miss queue: each run of wrong emissions that is a near miss, as its row (with its
    nearest play and an empty ``verdict``, ``correct`` or ``wrong``, for a person to fill) and the run."""
    by_hour: defaultdict[str, list[Play]] = defaultdict(list)
    for p in plays:
        by_hour[p.hour_key].append(p)
    queued = []
    for tag, ls in legs.items():
        for run in wrong_runs(ls.verdicts):
            if found := near_miss(run, by_hour, references):
                play, matched_on, emission = found
                row = {
                    **run_row(tag, run),
                    "reference_artists": _tags(emission, references),
                    "play_id": play.play_id,
                    "play_carryover": play.carryover,
                    "play_artist": play.artist,
                    "play_title": play.title,
                    "play_album": play.album,
                    "matched_on": matched_on,
                    "verdict": "",
                }
                queued.append((row, run))
    return queued


def _steady(run: Sequence[Verdict]) -> bool:
    """Whether at least two emissions carry offsets and their song-start estimates stay within
    :data:`STEADY_START_S`: a real playback's reference position advances with the wall clock, so the
    estimate holds, while a position that stands still moves it by the whole 15 s grid step."""
    starts = [s for v in run if (s := _song_start(v.emission.found)) is not None]
    return len(starts) >= 2 and max(starts) - min(starts) <= STEADY_START_S


def _agrees(
    run: Sequence[Verdict],
    others: Iterable[Emission],
    references: Mapping[str, tuple[str, ...]] | None,
) -> bool:
    """Whether another recognizer's emission in the run's hour and span, ``[first at, last at +
    capture length)``, names the same song: titles equal under ``names.fuzzy`` and some artist of
    either (:func:`_artists`, so an Olaf file's tags count) equal to some artist of the other."""
    begin = run[0].emission.found["at"]
    end = run[-1].emission.found["at"] + run[-1].emission.address.length_s
    return any(
        begin <= o.found["at"] < end
        and _same(o.found["song"], v.emission.found["song"])
        and any(
            _same(a, b)
            for a in _artists(v.emission.found, references)
            for b in _artists(o.found, references)
        )
        for o in others
        for v in run
    )


def false_positive_runs(
    plays: Sequence[Play],
    legs: Mapping[str, LegScore],
    references: Mapping[str, tuple[str, ...]] | None,
) -> list[tuple[dict[str, Any], list[Verdict]]]:
    """The flowsheet-false-positive queue: each run of wrong emissions the near-miss queue does not
    hold, as its row (a ``preflag`` and an empty ``verdict``, ``wrong``, ``unlogged-correct``, or
    ``talk``) and the run.

    The playlist is an imperfect label, so a wrong emission may be a recognizer error, a song the DJ
    played and did not log, or talk over a bed. ``preflag`` is ``likely-unlogged-correct`` when the
    run's song start is steady (:func:`_steady`) or the other recognizer of the same leg (same
    capture length and profile, the part of the ``<leg>/<recognizer>`` tag before ``/``) agrees
    (:func:`_agrees`); a recognizer not scored in this run cannot agree.
    """
    by_hour: defaultdict[str, list[Play]] = defaultdict(list)
    for p in plays:
        by_hour[p.hour_key].append(p)
    queued = []
    for tag, ls in legs.items():
        others: defaultdict[str, list[Emission]] = defaultdict(list)
        for other, theirs in legs.items():
            if other != tag and other.partition("/")[0] == tag.partition("/")[0]:
                for v in theirs.verdicts:
                    others[v.emission.address.hour_key].append(v.emission)
        for run in wrong_runs(ls.verdicts):
            if near_miss(run, by_hour, references):
                continue
            hour = run[0].emission.address.hour_key
            likely = _steady(run) or _agrees(run, others[hour], references)
            row = {
                **run_row(tag, run),
                "reference_artists": _tags(run[0].emission, references),
                "preflag": LIKELY_UNLOGGED if likely else "",
                "verdict": "",
            }
            queued.append((row, run))
    return queued


# What a person may write in a queue's ``verdict`` column, each as the verdict it stands for.
NEAR_VERDICTS = {"correct": "correct", "wrong": "wrong"}
FP_VERDICTS = {"wrong": "wrong", "unlogged-correct": "correct", "talk": "talk"}
VERDICTS = {NEAR_COLUMNS: NEAR_VERDICTS, FP_COLUMNS: FP_VERDICTS}
KEY_COLUMNS = ("leg", "recognizer", "address")  # a run: the leg's tag, its identity, its first clip


def _key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(row.get(c, "").strip() for c in KEY_COLUMNS)


def _extent(row: Mapping[str, str]) -> tuple[str, str]:
    """A row's ``(last_address, emissions)`` as a spreadsheet may have left them: blanks stripped, and an
    ``emissions`` of ``3.0`` or ``03`` read as ``3`` (anything that is not a number stays as written)."""
    count = row.get("emissions", "").strip()
    try:
        count = str(int(float(count)))
    except (ValueError, OverflowError):
        pass
    return row.get("last_address", "").strip(), count


def read_queue(
    path: Path, columns: Sequence[str]
) -> list[tuple[int, tuple[str, ...], str, tuple[str, str]]]:
    """A queue's rows as ``(row number, key, verdict, extent)``, ``[]`` when ``path`` is absent.

    The row number is the spreadsheet's (the header is row 1). The extent is the run's
    ``(last_address, emissions)`` as labeled (:func:`_extent`). The verdict is what the person's
    entry stands for (``correct``, ``wrong``, or ``talk``; blank stays ``""``, matched without
    regard to case and surrounding blanks). ``SystemExit``, one line naming ``path``, when the
    file is not a queue of this command's (its header, matched without regard to case and past a
    byte-order mark a spreadsheet adds, is not ``columns``; a JSONL store, another CSV, an empty
    file), cannot be read, holds a verdict outside the queue's vocabulary, or holds two rows for
    one run (:data:`KEY_COLUMNS`), which would leave a conflict to file order.
    """
    allowed = VERDICTS[tuple(columns)]
    rows: list[tuple[int, tuple[str, ...], str, tuple[str, str]]] = []
    first: dict[tuple[str, ...], int] = {}
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None or [h.strip().lower() for h in header] != list(columns):
                raise SystemExit(f"{path}: is not a queue of this command's; not overwriting it")
            for number, cells in enumerate(reader, 2):
                if not any(cells):
                    continue
                row = dict(zip(columns, cells, strict=False))
                entry = row.get("verdict", "").strip()
                if entry and entry.lower() not in allowed:
                    raise SystemExit(f"{path}: row {number}: unknown verdict {entry!r}")
                if (key := _key(row)) in first:
                    raise SystemExit(f"{path}: rows {first[key]} and {number} are the same run")
                first[key] = number
                rows.append((number, key, allowed.get(entry.lower(), ""), _extent(row)))
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError, csv.Error):
        raise SystemExit(f"{path}: cannot be read; not overwriting it") from None
    return rows


def write_queue(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a queue as UTF-8 CSV with a byte-order mark, so a spreadsheet shows diacritics, and
    its header always, so an empty queue still says what it holds.

    A person's work is never overwritten: a file with a verdict in it is refused here, and
    :func:`settle_queue` is what keeps it.
    """
    if any(row[2] for row in read_queue(path, columns)):
        raise SystemExit(f"{path}: has verdicts filled in; not overwriting it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)


def settle_queue(
    path: Path, columns: Sequence[str], queued: Sequence[tuple[dict[str, Any], list[Verdict]]]
) -> dict[str, dict[str, str]]:
    """Write the queue of ``queued`` runs unless a person has begun filling in the file at ``path``;
    then keep it, byte for byte, and return each leg's verdicts by emission address.

    A verdict applies to every emission of its run. A filled-in row is matched to a current run by
    :data:`KEY_COLUMNS`; a row that matches none is logged once, in one line, and ignored, and a
    current run the file has no row for stays wrong. A row applies only to a run that still ends where
    it did when labeled (same ``last_address`` and ``emissions``); a run that has grown or shrunk is
    logged with its row and both extents, not applied, and its row is never rewritten. A blank
    verdict gives no entry.
    """
    rows = read_queue(path, columns)
    if not any(row[2] for row in rows):
        write_queue(path, columns, (row for row, _ in queued))
        return {}
    current = {_key(row): (row["leg"], run) for row, run in queued}
    if stale := [str(n) for n, key, *_ in rows if key not in current]:
        log.warning("%s: no current run matches row(s) %s; ignored", path, ", ".join(stale))
    if unqueued := current.keys() - {key for _, key, *_ in rows}:
        runs = "; ".join(f"{leg} {address}" for leg, _, address in sorted(unqueued))
        log.warning(
            "%s: %d current run(s) have no row and stay wrong: %s", path, len(unqueued), runs
        )
    calls: defaultdict[str, dict[str, str]] = defaultdict(dict)
    for number, key, verdict, extent in rows:
        if verdict and key in current:
            tag, run = current[key]
            now = (run[-1].emission.address.key, str(len(run)))
            if extent != now:
                log.warning(
                    "%s: row %d: run has grown or shrunk since it was labeled (last_address %s, %s emissions; now %s, %s); ignored",
                    path,
                    number,
                    *extent,
                    *now,
                )
                continue
            calls[tag].update({v.emission.address.key: verdict for v in run})
    return calls
