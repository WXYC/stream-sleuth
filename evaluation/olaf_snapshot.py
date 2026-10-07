"""Build one Olaf snapshot from the reference pool: its own ``pool.db``, one ``store`` per file.

A snapshot is ``$STREAM_SLEUTH_DATA_DIR/olaf/<snapshot>/``: Olaf's ``HOME``, with the
snapshot's own ``pool.db`` and staging directory inside it. ``pool.db`` records which keys
are ``indexed`` and ``stream()`` skips those, so a snapshot must never share one: a second
snapshot over the first's ``pool.db`` would index nothing. This lives here, not beside the
adapter, because the live recognizer never imports the harness.

A snapshot's stored results are filed under an identity that names the snapshot, not its
contents, so three rules keep one name meaning one index: a build is refused once the
snapshot has results (:data:`RESULTS`), a name that differs only in case from an existing
snapshot is refused (one directory on a case-insensitive volume), and a build or a query run
holds the snapshot's lock (:func:`snapshot_lock`), so two never overlap.

A build that returns normally then writes a completion marker (:data:`MARKER`, naming the
Olaf commit and the indexed and failed counts), and a run refuses a snapshot without one
(:func:`require_built`), so an interrupted build is never queried and its misses frozen
into stored results. A build with ``failed`` files counts as complete; every run logs how
many. A snapshot built before markers existed is marked by hand with
``python -m evaluation.olaf_snapshot --mark-built NAME`` (:func:`mark_built`).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from evaluation.pool import PoolObject, open_pool_db, open_read_only, stream
from stream_sleuth.paths import DataPathError
from stream_sleuth.recognizers.olaf import (
    OLAF_COMMIT,
    SNAPSHOT_INDEX,
    OlafRecognizer,
    snapshot_dir,
)

log = logging.getLogger(__name__)

# Where a query run files its results, inside the snapshot it queried.
RESULTS = "results.jsonl"
# The completion marker a finished build (or ``--mark-built``) leaves in the snapshot.
MARKER = "built.json"


class SnapshotError(RuntimeError):
    """A snapshot build or run is refused: frozen, a case-only name clash, locked, or not built."""


def checked_snapshot_dir(snapshot: str) -> Path:
    """:func:`snapshot_dir`, refused (``SnapshotError``) when another snapshot's name differs only in case."""
    home = snapshot_dir(snapshot)
    siblings = home.parent.iterdir() if home.parent.is_dir() else ()
    if clash := [
        p.name for p in siblings if p.name != snapshot and p.name.lower() == snapshot.lower()
    ]:
        raise SnapshotError(f"snapshot {snapshot!r} differs only in case from {clash}")
    return home


@contextmanager
def snapshot_lock(home: Path) -> Iterator[None]:
    """Hold an exclusive, non-blocking ``flock`` on ``<home>/.lock``; ``SnapshotError`` if it is held."""
    home.mkdir(parents=True, exist_ok=True)
    fd = os.open(home / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SnapshotError(f"{home}: another build or run holds it") from None
        yield
    finally:
        os.close(fd)


def build_snapshot(
    snapshot: str,
    objects: Iterable[PoolObject],
    *,
    olaf_bin: str | None = None,
    client: Any = None,
    bucket: str | None = None,
) -> Counter[str]:
    """Stream ``objects`` into the named snapshot and return ``stream()``'s counts.

    Each staged file is stored under its pool stage id as soon as it is fetched, because
    ``stream()`` deletes it when the consumer returns. A rerun on the same snapshot skips
    what it already indexed. Raises :class:`stream_sleuth.paths.DataPathError` for a bad
    snapshot name or data directory before anything is created, and :class:`SnapshotError`
    for a case-only name clash, a snapshot that already holds results (extending its index
    would change what they were filed under), or one another build or run holds.
    """
    home = checked_snapshot_dir(snapshot)
    olaf = OlafRecognizer(home, olaf_bin=olaf_bin)

    def consumer(path: Path, stage: str) -> None:
        olaf.store([(str(path), stage)])

    with snapshot_lock(home):
        if (home / RESULTS).exists():
            raise SnapshotError(
                f"{snapshot!r} already has results ({home / RESULTS}); not building"
            )
        (home / MARKER).unlink(missing_ok=True)  # an unfinished rerun must not stay "built"
        with closing(open_pool_db(home / "pool.db")) as db:
            counts = stream(
                objects, consumer, db=db, staging_dir=home / "staging", client=client, bucket=bucket
            )
            _write_marker(home, db, "build")
            return counts


def _write_marker(home: Path, db: sqlite3.Connection, source: str) -> dict[str, Any]:
    """Write ``<home>/built.json`` atomically from ``db``'s counts; the caller holds the lock."""
    by_status = dict(db.execute("SELECT status, COUNT(*) FROM files GROUP BY status"))
    marker = {
        "olaf_commit": OLAF_COMMIT,
        "indexed": by_status.get("indexed", 0),
        "failed": by_status.get("failed", 0),
        "source": source,
    }
    temp = home / (MARKER + ".tmp")
    temp.write_text(json.dumps(marker) + "\n")
    os.replace(temp, home / MARKER)
    return marker


def require_built(home: Path) -> dict[str, Any]:
    """Return the snapshot's marker and log its counts, or raise ``SnapshotError`` in one line.

    Refused: no marker (an interrupted or pre-marker build), one that cannot be read, or
    one that names another Olaf commit (its index is not this commit's).
    """
    path = home / MARKER
    if not path.is_file():
        raise SnapshotError(
            f"{home}: no completion marker ({MARKER}); finish the build, or for a snapshot "
            "built before markers run: python -m evaluation.olaf_snapshot --mark-built NAME"
        )
    try:
        marker = json.loads(path.read_text())
        commit, indexed, failed = marker["olaf_commit"], marker["indexed"], marker["failed"]
    except (ValueError, KeyError, TypeError):
        raise SnapshotError(f"{path}: unreadable completion marker") from None
    if commit != OLAF_COMMIT:
        raise SnapshotError(f"{path}: built by another Olaf commit ({commit}, not {OLAF_COMMIT})")
    log.info("snapshot %s: %s indexed, %s failed", home.name, indexed, failed)
    return marker  # type: ignore[no-any-return]


def mark_built(snapshot: str) -> dict[str, Any]:
    """Mark a snapshot built before markers existed as complete; touch nothing else of it.

    Under the snapshot's lock, and only if there is no marker yet, its Olaf index exists, and
    its ``pool.db`` (opened read-only) holds no row that is neither ``indexed`` nor
    ``failed`` (a ``staged`` one). ``pool.db`` records only finished files, so it cannot
    prove the build ran to the end: the operator vouches for that by running this.
    """
    home = checked_snapshot_dir(snapshot)
    if not (home / "pool.db").is_file():
        raise SnapshotError(f"{home}: no snapshot here")
    with snapshot_lock(home):
        if (home / MARKER).exists():
            raise SnapshotError(f"{home}: already has a completion marker; not marking")
        if not (home / SNAPSHOT_INDEX).is_file():
            raise SnapshotError(f"{home}: no Olaf index ({SNAPSHOT_INDEX}); not marking")
        with closing(open_read_only(home / "pool.db")) as db:
            if unfinished := db.execute(
                "SELECT COUNT(*) FROM files WHERE status NOT IN ('indexed', 'failed')"
            ).fetchone()[0]:
                raise SnapshotError(f"{home}: {unfinished} staged files in pool.db; not marking")
            return _write_marker(home, db, "mark-built")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mark a pre-marker Olaf snapshot as built.")
    parser.add_argument("--mark-built", metavar="NAME", required=True, help="the snapshot's name")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        marker = mark_built(args.mark_built)
    except (SnapshotError, DataPathError) as refusal:
        raise SystemExit(str(refusal)) from None
    log.info(
        "marked %s built: %s indexed, %s failed",
        args.mark_built,
        marker["indexed"],
        marker["failed"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
