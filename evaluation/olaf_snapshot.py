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
"""

from __future__ import annotations

import fcntl
import os
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from evaluation.pool import PoolObject, open_pool_db, stream
from stream_sleuth.recognizers.olaf import OlafRecognizer, snapshot_dir

# Where a query run files its results, inside the snapshot it queried.
RESULTS = "results.jsonl"


class SnapshotError(RuntimeError):
    """A snapshot build or run is refused: frozen, a case-only name clash, or locked."""


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
        with closing(open_pool_db(home / "pool.db")) as db:
            return stream(
                objects, consumer, db=db, staging_dir=home / "staging", client=client, bucket=bucket
            )
