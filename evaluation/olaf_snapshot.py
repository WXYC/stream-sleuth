"""Build one Olaf snapshot from the reference pool: its own ``pool.db``, one ``store`` per file.

A snapshot is ``$STREAM_SLEUTH_DATA_DIR/olaf/<snapshot>/``: Olaf's ``HOME``, with the
snapshot's own ``pool.db`` and staging directory inside it. ``pool.db`` records which keys
are ``indexed`` and ``stream()`` skips those, so a snapshot must never share one: a second
snapshot over the first's ``pool.db`` would index nothing. This lives here, not beside the
adapter, because the live recognizer never imports the harness.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from evaluation.pool import PoolObject, open_pool_db, stream
from stream_sleuth.recognizers.olaf import OlafRecognizer, snapshot_dir


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
    snapshot name or data directory before anything is created.
    """
    home = snapshot_dir(snapshot)
    olaf = OlafRecognizer(home, olaf_bin=olaf_bin)

    def consumer(path: Path, stage: str) -> None:
        olaf.store([(str(path), stage)])

    with closing(open_pool_db(home / "pool.db")) as db:
        return stream(
            objects, consumer, db=db, staging_dir=home / "staging", client=client, bucket=bucket
        )
