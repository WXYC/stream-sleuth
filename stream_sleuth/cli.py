"""Command line for the local recognizer's index.

    python -m stream_sleuth.cli index build --home SNAPSHOT_DIR PATH ID [PATH ID ...]

stores each audio file in the Olaf snapshot under ``SNAPSHOT_DIR`` with its identifier.
``SNAPSHOT_DIR`` must be absolute, outside the checkout, and not the home directory, and
is refused when it holds ``results.jsonl`` or any file of an evaluation-harness snapshot
(``pool.db``, ``built.json``, ``building``), or when a build or query run holds its ``.lock``.
Keep it outside ``$STREAM_SLEUTH_DATA_DIR/olaf/``, which belongs to the harness: build a
harness snapshot with ``evaluation.olaf_snapshot.build_snapshot``.
"""

import argparse
import fcntl
import os
import sys
from pathlib import Path

from .recognizers.olaf import OlafError, OlafRecognizer

# The harness's snapshot files (RESULTS, POOL_DB, MARKER and BUILDING, and the file
# ``snapshot_lock`` locks, in ``evaluation/olaf_snapshot.py``), mirrored as literals because
# the runtime package never imports ``evaluation``; ``tests/unit/test_cli_snapshot_guard.py``
# fails if they drift.
SNAPSHOT_RESULTS = "results.jsonl"
SNAPSHOT_POOL_DB = "pool.db"
SNAPSHOT_MARKER = "built.json"
SNAPSHOT_BUILDING = "building"
SNAPSHOT_LOCK = ".lock"


def _refusal(home: Path) -> str | None:
    """Why ``home`` must not be built into: it holds results or a harness snapshot, or a run holds its lock.

    Looks at the file names only. The lock probe is a non-blocking ``flock`` released at
    once, and never creates the lock file; a run that starts after the probe is not seen.
    """
    if (home / SNAPSHOT_RESULTS).exists():
        return f"{home} already has results ({home / SNAPSHOT_RESULTS}); not building"
    for name in (SNAPSHOT_POOL_DB, SNAPSHOT_MARKER, SNAPSHOT_BUILDING):
        if (home / name).exists():
            return (
                f"{home} is a harness snapshot ({name} present); not building "
                "(build those with evaluation.olaf_snapshot.build_snapshot)"
            )
    try:
        fd = os.open(home / SNAPSHOT_LOCK, os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return f"{home}: another build or run holds it"
    finally:
        os.close(fd)
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the command line; returns 0, or exits 2 on a usage error and 1 if Olaf fails."""
    parser = argparse.ArgumentParser(prog="python -m stream_sleuth.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="manage a local fingerprint index")
    actions = index.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="store audio files in an Olaf snapshot")
    build.add_argument(
        "--home",
        required=True,
        help="the snapshot directory (Olaf's HOME): absolute, outside the checkout and outside $STREAM_SLEUTH_DATA_DIR/olaf/",
    )
    build.add_argument("--olaf-bin", help="default: STREAM_SLEUTH_OLAF_BIN, else olaf on PATH")
    build.add_argument(
        "items",
        nargs="+",
        metavar="PATH ID",
        help="audio file and identifier pairs; the harness expects evaluation.pool.stage_id(key)",
    )
    args = parser.parse_args(argv)
    if len(args.items) % 2:
        parser.error("items must be PATH ID pairs")
    pairs = list(zip(args.items[::2], args.items[1::2], strict=True))
    try:
        recognizer = OlafRecognizer(args.home, olaf_bin=args.olaf_bin)
    except OlafError as e:
        parser.error(str(e))
    if refusal := _refusal(recognizer.home):
        parser.error(refusal)
    try:
        recognizer.store(pairs)
    except OlafError as e:
        parser.exit(1, f"{parser.prog}: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
