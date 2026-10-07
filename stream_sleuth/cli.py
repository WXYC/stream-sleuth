"""Command line for the local recognizer's index.

    python -m stream_sleuth.cli index build --home SNAPSHOT_DIR PATH ID [PATH ID ...]

stores each audio file in the Olaf snapshot under ``SNAPSHOT_DIR`` with its identifier.
``SNAPSHOT_DIR`` must be absolute, outside the checkout, and not the home directory;
``$STREAM_SLEUTH_DATA_DIR/olaf/<snapshot>/`` is the place for it.
"""

import argparse
import sys

from .recognizers.olaf import OlafError, OlafRecognizer


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
        help="the snapshot directory (Olaf's HOME): absolute, e.g. $STREAM_SLEUTH_DATA_DIR/olaf/NAME",
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
    try:
        recognizer.store(pairs)
    except OlafError as e:
        parser.exit(1, f"{parser.prog}: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
