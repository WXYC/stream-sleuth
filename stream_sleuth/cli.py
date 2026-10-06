"""Command line for the local recognizer's index.

    python -m stream_sleuth.cli index build --home SNAPSHOT_DIR PATH ID [PATH ID ...]

stores each audio file in the Olaf snapshot under ``SNAPSHOT_DIR`` with its identifier.
"""

import argparse
import sys

from .recognizers.olaf import OlafRecognizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stream-sleuth")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="manage a local fingerprint index")
    actions = index.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="store audio files in an Olaf snapshot")
    build.add_argument("--home", required=True, help="the snapshot directory (Olaf's HOME)")
    build.add_argument("--olaf-bin", help="default: STREAM_SLEUTH_OLAF_BIN, else olaf on PATH")
    build.add_argument(
        "items", nargs="+", metavar="PATH ID", help="audio file and identifier pairs"
    )
    args = parser.parse_args(argv)
    if len(args.items) % 2:
        parser.error("items must be PATH ID pairs")
    pairs = list(zip(args.items[::2], args.items[1::2], strict=True))
    OlafRecognizer(args.home, olaf_bin=args.olaf_bin).store(pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
