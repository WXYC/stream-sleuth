"""A person's edits to a review-queue CSV, for the score and queue tests.

``edit_queue`` reads a queue the way ``evaluation.queues`` does (a byte-order mark is allowed) and
writes it back as a person would leave it, so a test holds the bytes a rescore must keep.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path


def read_queue(path: Path) -> list[dict[str, str]]:
    """The queue's rows, as a spreadsheet's save of it would also read."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def edit_queue(
    path: Path,
    where: Mapping[str, str] | None = None,
    *,
    spreadsheet: bool = False,
    **cells: str,
) -> bytes:
    """Set ``cells`` in the rows whose columns equal ``where`` (default: the first row); return the bytes.

    With ``spreadsheet=True`` the file is saved as Excel or Sheets would: a byte-order mark, CRLF
    line endings, every field quoted, and booleans upper-cased. With no ``cells`` that is the only
    change.
    """
    rows = read_queue(path)
    chosen = rows[:1] if where is None else [r for r in rows if r.items() >= where.items()]
    if not chosen:
        raise ValueError(f"no row of {path.name} matches {dict(where or {})}")
    for row in chosen:
        row.update(cells)
    if spreadsheet:
        rows = [{k: v.upper() if v in ("True", "False") else v for k, v in r.items()} for r in rows]
    with path.open("w", encoding="utf-8-sig" if spreadsheet else "utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            list(rows[0]),
            quoting=csv.QUOTE_ALL if spreadsheet else csv.QUOTE_MINIMAL,
            lineterminator="\r\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path.read_bytes()
