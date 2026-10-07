"""The station-neutral reader for ``selection.json``, the frozen record at the ``plays.jsonl`` boundary.

A station producer writes the file (WXYC's is ``evaluation.corpus``); the leg runner and the
scorer read it here, and ``corpus`` reads it here too before applying its own checks. This
module imports nothing station-specific, so ``corpus`` may import it and never the reverse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn


def fail(path: Path, problem: str) -> NoReturn:
    """Refuse a run with one line naming ``path``; the form every selection refusal takes."""
    raise SystemExit(f"{path}: {problem}")


def load_selection(path: Path) -> dict[str, Any]:
    """The whole record of a ``selection.json``, after the structural checks every reader needs.

    The file must be UTF-8 JSON holding an object whose ``hours`` is a non-empty object keyed
    by hour, each hour's label an object whose ``subset`` is a JSON boolean (a hand-edited
    ``"false"`` would be truthy). Other fields, such as ``shortfalls``, come back untouched, and
    a caller checks whatever else it relies on. Anything wrong exits with one line naming
    ``path`` (and the hour, when one is at fault), before the caller does any work.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        fail(path, f"cannot read: {e}")
    except json.JSONDecodeError as e:
        fail(path, f"not JSON ({e}); pass the selection.json that `select` wrote")
    if not isinstance(record, dict):
        fail(path, "not an object")
    hours = record.get("hours")
    if hours is None:
        fail(path, "no `hours` key")
    if not isinstance(hours, dict):
        fail(path, "`hours` is not an object")
    if not hours:
        fail(path, "`hours` is empty")
    for key, label in hours.items():
        if not isinstance(label, dict):
            fail(path, f"`hours` entry {key} is not an object")
        if not isinstance(label.get("subset"), bool):
            fail(path, f"`hours` entry {key}: subset {label.get('subset')!r} is not a boolean")
    return record


def read_hours(selection: Path) -> dict[str, list[str]]:
    """The hour keys of ``selection.json``, read once: ``all`` of them, and the ``subset``.

    :func:`~evaluation.selection.load_selection` checks the structure (a hand-edited
    ``"false"`` would be truthy, and ``hours.txt`` passed by mistake is not JSON); this adds
    at least one hour in the subset, or the subset legs would report ``done`` having sent
    nothing. Anything wrong refuses the run with one line naming the file.
    """
    hours = load_selection(selection)["hours"]
    if not (chosen := [key for key, label in hours.items() if label["subset"]]):
        fail(selection, "no hour with subset: true")
    return {"all": list(hours), "subset": chosen}
