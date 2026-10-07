"""The station-neutral ``selection.json`` reader: each structural refusal, once."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.selection import load_selection

HOUR = "2026/08/12/202608121400.mp3"
OTHER = "2026/08/12/202608122000.mp3"


def written(path: Path, content: object) -> Path:
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return path


def test_the_whole_record_is_returned_so_a_caller_can_read_its_other_fields(
    tmp_path: Path,
) -> None:
    record = {
        "export": "export",
        "shortfalls": {"canonical-low": 1},
        "hours": {
            HOUR: {"subset": True, "group": "contrast"},
            OTHER: {"subset": False, "anything": [1, 2]},
        },
    }
    assert load_selection(written(tmp_path / "selection.json", record)) == record


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        pytest.param(None, "cannot read", id="no-file"),
        pytest.param(b"\xff\xfe\x00", "cannot read", id="invalid-utf8"),
        pytest.param(f"{HOUR}\n{OTHER}\n", "not JSON", id="hours-txt-by-mistake"),
        pytest.param([{"hours": {}}], "not an object", id="top-level-list"),
        pytest.param({"export": "export"}, "no `hours` key", id="no-hours"),
        pytest.param({"hours": [HOUR, OTHER]}, "`hours` is not an object", id="hours-list"),
        pytest.param({"hours": {}}, "`hours` is empty", id="empty-hours"),
        pytest.param(
            {"hours": {HOUR: {"subset": True}, OTHER: True}},
            f"`hours` entry {OTHER} is not an object",
            id="label-not-an-object",
        ),
        pytest.param(
            {"hours": {HOUR: {"subset": True}, OTHER: {"subset": "false"}}},
            f"`hours` entry {OTHER}: subset 'false' is not a boolean",
            id="subset-string",
        ),
        pytest.param(
            {"hours": {HOUR: {"subset": 1}}},
            f"`hours` entry {HOUR}: subset 1 is not a boolean",
            id="subset-number",
        ),
        pytest.param(
            {"hours": {HOUR: {"group": "contrast"}}},
            f"`hours` entry {HOUR}: subset None is not a boolean",
            id="no-subset-label",
        ),
    ],
)
def test_a_structurally_bad_selection_exits_with_one_line_naming_the_file(
    tmp_path: Path, content: Any, problem: str
) -> None:
    path = tmp_path / "selection.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        written(path, content)
    with pytest.raises(SystemExit) as refusal:
        load_selection(path)
    message = str(refusal.value.code)
    assert message.startswith(f"{path}: ") and problem in message and "\n" not in message
