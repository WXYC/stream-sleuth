"""``tests.queues.edit_queue``: it leaves a queue the way a person's edit or save would."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.queues import edit_queue, read_queue

PLAIN = "leg,verdict,flag\r\n12s/shazam,,True\r\n12s/olaf,,False\r\n"
BOM = b"\xef\xbb\xbf"


@pytest.fixture
def queue(tmp_path: Path) -> Path:
    path = tmp_path / "queue.csv"
    path.write_text(PLAIN, encoding="utf-8", newline="")
    return path


def test_the_first_row_is_the_default_target(queue: Path) -> None:
    edit_queue(queue, verdict="correct")

    assert [r["verdict"] for r in read_queue(queue)] == ["correct", ""]


def test_where_selects_every_row_whose_columns_equal_it(queue: Path) -> None:
    edit_queue(queue, {"leg": "12s/olaf"}, verdict="wrong")

    assert [r["verdict"] for r in read_queue(queue)] == ["", "wrong"]


def test_a_where_that_matches_no_row_is_refused(queue: Path) -> None:
    with pytest.raises(ValueError, match="no row"):
        edit_queue(queue, {"leg": "6s/olaf"}, verdict="wrong")


def test_a_plain_edit_changes_only_the_cells(queue: Path) -> None:
    saved = edit_queue(queue, verdict="")

    assert saved == PLAIN.encode()


def test_a_spreadsheet_save_adds_a_mark_crlf_quotes_and_upper_case_booleans(queue: Path) -> None:
    saved = edit_queue(queue, spreadsheet=True)

    assert (
        saved
        == BOM + b'"leg","verdict","flag"\r\n"12s/shazam","","TRUE"\r\n"12s/olaf","","FALSE"\r\n'
    )
    assert read_queue(queue)[0]["flag"] == "TRUE"
