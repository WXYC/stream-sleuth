"""The runtime ``index build`` CLI keeps off a harness snapshot, one with results, or one a run holds.

The runtime package never imports ``evaluation``, so ``stream_sleuth/cli.py`` mirrors the
harness's file names as literals; the drift test here is the one place both packages meet.
"""

from __future__ import annotations

import fcntl
import importlib
import os

import pytest


@pytest.fixture
def cli(fresh_recognizer):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    return importlib.import_module("stream_sleuth.cli")


@pytest.fixture
def stored(cli, monkeypatch):
    """Every ``store`` call's home and pairs; the real Olaf is never run."""
    calls = []
    monkeypatch.setattr(cli.OlafRecognizer, "store", lambda self, pairs: calls.append(pairs))
    return calls


def build(cli, home):
    return cli.main(["index", "build", "--home", str(home), "/a.mp3", "id-a"])


def test_a_fresh_home_still_builds(cli, stored, tmp_path):
    assert build(cli, tmp_path / "snap") == 0
    assert stored == [[("/a.mp3", "id-a")]]


def test_an_existing_home_without_results_or_a_lock_still_builds(cli, stored, tmp_path):
    (tmp_path / "snap" / ".olaf").mkdir(parents=True)
    assert build(cli, tmp_path / "snap") == 0
    assert stored == [[("/a.mp3", "id-a")]]


def test_a_home_with_results_is_refused_in_one_line_and_left_untouched(
    cli, stored, tmp_path, capsys
):
    home = tmp_path / "snap"
    home.mkdir()
    (home / cli.SNAPSHOT_RESULTS).write_text('{"key": "x"}\n')
    with pytest.raises(SystemExit) as exit_info:
        build(cli, home)
    assert exit_info.value.code == 2
    refusal = capsys.readouterr().err.strip().splitlines()[-1]  # after argparse's usage
    assert "already has results" in refusal
    assert str(home) in refusal
    assert stored == []
    assert sorted(p.name for p in home.iterdir()) == [cli.SNAPSHOT_RESULTS]


@pytest.mark.parametrize("name", ["pool.db", "built.json", "building"])
def test_a_home_holding_any_harness_snapshot_file_is_refused_and_left_untouched(
    cli, stored, tmp_path, capsys, name
):
    home = tmp_path / "snap"
    home.mkdir()
    (home / name).write_bytes(b"Juana Molina")
    with pytest.raises(SystemExit) as exit_info:
        build(cli, home)
    assert exit_info.value.code == 2
    refusal = capsys.readouterr().err.strip().splitlines()[-1]  # after argparse's usage
    assert "harness snapshot" in refusal
    assert name in refusal
    assert "build_snapshot" in refusal
    assert str(home) in refusal
    assert stored == []
    assert [p.name for p in home.iterdir()] == [name]
    assert (home / name).read_bytes() == b"Juana Molina"


def test_a_home_whose_lock_is_held_is_refused_and_the_lock_stays_held(
    cli, stored, tmp_path, capsys
):
    home = tmp_path / "snap"
    home.mkdir()
    fd = os.open(home / cli.SNAPSHOT_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as exit_info:
            build(cli, home)
        assert exit_info.value.code == 2
        assert "another build or run holds" in capsys.readouterr().err
        assert stored == []
        # The probe's own descriptor is closed: this holder still owns the lock.
        with pytest.raises(BlockingIOError):
            other = os.open(home / cli.SNAPSHOT_LOCK, os.O_RDWR)
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(other)
    finally:
        os.close(fd)


def test_a_stale_unheld_lock_file_does_not_block_and_the_probe_releases_it(cli, stored, tmp_path):
    home = tmp_path / "snap"
    home.mkdir()
    (home / cli.SNAPSHOT_LOCK).write_bytes(b"")
    assert build(cli, home) == 0
    fd = os.open(home / cli.SNAPSHOT_LOCK, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # free after the build
    finally:
        os.close(fd)


def test_the_probe_does_not_create_the_lock_file(cli, stored, tmp_path):
    home = tmp_path / "snap"
    home.mkdir()
    assert build(cli, home) == 0
    assert not (home / cli.SNAPSHOT_LOCK).exists()


@pytest.mark.parametrize(
    ("literal", "constant"),
    [
        ("SNAPSHOT_RESULTS", "RESULTS"),
        ("SNAPSHOT_POOL_DB", "POOL_DB"),
        ("SNAPSHOT_MARKER", "MARKER"),
        ("SNAPSHOT_BUILDING", "BUILDING"),
    ],
)
def test_the_file_name_literals_match_the_harness_constants(cli, literal, constant):
    from evaluation import olaf_snapshot

    assert getattr(cli, literal) == getattr(olaf_snapshot, constant)


def test_the_lock_literal_matches_the_file_the_harness_locks(cli, tmp_path):
    from evaluation import olaf_snapshot

    home = tmp_path / "snap"
    home = tmp_path / "snap"
    with olaf_snapshot.snapshot_lock(home):
        assert [p.name for p in home.iterdir()] == [cli.SNAPSHOT_LOCK]
