"""The archive hour-key mapping and the hour fetch, against a synthetic moto bucket."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from moto import mock_aws

from evaluation import archive
from evaluation.s3_readonly import MissingSettingError, archive_client
from tests.unit.test_s3_readonly import seed_objects

BUCKET = "synthetic-archive"
SUMMER_KEY = "2026/08/12/202608121600.mp3"
WINTER_KEY = "2026/01/15/202601150300.mp3"
HOURS = {SUMMER_KEY: b"h" * 64, WINTER_KEY: b"w" * 32}


def utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("instant", "key"),
    [
        # EDT (UTC-4): 20:30 UTC is 16:30 local.
        (utc(2026, 8, 12, 20, 30), "2026/08/12/202608121600.mp3"),
        # EST (UTC-5): 08:59 UTC is 03:59 local.
        (utc(2026, 1, 15, 8, 59), "2026/01/15/202601150300.mp3"),
        # A UTC instant after local midnight's UTC equivalent stays on the local date.
        (utc(2026, 8, 13, 3, 15), "2026/08/12/202608122300.mp3"),
        (utc(2026, 8, 13, 4, 0), "2026/08/13/202608130000.mp3"),
        # Spring forward (2026-03-08): 01:59 EST is followed by 03:00 EDT; no 02:00 hour.
        (utc(2026, 3, 8, 6, 59), "2026/03/08/202603080100.mp3"),
        (utc(2026, 3, 8, 7, 0), "2026/03/08/202603080300.mp3"),
        # Fall back (2026-11-01): the hours either side of the repeated 01:00 hour.
        (utc(2026, 11, 1, 4, 59), "2026/11/01/202611010000.mp3"),
        (utc(2026, 11, 1, 7, 0), "2026/11/01/202611010200.mp3"),
    ],
)
def test_hour_key_names_the_eastern_local_hour(instant, key):
    assert archive.hour_key(instant) == key


@pytest.mark.parametrize("instant", [utc(2026, 11, 1, 5, 30), utc(2026, 11, 1, 6, 30)])
def test_fall_back_hour_is_excluded(instant):
    """01:00 local is recorded twice under one key and the later overwrites the earlier."""
    assert archive.hour_key(instant) is None


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        archive.hour_key(datetime(2026, 8, 12, 20, 30))


@pytest.mark.parametrize(
    ("key", "start"),
    [
        ("2026/08/12/202608121600.mp3", utc(2026, 8, 12, 20)),
        ("2026/01/15/202601150300.mp3", utc(2026, 1, 15, 8)),
    ],
)
def test_hour_start_inverts_hour_key(key, start):
    assert archive.hour_start(key) == start
    assert archive.hour_key(start) == key


def test_archive_bucket_requires_the_setting(monkeypatch):
    monkeypatch.delenv("STREAM_SLEUTH_ARCHIVE_BUCKET", raising=False)
    with pytest.raises(MissingSettingError):
        archive.archive_bucket()


@pytest.fixture
def synthetic_archive(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("AWS_", "STREAM_SLEUTH_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("STREAM_SLEUTH_ARCHIVE_BUCKET", BUCKET)
    with mock_aws():
        seed_objects(None, BUCKET, HOURS)
        yield


@pytest.mark.usefixtures("synthetic_archive")
def test_fetch_writes_the_hour_and_a_rerun_skips_it(tmp_path):
    dest = archive.fetch(SUMMER_KEY, archive_dir=tmp_path)
    assert dest == tmp_path / SUMMER_KEY
    assert dest.read_bytes() == HOURS[SUMMER_KEY]
    mtime = dest.stat().st_mtime_ns

    assert archive.fetch(SUMMER_KEY, archive_dir=tmp_path) == dest
    assert dest.stat().st_mtime_ns == mtime


@pytest.mark.usefixtures("synthetic_archive")
def test_existing_file_of_the_wrong_size_is_left_untouched(tmp_path):
    dest = tmp_path / SUMMER_KEY
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"older recording")
    with pytest.raises(archive.HourSizeMismatchError):
        archive.fetch(SUMMER_KEY, archive_dir=tmp_path)
    assert dest.read_bytes() == b"older recording"


class _ShortHead:
    """The guarded client, but HeadObject reports more bytes than GetObject returns."""

    def __init__(self, client):
        self._client = client

    def head_object(self, **kwargs):
        head = self._client.head_object(**kwargs)
        return {**head, "ContentLength": head["ContentLength"] + 8}

    def get_object(self, **kwargs):
        return self._client.get_object(**kwargs)


@pytest.mark.usefixtures("synthetic_archive")
def test_short_read_leaves_only_a_part_file(tmp_path):
    with pytest.raises(OSError, match="short read"):
        archive.fetch(SUMMER_KEY, archive_dir=tmp_path, client=_ShortHead(archive_client()))
    assert not (tmp_path / SUMMER_KEY).exists()
    assert (tmp_path / (SUMMER_KEY + ".part")).exists()


@pytest.mark.usefixtures("synthetic_archive")
def test_fetch_all_reports_failures_and_keeps_going(tmp_path):
    (tmp_path / SUMMER_KEY).parent.mkdir(parents=True)
    (tmp_path / SUMMER_KEY).write_bytes(b"older recording")
    missing = "2026/02/01/202602010000.mp3"

    failed = archive.fetch_all([SUMMER_KEY, missing, WINTER_KEY], archive_dir=tmp_path)

    assert failed == [SUMMER_KEY, missing]
    assert (tmp_path / WINTER_KEY).read_bytes() == HOURS[WINTER_KEY]


@pytest.mark.parametrize(
    "key",
    [
        "2026/11/01/202611010100.mp3",  # fall-back: two starts
        "2026/03/08/202603080200.mp3",  # spring-forward: never recorded
        "2026/08/13/202608121600.mp3",  # directory disagrees with the name
    ],
)
def test_hour_start_rejects_keys_without_a_single_start(key):
    with pytest.raises(ValueError):
        archive.hour_start(key)
