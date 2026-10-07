"""The archive hour-key mapping and the hour fetch, against a synthetic moto bucket."""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import pytest
from moto import mock_aws
from urllib3 import HTTPConnectionPool
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from evaluation import archive
from evaluation.s3_readonly import MissingSettingError, archive_bucket, archive_client
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
        archive_bucket()


@pytest.fixture
def synthetic_archive(monkeypatch, tmp_path_factory):
    """A moto archive bucket, with no real AWS configuration, credentials, or settings visible."""
    for name in list(os.environ):
        if name.startswith(("AWS_", "STREAM_SLEUTH_")):
            monkeypatch.delenv(name)
    aws_dir = tmp_path_factory.mktemp("aws")
    for name, path in (("AWS_CONFIG_FILE", "config"), ("AWS_SHARED_CREDENTIALS_FILE", "creds")):
        (aws_dir / path).write_text("")
        monkeypatch.setenv(name, str(aws_dir / path))
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
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


class _Dropping(io.RawIOBase):
    """A socket stream that yields ``data``, then fails as a dropped or stalled connection does."""

    def __init__(self, data: bytes, error: Exception):
        self._data, self._error = data, error

    def read(self, size=-1):
        if not self._data:
            raise self._error
        data, self._data = self._data, b""
        return data


class _TruncatedHour:
    """The guarded client, but SUMMER_KEY's GetObject body ends 8 bytes early.

    The body is a real botocore ``StreamingBody`` declaring the object's full
    length, so botocore's own stream checks run exactly as they do for a dropped
    connection. This module may not import botocore, so the class comes from the
    response body's MRO; the body may be its ``StreamingChecksumBody`` subclass,
    which checks the length first and raises the same errors.
    """

    def __init__(self, client, raw):
        self._client, self._raw = client, raw

    def __getattr__(self, name):
        return getattr(self._client, name)

    def get_object(self, **kwargs):
        response = self._client.get_object(**kwargs)
        if kwargs["Key"] != SUMMER_KEY:
            return response
        mro = type(response["Body"]).__mro__
        streaming_body = next(c for c in mro if c.__name__ == "StreamingBody")
        response["Body"].close()
        short = HOURS[SUMMER_KEY][:-8]
        return {**response, "Body": streaming_body(self._raw(short), response["ContentLength"])}


_POOL = HTTPConnectionPool("archive.example.test")  # names the host in the error; never connects
_TRUNCATIONS = {
    "eof before content length": io.BytesIO,
    "connection reset": lambda d: _Dropping(d, ProtocolError("Connection broken")),
    "read timeout": lambda d: _Dropping(d, ReadTimeoutError(_POOL, "/", "Read timed out.")),
}


@pytest.mark.usefixtures("synthetic_archive")
@pytest.mark.parametrize("raw", _TRUNCATIONS.values(), ids=_TRUNCATIONS.keys())
def test_short_read_leaves_only_a_part_file(tmp_path, raw):
    with pytest.raises(archive.ShortReadError):
        archive.fetch(
            SUMMER_KEY, archive_dir=tmp_path, client=_TruncatedHour(archive_client(), raw)
        )
    assert not (tmp_path / SUMMER_KEY).exists()
    assert (tmp_path / (SUMMER_KEY + ".part")).read_bytes() == HOURS[SUMMER_KEY][:-8]


@pytest.mark.usefixtures("synthetic_archive")
def test_fetch_all_skips_a_short_read_and_keeps_going(tmp_path):
    client = _TruncatedHour(archive_client(), io.BytesIO)

    failed = archive.fetch_all([SUMMER_KEY, WINTER_KEY], archive_dir=tmp_path, client=client)

    assert failed == [SUMMER_KEY]
    assert not (tmp_path / SUMMER_KEY).exists()
    assert (tmp_path / WINTER_KEY).read_bytes() == HOURS[WINTER_KEY]


class _Recording:
    """The guarded client, recording the name of every client method called."""

    def __init__(self, client):
        self._client, self.calls = client, set()

    def __getattr__(self, name):
        self.calls.add(name)
        return getattr(self._client, name)


@pytest.mark.usefixtures("synthetic_archive")
def test_fetch_all_calls_only_head_object_and_get_object(tmp_path):
    client = _Recording(archive_client())
    missing = "2026/02/01/202602010000.mp3"

    archive.fetch_all([SUMMER_KEY, missing], archive_dir=tmp_path, client=client)
    archive.fetch_all([SUMMER_KEY], archive_dir=tmp_path, client=client)  # the rerun heads it

    assert client.calls == {"head_object", "get_object"}


@pytest.mark.usefixtures("synthetic_archive")
def test_fetch_all_reports_failures_and_keeps_going(tmp_path):
    (tmp_path / SUMMER_KEY).parent.mkdir(parents=True)
    (tmp_path / SUMMER_KEY).write_bytes(b"older recording")
    missing = "2026/02/01/202602010000.mp3"
    fall_back = "2026/11/01/202611010100.mp3"

    failed = archive.fetch_all([SUMMER_KEY, missing, fall_back, WINTER_KEY], archive_dir=tmp_path)

    assert failed == [SUMMER_KEY, missing, fall_back]
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


@pytest.mark.usefixtures("synthetic_archive")
@pytest.mark.parametrize(
    "key",
    ["2026/11/01/202611010100.mp3", "../../2026/08/12/202608121600.mp3", "/tmp/202608121600.mp3"],
)
def test_fetch_rejects_keys_that_are_not_attributable_hours(tmp_path, key):
    with pytest.raises(ValueError):
        archive.fetch(key, archive_dir=tmp_path)
    assert not any(tmp_path.iterdir())


@pytest.mark.usefixtures("synthetic_archive")
def test_fetch_all_stops_on_an_error_that_would_fail_every_hour(tmp_path):
    with pytest.raises(Exception, match="NoSuchBucket"):
        archive.fetch_all([SUMMER_KEY, WINTER_KEY], archive_dir=tmp_path, bucket="no-such-bucket")
