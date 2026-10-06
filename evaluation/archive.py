"""WXYC's hourly broadcast archive: hour keys and a never-overwriting fetch.

WXYC-specific. The archive holds one MP3 per hour, keyed
``YYYY/MM/DD/YYYYMMDDHH00.mp3`` by the hour's **America/New_York local time**
(the recorder sets that zone: ``WXYC-Archive/record_audio.pl``), not UTC. Two
DST consequences follow: the fall-back hour (01:00 local, recorded twice under
one key, the later recording overwriting the earlier) cannot be attributed and
is excluded, and the spring-forward hour (02:00 local) never exists. Every
client comes from :func:`evaluation.s3_readonly.archive_client`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

from evaluation.s3_readonly import MissingSettingError, archive_client

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
_KEY_FORMAT = "%Y/%m/%d/%Y%m%d%H00.mp3"


class HourSizeMismatchError(FileExistsError):
    """A local hour file exists with a size other than the object's; it is left alone."""


def archive_bucket() -> str:
    """Return the archive bucket named by ``STREAM_SLEUTH_ARCHIVE_BUCKET``."""
    bucket = os.environ.get("STREAM_SLEUTH_ARCHIVE_BUCKET")
    if not bucket:
        raise MissingSettingError("set STREAM_SLEUTH_ARCHIVE_BUCKET")
    return bucket


def hour_key(instant: datetime) -> str | None:
    """Return the key of the archive hour containing ``instant``, or ``None`` for the fall-back hour."""
    if instant.tzinfo is None:
        raise ValueError("hour_key needs a timezone-aware datetime")
    local = instant.astimezone(EASTERN)
    if local.replace(fold=1 - local.fold).utcoffset() != local.utcoffset():
        return None  # the repeated hour: its object holds only the later recording
    return local.strftime(_KEY_FORMAT)


def hour_start(key: str) -> datetime:
    """Return the instant an hour key's recording starts; the inverse of :func:`hour_key`."""
    stem = PurePosixPath(key).stem  # strptime cannot repeat a directive, so parse the name
    start = datetime.strptime(stem, "%Y%m%d%H%M").replace(tzinfo=EASTERN)
    if hour_key(start) != key:
        raise ValueError(f"{key} is not an archive hour key with a single start")
    return start


def fetch(key: str, *, archive_dir: Path, client: Any = None, bucket: str | None = None) -> Path:
    """Download the hour ``key`` to ``archive_dir/key`` unless it is already there.

    The object streams into a ``.part`` file that is renamed only once its size
    matches ``HeadObject``'s; a short read raises and leaves the ``.part``. An
    existing file of the right size is kept; one of any other size raises
    :class:`HourSizeMismatchError` and is never overwritten.
    """
    client = client or archive_client()
    bucket = bucket or archive_bucket()
    dest = archive_dir / key
    size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    if dest.exists():
        if dest.stat().st_size == size:
            return dest
        raise HourSizeMismatchError(f"{dest} has {dest.stat().st_size} bytes, the object {size}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    with open(part, "wb") as f:
        for chunk in body.iter_chunks(1 << 20):
            f.write(chunk)
    if part.stat().st_size != size:
        raise OSError(f"short read for {key}: {part.stat().st_size} of {size} bytes")
    part.rename(dest)
    log.info("fetched %s (%.1f MB)", key, size / 1e6)
    return dest


def fetch_all(
    keys: Iterable[str], *, archive_dir: Path, client: Any = None, bucket: str | None = None
) -> list[str]:
    """Fetch every hour in ``keys``, logging and continuing past failures; return the failed keys."""
    client = client or archive_client()
    bucket = bucket or archive_bucket()
    failed = []
    for key in keys:
        try:
            fetch(key, archive_dir=archive_dir, client=client, bucket=bucket)
        except Exception as exc:  # recorded per hour; the run continues
            log.error("failed %s: %s", key, exc)
            failed.append(key)
    return failed
