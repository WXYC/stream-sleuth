"""WXYC's hourly broadcast archive: hour keys and a never-overwriting fetch.

WXYC-specific. The archive holds one MP3 per hour, keyed
``YYYY/MM/DD/YYYYMMDDHH00.mp3`` by the hour's **America/New_York local time**
(the recorder sets that zone: ``WXYC-Archive/record_audio.pl``), not UTC. Two
DST consequences follow. The fall-back hour (01:00 local) is recorded twice
under one key, and the object holds whichever recording finished uploading
last, so neither pass is trusted and the hour is excluded. The spring-forward
hour (02:00 local) never exists. Every client comes from
:func:`evaluation.s3_readonly.archive_client`. One process fetches into an
archive directory at a time.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

from evaluation.s3_readonly import STREAM_ERRORS, archive_bucket, archive_client

log = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
_KEY_FORMAT = "%Y/%m/%d/%Y%m%d%H00.mp3"


class HourSizeMismatchError(FileExistsError):
    """A local hour file exists with a size other than the object's; it is left alone."""


class ShortReadError(OSError):
    """The download ended before the object's ``ContentLength``; only the ``.part`` remains."""


def hour_key(instant: datetime) -> str | None:
    """Return the key of the archive hour containing ``instant``, or ``None`` for the fall-back hour."""
    if instant.tzinfo is None:
        raise ValueError("hour_key needs a timezone-aware datetime")
    local = instant.astimezone(EASTERN)
    if local.replace(fold=1 - local.fold).utcoffset() != local.utcoffset():
        return None  # the repeated hour (see the module docstring)
    return local.strftime(_KEY_FORMAT)


def hour_start(key: str) -> datetime:
    """Return the instant an hour key's recording starts; the inverse of :func:`hour_key`.

    Raises ``ValueError`` for anything that is not a key :func:`hour_key` can
    produce, including the fall-back and spring-forward hours.
    """
    stem = PurePosixPath(key).stem  # strptime cannot repeat a directive, so parse the name
    start = datetime.strptime(stem, "%Y%m%d%H%M").replace(tzinfo=EASTERN)
    if hour_key(start) != key:
        raise ValueError(f"{key} is not an archive hour key with a single start")
    return start


def _is_missing(exc: Exception) -> bool:
    error = getattr(exc, "response", {}).get("Error", {})
    return error.get("Code") in {"404", "NoSuchKey"}


def fetch(key: str, *, archive_dir: Path, client: Any = None, bucket: str | None = None) -> Path:
    """Download the hour ``key`` to ``archive_dir/key`` unless it is already there.

    ``key`` must be one :func:`hour_key` produces. An existing file of the
    object's size is kept; one of any other size raises
    :class:`HourSizeMismatchError` and is never overwritten. A new download
    streams into a ``.part`` file renamed only once its size matches the
    object's ``ContentLength``; a short read raises :class:`ShortReadError`.
    """
    hour_start(key)  # rejects excluded hours and anything that could escape archive_dir
    client = client or archive_client()
    bucket = bucket or archive_bucket()
    dest = archive_dir / key
    if dest.exists():
        size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
        if dest.stat().st_size != size:
            raise HourSizeMismatchError(
                f"{dest} has {dest.stat().st_size} bytes, the object {size}"
            )
        log.info("have %s", key)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    response = client.get_object(Bucket=bucket, Key=key)
    size = response["ContentLength"]
    with contextlib.closing(response["Body"]) as body, open(part, "wb") as f:
        try:
            for chunk in body.iter_chunks(1 << 20):
                f.write(chunk)
        except STREAM_ERRORS as exc:
            raise ShortReadError(f"short read for {key}: {exc}") from exc
    if part.stat().st_size != size:
        raise ShortReadError(f"short read for {key}: {part.stat().st_size} of {size} bytes")
    part.rename(dest)
    log.info("fetched %s (%.1f MB)", key, size / 1e6)
    return dest


def fetch_all(
    keys: Iterable[str], *, archive_dir: Path, client: Any = None, bucket: str | None = None
) -> list[str]:
    """Fetch every hour in ``keys`` and return the ones that failed.

    Per-hour failures (a missing object, a size mismatch, a short read, a key
    that is not an hour) are logged and skipped. Anything else, such as expired
    credentials or a missing bucket, would fail every hour, so it propagates.
    """
    client = client or archive_client()
    bucket = bucket or archive_bucket()
    failed = []
    for key in keys:
        try:
            fetch(key, archive_dir=archive_dir, client=client, bucket=bucket)
        except Exception as exc:
            if not isinstance(
                exc, (HourSizeMismatchError, ShortReadError, ValueError)
            ) and not _is_missing(exc):
                raise
            log.error("failed %s: %s", key, exc)
            failed.append(key)
    return failed
