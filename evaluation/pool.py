"""The reference pool: inventory, stream-and-index staging, and per-format tags.

A station's reference audio lives in an S3-compatible store and is too large to
mirror, so :func:`stream` fetches one object at a time into a staging directory,
reads its tags into ``pool.db``, hands the staged file to a consumer (the index
build), and deletes it whether the consumer succeeds or raises. Every client
comes from :func:`evaluation.s3_readonly.pool_client`.

The stage id is ``sha1(key)``, so a staged file's name, and the identifier the
consumer stores it under, are deterministic functions of the object key;
``pool.db`` keeps the ``stage_id -> key`` mapping. A rerun skips keys already
``indexed`` and retries only ``failed`` ones, so successful rows are never
rewritten.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from mutagen.aac import AAC
from mutagen.easyid3 import EasyID3
from mutagen.easymp4 import EasyMP4
from mutagen.flac import FLAC
from mutagen.id3 import ID3NoHeaderError
from mutagen.mp3 import EasyMP3
from mutagen.wave import WAVE

from evaluation.s3_readonly import MissingSettingError, pool_bucket, pool_client

log = logging.getLogger(__name__)

# File extension -> format. Objects with any other extension are counted by the
# inventory but never fetched.
FORMATS = {
    ".mp3": "mp3",
    ".aac": "aac",
    ".wav": "wav",
    ".flac": "flac",
    ".m4a": "mp4",
    ".mp4": "mp4",
}

# pool.db column -> (easy tag name, ID3 frame for WAV's RIFF `id3 ` chunk).
_TAG_FIELDS = {
    "artist": ("artist", "TPE1"),
    "album_artist": ("albumartist", "TPE2"),
    "album": ("album", "TALB"),
    "title": ("title", "TIT2"),
    "track_number": ("tracknumber", "TRCK"),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    key TEXT PRIMARY KEY,
    stage_id TEXT NOT NULL UNIQUE,
    prefix TEXT NOT NULL,
    format TEXT NOT NULL,
    size INTEGER NOT NULL,
    artist TEXT, album_artist TEXT, album TEXT, title TEXT, track_number TEXT,
    duration_s REAL, bitrate_kbps INTEGER,
    status TEXT NOT NULL,  -- indexed | failed
    error TEXT
)
"""

Consumer = Callable[[Path, str], None]

# A staged file's name: its stage id plus the object's extension. Stale-stage
# cleanup removes only names of this shape, so a misconfigured staging_dir can
# never cost anything else.
_STAGE_NAME = re.compile(r"[0-9a-f]{40}\.[0-9a-z]+")


@dataclass(frozen=True)
class PoolObject:
    """One object in the store, as listed by :func:`inventory`."""

    key: str
    prefix: str
    size: int
    format: str | None


def configured_prefixes() -> list[str]:
    """Return the comma-separated prefixes in ``STREAM_SLEUTH_POOL_PREFIXES``."""
    prefixes = [p.strip() for p in os.environ.get("STREAM_SLEUTH_POOL_PREFIXES", "").split(",")]
    if not any(prefixes):
        raise MissingSettingError(
            "set STREAM_SLEUTH_POOL_PREFIXES, e.g. rotation/Heavy/,rotation/Light/"
        )
    return [p for p in prefixes if p]


def stage_id(key: str) -> str:
    """Return the deterministic stage id for an object key: its SHA-1 in hex."""
    return hashlib.sha1(key.encode()).hexdigest()


def inventory(
    prefixes: Iterable[str], *, client: Any = None, bucket: str | None = None
) -> list[PoolObject]:
    """List every object under ``prefixes`` with its size and format, fetching nothing.

    A key under two overlapping prefixes is listed once, under the first.
    """
    client = client or pool_client()
    bucket = bucket or pool_bucket()
    objects = []
    seen: set[str] = set()
    for prefix in prefixes:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                if item["Key"] in seen:
                    continue
                seen.add(item["Key"])
                fmt = FORMATS.get(PurePosixPath(item["Key"]).suffix.lower())
                objects.append(PoolObject(item["Key"], prefix, item["Size"], fmt))
    return objects


def summarize(objects: Iterable[PoolObject]) -> dict[tuple[str, str | None], tuple[int, int]]:
    """Count objects and bytes per ``(prefix, format)``; format ``None`` is unsupported."""
    summary: dict[tuple[str, str | None], tuple[int, int]] = {}
    for obj in objects:
        count, size = summary.get((obj.prefix, obj.format), (0, 0))
        summary[obj.prefix, obj.format] = (count + 1, size + obj.size)
    return summary


def read_tags(path: Path, fmt: str) -> dict[str, Any]:
    """Read tags and stream info with the reader for ``fmt``; never ``mutagen.File()``."""
    if fmt == "wav":
        audio: Any = WAVE(path)
        frames = audio.tags or {}
        tags = {
            col: str(frames[frame].text[0]) if frame in frames else None
            for col, (_, frame) in _TAG_FIELDS.items()
        }
    else:
        if fmt == "aac":
            audio = AAC(path)
            try:
                easy: Any = EasyID3(path)
            except ID3NoHeaderError:
                easy = {}
        else:
            audio = {"mp3": EasyMP3, "flac": FLAC, "mp4": EasyMP4}[fmt](path)
            easy = audio.tags or {}
        tags = {col: (easy.get(name) or [None])[0] for col, (name, _) in _TAG_FIELDS.items()}
    bitrate = getattr(audio.info, "bitrate", 0)
    return {
        **tags,
        "duration_s": round(audio.info.length, 3),
        "bitrate_kbps": round(bitrate / 1000) if bitrate else None,
    }


def open_pool_db(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) ``pool.db`` at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute(SCHEMA)
    return db


def _record(
    db: sqlite3.Connection, obj: PoolObject, tags: dict[str, Any], status: str, error: str | None
) -> None:
    row = {
        "key": obj.key,
        "stage_id": stage_id(obj.key),
        "prefix": obj.prefix,
        "format": obj.format,
        "size": obj.size,
        **tags,
        "status": status,
        "error": error,
    }
    columns = ", ".join(row)
    updates = ", ".join(f"{c} = excluded.{c}" for c in row if c != "key")
    db.execute(
        f"INSERT INTO files ({columns}) VALUES ({', '.join('?' * len(row))}) "
        f"ON CONFLICT(key) DO UPDATE SET {updates} WHERE files.status != 'indexed'",
        list(row.values()),
    )
    db.commit()


def stream(
    objects: Iterable[PoolObject],
    consumer: Consumer,
    *,
    db: sqlite3.Connection,
    staging_dir: Path,
    client: Any = None,
    bucket: str | None = None,
) -> Counter[str]:
    """Fetch, tag, and hand each audio object to ``consumer(staged_path, stage_id)``.

    One file is staged at a time and deleted when the consumer returns or raises.
    Stage files left in ``staging_dir`` by a crashed run are removed first; nothing
    else there is touched. Returns counts of ``indexed``, ``failed``,
    ``already_indexed``, and ``skipped_format``, plus ``untagged``: files indexed
    this run that lack an artist or a title tag (also counted in ``indexed``).
    """
    client = client or pool_client()
    bucket = bucket or pool_bucket()
    staging_dir.mkdir(parents=True, exist_ok=True)
    for stale in staging_dir.iterdir():
        if stale.is_file() and _STAGE_NAME.fullmatch(stale.name):
            stale.unlink()
    done = {key for (key,) in db.execute("SELECT key FROM files WHERE status = 'indexed'")}
    counts: Counter[str] = Counter()
    for obj in objects:
        if obj.format is None:
            counts["skipped_format"] += 1
            continue
        if obj.key in done:
            counts["already_indexed"] += 1
            continue
        path = staging_dir / (stage_id(obj.key) + PurePosixPath(obj.key).suffix.lower())
        tags: dict[str, Any] = {}
        try:
            # get_object, not download_file: no s3transfer threads between the
            # call and the read-only guard.
            response = client.get_object(Bucket=bucket, Key=obj.key)
            with closing(response["Body"]) as body, open(path, "wb") as f:
                for chunk in body.iter_chunks(1 << 20):
                    f.write(chunk)
            tags = read_tags(path, obj.format)
            consumer(path, stage_id(obj.key))
        except Exception as exc:  # recorded per file; the run continues
            log.warning("failed %s: %s", stage_id(obj.key), exc)
            _record(db, obj, tags, "failed", f"{type(exc).__name__}: {exc}"[:500])
            counts["failed"] += 1
        else:
            _record(db, obj, tags, "indexed", None)
            counts["indexed"] += 1
            if not (tags["artist"] and tags["title"]):
                counts["untagged"] += 1
        finally:
            path.unlink(missing_ok=True)
        if (counts["indexed"] + counts["failed"]) % 50 == 0:
            log.info("pool progress: %s", dict(counts))
    log.info("pool done: %s", dict(counts))
    return counts
