"""The reference-pool inventory and staging loop, against a synthetic moto bucket.

Tag reading is replaced here by a stub so these tests need no ffmpeg; the real
per-format readers are tested in ``tests/integration/test_pool_tags.py``.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3

import pytest
from moto import mock_aws

from evaluation import pool
from tests.unit.test_s3_readonly import seed_objects

ENDPOINT = "https://pool.example.test"
BUCKET = "synthetic-pool"
OBJECTS = {
    "rotation/Heavy/juana-molina/doga/01-la-paradoja.mp3": b"a" * 10,
    "rotation/Heavy/jessica-pratt/on-your-own-love-again/02-back-baby.flac": b"b" * 20,
    "rotation/Heavy/notes.pdf": b"c" * 5,
    "rotation/Light/chuquimamani-condori/edits/03-call-your-name.m4a": b"d" * 30,
}
SYNTHETIC_TAGS = {
    "artist": "Juana Molina",
    "album_artist": None,
    "album": "DOGA",
    "title": "la paradoja",
    "track_number": "1",
    "duration_s": 1.0,
    "bitrate_kbps": 128,
}


@pytest.fixture(autouse=True)
def synthetic_pool(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("AWS_", "STREAM_SLEUTH_", "DIGITAL_ARCHIVE_STORE_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    # Without this, moto passes an unrecognized host through to a real request.
    monkeypatch.setenv("MOTO_S3_CUSTOM_ENDPOINTS", ENDPOINT)
    monkeypatch.setenv("STREAM_SLEUTH_POOL_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("STREAM_SLEUTH_POOL_BUCKET", BUCKET)
    monkeypatch.setenv("STREAM_SLEUTH_POOL_KEY_ID", "testing")
    monkeypatch.setenv("STREAM_SLEUTH_POOL_SECRET", "testing")
    monkeypatch.setattr(pool, "read_tags", lambda path, fmt: dict(SYNTHETIC_TAGS))
    with mock_aws():
        seed_objects(ENDPOINT, BUCKET, OBJECTS)
        yield


@pytest.fixture
def db(tmp_path):
    return pool.open_pool_db(tmp_path / "pool.db")


def _rows(db: sqlite3.Connection) -> dict[str, tuple[str, str | None]]:
    return {
        key: (status, error)
        for key, status, error in db.execute("SELECT key, status, error FROM files")
    }


def test_stage_id_is_the_sha1_of_the_key():
    key = "rotation/Heavy/juana-molina/doga/01-la-paradoja.mp3"
    assert pool.stage_id(key) == hashlib.sha1(key.encode()).hexdigest()


def test_inventory_lists_every_object_under_the_prefixes_with_its_format():
    objects = pool.inventory(["rotation/Heavy/", "rotation/Light/"])

    heavy, light = "rotation/Heavy/", "rotation/Light/"
    assert {(o.key, o.prefix, o.format, o.size) for o in objects} == {
        ("rotation/Heavy/juana-molina/doga/01-la-paradoja.mp3", heavy, "mp3", 10),
        (
            "rotation/Heavy/jessica-pratt/on-your-own-love-again/02-back-baby.flac",
            heavy,
            "flac",
            20,
        ),
        ("rotation/Heavy/notes.pdf", heavy, None, 5),
        ("rotation/Light/chuquimamani-condori/edits/03-call-your-name.m4a", light, "mp4", 30),
    }


def test_summarize_counts_objects_and_bytes_per_prefix_and_format():
    summary = pool.summarize(pool.inventory(["rotation/Heavy/", "rotation/Light/"]))

    assert summary == {
        ("rotation/Heavy/", "mp3"): (1, 10),
        ("rotation/Heavy/", "flac"): (1, 20),
        ("rotation/Heavy/", None): (1, 5),
        ("rotation/Light/", "mp4"): (1, 30),
    }


def test_stream_hands_each_audio_file_to_the_consumer_and_leaves_staging_empty(tmp_path, db):
    staging = tmp_path / "pool-staging"
    seen = []

    def consumer(path, stage):
        assert path.read_bytes() == OBJECTS[next(k for k in OBJECTS if pool.stage_id(k) == stage)]
        seen.append((path.name, stage))

    counts = pool.stream(pool.inventory(["rotation/"]), consumer, db=db, staging_dir=staging)

    assert counts == {"indexed": 3, "skipped_format": 1}
    assert sorted(seen) == sorted(
        (pool.stage_id(k) + os.path.splitext(k)[1], pool.stage_id(k))
        for k in OBJECTS
        if not k.endswith(".pdf")
    )
    assert list(staging.iterdir()) == []
    row = db.execute(
        "SELECT artist, album, title, stage_id FROM files WHERE key LIKE '%paradoja%'"
    ).fetchone()
    assert row == ("Juana Molina", "DOGA", "la paradoja", pool.stage_id(next(iter(OBJECTS))))


def test_a_consumer_that_raises_marks_that_file_failed_and_the_run_continues(tmp_path, db):
    staging = tmp_path / "pool-staging"

    def consumer(path, stage):
        if path.suffix == ".flac":
            raise RuntimeError("indexer crashed")

    counts = pool.stream(pool.inventory(["rotation/"]), consumer, db=db, staging_dir=staging)

    assert counts == {"indexed": 2, "failed": 1, "skipped_format": 1}
    assert list(staging.iterdir()) == []
    failed = {k: v for k, v in _rows(db).items() if v[0] == "failed"}
    assert list(failed) == [k for k in OBJECTS if k.endswith(".flac")]
    assert "indexer crashed" in failed[next(iter(failed))][1]


def test_stale_stages_from_a_crashed_run_are_cleared_before_starting(tmp_path, db):
    staging = tmp_path / "pool-staging"
    staging.mkdir()
    (staging / (pool.stage_id("rotation/Heavy/left-behind.mp3") + ".mp3")).write_bytes(b"x")

    pool.stream([], lambda path, stage: None, db=db, staging_dir=staging)

    assert list(staging.iterdir()) == []


def test_stale_stage_cleanup_removes_only_stage_named_files(tmp_path, db):
    # A misconfigured staging_dir must never cost anything but stage files.
    staging = tmp_path / "pool-staging"
    (staging / "nested").mkdir(parents=True)
    keep = {staging / "pool.db", staging / "notes.txt", staging / "nested"}
    for path in keep - {staging / "nested"}:
        path.write_bytes(b"keep")

    pool.stream([], lambda path, stage: None, db=db, staging_dir=staging)

    assert set(staging.iterdir()) == keep


def test_a_rerun_skips_indexed_files_and_retries_only_failed_ones(tmp_path, db):
    staging = tmp_path / "pool-staging"
    objects = pool.inventory(["rotation/"])

    def flaky(path, stage):
        if path.suffix == ".flac":
            raise RuntimeError("indexer crashed")

    pool.stream(objects, flaky, db=db, staging_dir=staging)
    indexed_before = db.execute(
        "SELECT * FROM files WHERE status = 'indexed' ORDER BY key"
    ).fetchall()

    retried = []
    counts = pool.stream(
        objects, lambda path, stage: retried.append(path.suffix), db=db, staging_dir=staging
    )

    assert retried == [".flac"]
    assert counts == {"indexed": 1, "already_indexed": 2, "skipped_format": 1}
    assert set(indexed_before) <= set(
        db.execute("SELECT * FROM files WHERE status = 'indexed'").fetchall()
    )
    assert {status for status, _ in _rows(db).values()} == {"indexed"}


def test_configured_prefixes_split_the_setting_and_drop_blanks(monkeypatch):
    monkeypatch.setenv("STREAM_SLEUTH_POOL_PREFIXES", " rotation/Heavy/, rotation/Light/ ,,")

    assert pool.configured_prefixes() == ["rotation/Heavy/", "rotation/Light/"]


def test_configured_prefixes_require_the_setting(monkeypatch):
    with pytest.raises(pool.MissingSettingError, match="STREAM_SLEUTH_POOL_PREFIXES"):
        pool.configured_prefixes()


def test_overlapping_prefixes_list_each_object_once():
    objects = pool.inventory(["rotation/", "rotation/Heavy/"])

    assert sorted(o.key for o in objects) == sorted(OBJECTS)


@pytest.mark.parametrize(
    ("artist", "title", "untagged"),
    [
        ("Juana Molina", "la paradoja", 0),
        (None, None, 1),
        ("Juana Molina", None, 1),
        (None, "la paradoja", 1),
    ],
)
def test_files_missing_artist_or_title_are_indexed_and_counted_untagged(
    monkeypatch, tmp_path, db, artist, title, untagged
):
    def read_tags(path, fmt):
        if fmt == "mp3":
            return {**SYNTHETIC_TAGS, "artist": artist, "title": title}
        return dict(SYNTHETIC_TAGS)

    monkeypatch.setattr(pool, "read_tags", read_tags)

    counts = pool.stream(
        pool.inventory(["rotation/"]),
        lambda path, stage: None,
        db=db,
        staging_dir=tmp_path / "pool-staging",
    )

    assert counts["indexed"] == 3
    assert counts["untagged"] == untagged


def test_an_indexed_row_is_never_overwritten_by_a_later_failure(tmp_path, db):
    # The same key twice in one run: the second pass reaches the upsert after the
    # first indexed it, so only the upsert's guard keeps the indexed row intact.
    obj = next(o for o in pool.inventory(["rotation/"]) if o.format == "mp3")
    calls = []

    def fails_second_time(path, stage):
        calls.append(stage)
        if len(calls) > 1:
            raise RuntimeError("indexer crashed")

    counts = pool.stream([obj, obj], fails_second_time, db=db, staging_dir=tmp_path / "staging")

    assert counts == {"indexed": 1, "failed": 1}
    assert _rows(db) == {obj.key: ("indexed", None)}
