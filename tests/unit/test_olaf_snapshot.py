"""The Olaf snapshot build: one ``pool.db`` and one ``store`` per staged file, per snapshot.

``OlafRecognizer.store`` is replaced by a recorder, so these tests need no Olaf build;
the real one is exercised by ``tests/integration/test_olaf_snapshot.py``.
"""

from __future__ import annotations

import pytest
from moto import mock_aws

from evaluation import pool
from evaluation.olaf_snapshot import RESULTS, SnapshotError, build_snapshot, snapshot_lock
from stream_sleuth.paths import CHECKOUT, DataPathError
from stream_sleuth.recognizers.olaf import OlafRecognizer, snapshot_dir
from tests.unit.test_s3_readonly import seed_objects

ENDPOINT = "https://pool.example.test"
BUCKET = "synthetic-pool"
OBJECTS = {
    "rotation/Heavy/juana-molina/doga/01-la-paradoja.mp3": b"a" * 10,
    "rotation/Heavy/jessica-pratt/on-your-own-love-again/02-back-baby.flac": b"b" * 20,
}
TAGS = {"artist": "Juana Molina", "album": "DOGA", "title": "la paradoja"}


@pytest.fixture(autouse=True)
def synthetic_pool(monkeypatch, aws_isolated_env, tmp_path):
    aws_isolated_env.use_pool(ENDPOINT, BUCKET)
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(pool, "read_tags", lambda path, fmt: {**TAGS, "duration_s": 1.0})
    with mock_aws():
        seed_objects(ENDPOINT, BUCKET, OBJECTS)
        yield


@pytest.fixture
def stored(monkeypatch):
    """Every ``store`` call as a list of (home, [(path name, identifier)])."""
    calls = []

    def record(self, items):
        calls.append((self.home, [(path.rsplit("/", 1)[-1], sid) for path, sid in items]))

    monkeypatch.setattr(OlafRecognizer, "store", record)
    return calls


def test_each_staged_file_gets_its_own_store_call_in_the_snapshot(tmp_path, stored):
    counts = build_snapshot("rotation", pool.inventory(["rotation/"]))

    home = (tmp_path / "data" / "olaf" / "rotation").resolve()
    assert counts == {"indexed": 2}
    assert [h for h, _ in stored] == [home, home]
    assert sorted(items[0][1] for _, items in stored) == sorted(map(pool.stage_id, OBJECTS))
    assert all(len(items) == 1 for _, items in stored)
    assert (home / "pool.db").is_file()


def test_a_second_snapshot_indexes_every_file_again(tmp_path, stored):
    objects = pool.inventory(["rotation/"])

    first = build_snapshot("rotation", objects)
    second = build_snapshot("rotation-b", objects)
    rerun = build_snapshot("rotation", objects)

    assert first == second == {"indexed": 2}
    assert rerun == {"already_indexed": 2}
    assert len(stored) == 4
    assert (tmp_path / "data" / "olaf" / "rotation-b" / "pool.db").is_file()


def test_a_data_directory_inside_the_checkout_is_refused_before_anything_is_made(
    monkeypatch, stored
):
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(CHECKOUT / "scratch"))

    with pytest.raises(DataPathError):
        build_snapshot("rotation", pool.inventory(["rotation/"]))

    assert not (CHECKOUT / "scratch").exists()
    assert stored == []


def test_a_build_is_refused_once_the_snapshot_has_results(tmp_path, stored):
    objects = pool.inventory(["rotation/"])
    build_snapshot("rotation", objects)
    (tmp_path / "data" / "olaf" / "rotation" / RESULTS).write_text("{}\n")

    with pytest.raises(SnapshotError, match="results"):
        build_snapshot("rotation", objects)

    assert len(stored) == 2  # the refused build stored nothing


def test_a_name_differing_only_in_case_from_an_existing_snapshot_is_refused(tmp_path, stored):
    objects = pool.inventory(["rotation/"])
    build_snapshot("rotation", objects)

    with pytest.raises(SnapshotError, match="rotation"):
        build_snapshot("Rotation", objects)

    assert [p.name for p in (tmp_path / "data" / "olaf").iterdir()] == ["rotation"]
    assert len(stored) == 2


def test_two_builds_of_one_snapshot_cannot_overlap(stored):
    objects = pool.inventory(["rotation/"])
    with snapshot_lock(snapshot_dir("rotation")):
        with pytest.raises(SnapshotError, match="another"):
            build_snapshot("rotation", objects)
        assert stored == []
    assert build_snapshot("rotation", objects) == {"indexed": 2}
