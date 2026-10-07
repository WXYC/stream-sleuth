"""The Olaf snapshot build: one ``pool.db`` and one ``store`` per staged file, per snapshot.

``OlafRecognizer.store`` is replaced by a recorder, so these tests need no Olaf build;
the real one is exercised by ``tests/integration/test_olaf_snapshot.py``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from moto import mock_aws

from evaluation import olaf_snapshot, pool
from evaluation.olaf_snapshot import (
    BUILDING,
    MARKER,
    POOL_DB,
    RESULTS,
    SnapshotError,
    build_snapshot,
    main,
    mark_built,
    open_snapshot,
    require_built,
    snapshot_lock,
)
from stream_sleuth.paths import CHECKOUT, DataPathError
from stream_sleuth.recognizers.olaf import (
    OLAF_COMMIT,
    SNAPSHOT_INDEX,
    OlafRecognizer,
    recognizer_identity,
    snapshot_dir,
)
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


def marker(snapshot: str = "rotation") -> dict:
    return json.loads((snapshot_dir(snapshot) / MARKER).read_text())


def test_a_completed_build_writes_a_marker_naming_the_commit_and_the_counts(stored):
    build_snapshot("rotation", pool.inventory(["rotation/"]))

    assert marker() == {"olaf_commit": OLAF_COMMIT, "indexed": 2, "failed": 0, "source": "build"}
    assert require_built(snapshot_dir("rotation"))["indexed"] == 2


def test_a_rerun_counts_the_snapshots_files_not_the_runs(stored):
    objects = pool.inventory(["rotation/"])
    build_snapshot("rotation", objects)
    build_snapshot("rotation", objects)  # indexes nothing new

    assert marker()["indexed"] == 2


def test_a_build_with_failed_files_is_complete_and_records_how_many(monkeypatch, stored):
    def broken(path, fmt):
        if fmt == "flac":
            raise ValueError("bad tags")
        return {**TAGS, "duration_s": 1.0}

    monkeypatch.setattr(pool, "read_tags", broken)

    assert build_snapshot("rotation", pool.inventory(["rotation/"])) == {"indexed": 1, "failed": 1}

    assert (marker()["indexed"], marker()["failed"]) == (1, 1)


def test_no_marker_is_written_when_the_build_raises(monkeypatch, stored):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(olaf_snapshot, "stream", interrupted)

    with pytest.raises(KeyboardInterrupt):
        build_snapshot("rotation", pool.inventory(["rotation/"]))

    assert not (snapshot_dir("rotation") / MARKER).exists()


def test_a_marker_is_replaced_atomically_and_leaves_no_temp_file(stored):
    build_snapshot("rotation", pool.inventory(["rotation/"]))

    assert sorted(p.name for p in snapshot_dir("rotation").iterdir() if "built" in p.name) == [
        MARKER
    ]


def test_an_interrupted_rerun_is_refused_even_with_a_stale_marker(monkeypatch, stored):
    objects = pool.inventory(["rotation/"])
    build_snapshot("rotation", objects)
    monkeypatch.setattr(olaf_snapshot, "stream", lambda *a, **k: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        build_snapshot("rotation", objects)

    assert (snapshot_dir("rotation") / MARKER).exists()  # the stale one
    with pytest.raises(SnapshotError, match="interrupted"):
        require_built(snapshot_dir("rotation"))


def interrupt_after_one_file(monkeypatch):
    """``OlafRecognizer.store`` that indexes one file, then is interrupted on the second."""
    state = {"calls": 0, "armed": True}

    def store(self, items):
        state["calls"] += 1
        if state["armed"] and state["calls"] > 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(OlafRecognizer, "store", store)
    return state


def test_an_interrupted_build_is_refused_by_a_run_and_by_mark_built_and_a_rerun_heals_it(
    monkeypatch, stored
):
    objects = pool.inventory(["rotation/"])
    home = snapshot_dir("rotation")
    interruption = interrupt_after_one_file(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        build_snapshot("rotation", objects)
    # one of two files is indexed, the Olaf index exists, and there is no marker
    (home / SNAPSHOT_INDEX).parent.mkdir(parents=True)
    (home / SNAPSHOT_INDEX).write_bytes(b"lmdb")

    for refuse in (lambda: require_built(home), lambda: mark_built("rotation")):
        with pytest.raises(SnapshotError, match="interrupted") as error:
            refuse()
        assert "rerun the build" in str(error.value)
        assert "--mark-built" not in str(error.value)
        assert "\n" not in str(error.value)
    assert not (home / MARKER).exists()

    interruption["armed"] = False
    build_snapshot("rotation", objects)
    assert not (home / BUILDING).exists()
    assert require_built(home)["indexed"] == 2


def test_the_in_progress_sentinel_is_in_place_before_the_first_store(stored, monkeypatch):
    seen = []
    monkeypatch.setattr(
        OlafRecognizer,
        "store",
        lambda self, items: seen.append((self.home / BUILDING).exists()),
    )

    build_snapshot("rotation", pool.inventory(["rotation/"]))

    assert seen == [True, True]
    assert not (snapshot_dir("rotation") / BUILDING).exists()  # removed once marked


def test_the_marker_is_written_by_atomic_replace_of_a_complete_temp_file(monkeypatch, stored):
    replaced = []
    real = olaf_snapshot.os.replace

    def spy(src, dst):
        if Path(dst).name == MARKER:
            assert not Path(dst).exists()  # the target is not written in place first
            replaced.append((Path(src).name, json.loads(Path(src).read_text())))
        real(src, dst)

    monkeypatch.setattr(olaf_snapshot.os, "replace", spy)

    build_snapshot("rotation", pool.inventory(["rotation/"]))

    assert [(src, marker["indexed"]) for src, marker in replaced] == [(MARKER + ".tmp", 2)]


def test_a_marker_is_replaced_not_appended_to(stored):
    home = snapshot_dir("rotation")
    home.mkdir(parents=True)
    (home / MARKER).write_text("not json at all")

    build_snapshot("rotation", pool.inventory(["rotation/"]))

    assert marker()["indexed"] == 2


@pytest.mark.parametrize(
    ("contents", "refusal"),
    [
        (None, "no completion marker"),
        ("{", "unreadable"),
        ("[]", "unreadable"),
        (json.dumps({"olaf_commit": OLAF_COMMIT}), "unreadable"),
        (json.dumps({"olaf_commit": "0" * 40, "indexed": 3, "failed": 0}), "another Olaf commit"),
    ],
)
def test_a_run_refuses_a_snapshot_without_a_usable_marker(tmp_path, contents, refusal):
    home = tmp_path / "rotation"
    home.mkdir()
    if contents is not None:
        (home / MARKER).write_text(contents)

    with pytest.raises(SnapshotError, match=refusal) as error:
        require_built(home)

    assert "\n" not in str(error.value)


def test_a_run_logs_the_failed_count(tmp_path, caplog):
    home = tmp_path / "rotation"
    home.mkdir()
    (home / MARKER).write_text(json.dumps({"olaf_commit": OLAF_COMMIT, "indexed": 5, "failed": 2}))

    with caplog.at_level("INFO"):
        require_built(home)

    assert "5 indexed" in caplog.text and "2 failed" in caplog.text


def pre_marker_snapshot(statuses=("indexed", "indexed", "failed"), index=True) -> Path:
    """A snapshot as an earlier build left it: its ``pool.db`` and Olaf index, no marker."""
    home = snapshot_dir("rotation")
    with pool.open_pool_db(home / "pool.db") as db:
        for i, status in enumerate(statuses):
            key = f"rotation/Heavy/{i}.mp3"
            db.execute(
                "INSERT INTO files (key, stage_id, prefix, format, size, status)"
                " VALUES (?, ?, 'rotation/', 'mp3', 1, ?)",
                (key, hashlib.sha1(key.encode()).hexdigest(), status),
            )
    if index:
        (home / SNAPSHOT_INDEX).parent.mkdir(parents=True)
        (home / SNAPSHOT_INDEX).write_bytes(b"lmdb")
    return home


def fingerprint(home: Path) -> dict[str, tuple[int, bytes]]:
    return {
        str(p.relative_to(home)): (p.stat().st_mtime_ns, p.read_bytes())
        for p in sorted(home.rglob("*"))
        if p.is_file() and p.name != ".lock"
    }


def test_mark_built_marks_a_pre_marker_snapshot_without_touching_anything_else(tmp_path):
    home = pre_marker_snapshot()
    (home / RESULTS).write_text("{}\n")
    before = fingerprint(home)

    assert mark_built("rotation") == {
        "olaf_commit": OLAF_COMMIT,
        "indexed": 2,
        "failed": 1,
        "source": "mark-built",
    }

    assert marker() == {**marker(), "indexed": 2, "failed": 1, "source": "mark-built"}
    assert {k: v for k, v in fingerprint(home).items() if k != MARKER} == before
    assert require_built(home)["indexed"] == 2


@pytest.mark.parametrize(
    ("kwargs", "refusal"),
    [
        ({"statuses": ("indexed", "staged")}, "staged"),
        ({"index": False}, "Olaf index"),
    ],
)
def test_mark_built_refuses_an_unfinished_snapshot(kwargs, refusal):
    home = pre_marker_snapshot(**kwargs)

    with pytest.raises(SnapshotError, match=refusal):
        mark_built("rotation")

    assert not (home / MARKER).exists()


def test_mark_built_opens_pool_db_read_only_and_never_read_write(monkeypatch):
    home = pre_marker_snapshot()
    opened = []

    def read_write(path):
        raise AssertionError("pool.db opened read-write")

    real = olaf_snapshot.open_read_only
    monkeypatch.setattr(olaf_snapshot, "open_pool_db", read_write)
    monkeypatch.setattr(olaf_snapshot, "open_read_only", lambda p: opened.append(p) or real(p))

    mark_built("rotation")

    assert opened == [home / "pool.db"]


def test_mark_built_refuses_an_existing_marker_and_leaves_it_alone(stored):
    build_snapshot("rotation", pool.inventory(["rotation/"]))
    before = (snapshot_dir("rotation") / MARKER).read_bytes()

    with pytest.raises(SnapshotError, match="already"):
        mark_built("rotation")

    assert (snapshot_dir("rotation") / MARKER).read_bytes() == before


def test_mark_built_refuses_a_snapshot_that_is_not_there_and_creates_nothing(tmp_path):
    with pytest.raises(SnapshotError, match="no snapshot"):
        mark_built("rotation")

    assert not snapshot_dir("rotation").exists()


def test_mark_built_is_refused_while_a_build_or_run_holds_the_snapshot():
    home = pre_marker_snapshot()
    with snapshot_lock(home), pytest.raises(SnapshotError, match="another"):
        mark_built("rotation")
    assert not (home / MARKER).exists()


def test_the_mark_built_command_marks_and_a_refusal_is_one_line(capsys):
    pre_marker_snapshot()

    assert main(["--mark-built", "rotation"]) == 0
    assert (snapshot_dir("rotation") / MARKER).is_file()
    with pytest.raises(SystemExit, match="already") as refusal:
        main(["--mark-built", "rotation"])
    assert "\n" not in str(refusal.value)
    with pytest.raises(SystemExit, match="plain path component"):
        main(["--mark-built", "../rotation"])


def built_snapshot(name="rotation") -> Path:
    """A snapshot with a ``pool.db`` and a completion marker, as a finished build leaves it."""
    home = snapshot_dir(name)
    pool.open_pool_db(home / POOL_DB).close()
    (home / MARKER).write_text(
        json.dumps({"olaf_commit": OLAF_COMMIT, "indexed": 0, "failed": 0, "source": "build"})
    )
    return home


def test_open_snapshot_returns_the_home_identity_results_store_and_a_read_only_pool_db():
    home = built_snapshot()

    with open_snapshot("rotation", 7) as snapshot:
        assert snapshot.home == home
        assert snapshot.identity == recognizer_identity("rotation", 7)
        assert snapshot.store.path == home / RESULTS
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            snapshot.db.execute("DELETE FROM files")
        assert snapshot.db.execute("SELECT COUNT(*) FROM files").fetchone() == (0,)

    with pytest.raises(sqlite3.ProgrammingError):  # closed on exit
        snapshot.db.execute("SELECT 1")
    assert not (home / RESULTS).exists()  # opening never files results
    assert not (home / ".lock").exists()  # and, without lock=True, takes no lock


def test_open_snapshot_holds_the_lock_only_when_asked_and_checks_the_marker_under_it(monkeypatch):
    home = built_snapshot()
    held_during_check = []

    def check(path):
        with pytest.raises(SnapshotError, match="another"):
            with snapshot_lock(path):
                pass
        held_during_check.append(True)
        return {}

    monkeypatch.setattr(olaf_snapshot, "require_built", check)
    with open_snapshot("rotation", 12, lock=True):
        with pytest.raises(SnapshotError, match="another"):
            with snapshot_lock(home):
                pass
    assert held_during_check == [True]
    with snapshot_lock(home):  # released on exit
        pass
    with snapshot_lock(home), open_snapshot("rotation", 12):  # a reader never blocks on a run
        pass


@pytest.fixture
def opened(monkeypatch):
    """Every ``pool.db`` path the helper opens, so a refusal can be shown to open nothing."""
    paths = []
    real = olaf_snapshot.open_read_only

    def spy(path):
        paths.append(path)
        return real(path)

    monkeypatch.setattr(olaf_snapshot, "open_read_only", spy)
    return paths


def test_open_snapshot_does_not_open_pool_db_when_the_lock_is_held(opened):
    home = built_snapshot()
    with snapshot_lock(home), pytest.raises(SnapshotError, match="another"):
        with open_snapshot("rotation", 12, lock=True):
            raise AssertionError("not reached")
    assert opened == []


@pytest.mark.parametrize("lock", [False, True])
@pytest.mark.parametrize(
    ("setup", "name", "refusal"),
    [
        ("none", "rotation", "no snapshot here; build it first"),
        ("no_pool_db", "rotation", "no snapshot here; build it first"),
        ("built", "Rotation", "differs only in case"),
        ("no_marker", "rotation", "no completion marker"),
        ("interrupted", "rotation", "interrupted"),
        ("none", "../rotation", "plain path component"),
    ],
)
def test_open_snapshot_refuses_in_one_line_and_opens_nothing(setup, name, refusal, lock, opened):
    root = snapshot_dir("rotation").parent
    if setup == "no_pool_db":
        snapshot_dir("rotation").mkdir(parents=True)
    elif setup != "none":
        home = built_snapshot()
        if setup == "no_marker":
            (home / MARKER).unlink()
        if setup == "interrupted":
            (home / BUILDING).write_text("")

    def tree() -> list[str]:  # the lock file is the one thing a refusal under lock=True may leave
        return sorted(str(p) for p in root.rglob("*") if p.name != ".lock")

    before = tree() if root.exists() else []

    with pytest.raises(SnapshotError, match=refusal) as error:
        with open_snapshot(name, 12, lock=lock):
            raise AssertionError("not reached")

    assert "\n" not in str(error.value)
    assert (tree() if root.exists() else []) == before
    assert opened == []  # no refusal gets as far as opening pool.db
    if setup == "none":
        assert not root.exists()
