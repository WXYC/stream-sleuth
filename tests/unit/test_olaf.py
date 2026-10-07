"""The Olaf adapter: per-snapshot ``HOME``, ``store --with-ids``, and ``query --format json``.

A fake ``olaf`` script stands in for the binary: it records its argv and ``HOME`` and
prints whatever JSON the test gives it, so these tests need no Zig build.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[2]

# The record the plan quotes from Olaf's README (plan §6): with query_offset 0.000
# and query_start 1.936, the match begins 1.936 s into the clip.
README_RECORD = {
    "query_index": 1,
    "total_queries": 1,
    "query_path": "query.mp3",
    "query_offset": 0.000,
    "matches": [
        {
            "match_count": 41,
            "query_start": 1.936,
            "query_stop": 10.112,
            "path": "chuquimamani-condori-call-your-name",
            "match_identifier": 488372097,
            "reference_start": 70.864,
            "reference_stop": 79.040,
        }
    ],
}


def query_object(offset, *matches):
    return {"query_offset": offset, "matches": [dict(m) for m in matches]}


def match(path, count, query_start=0.0, reference_start=0.0):
    return {
        "match_count": count,
        "query_start": query_start,
        "path": path,
        "reference_start": reference_start,
    }


@pytest.fixture
def olaf(fresh_recognizer):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    return importlib.import_module("stream_sleuth.recognizers.olaf")


@pytest.fixture
def fake_olaf(tmp_path):
    """Write an executable fake ``olaf``; returns (path, set_output, calls)."""
    record = tmp_path / "calls.jsonl"
    reply = tmp_path / "reply.json"
    reply.write_text(json.dumps({"stdout": "", "code": 0}))
    script = tmp_path / "olaf"
    script.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(f"""
            import json, os, sys
            with open({str(record)!r}, "a") as f:
                f.write(json.dumps({{"argv": sys.argv[1:], "home": os.environ.get("HOME")}}) + "\\n")
            reply = json.load(open({str(reply)!r}))
            sys.stdout.write(reply["stdout"])
            sys.stderr.write(reply.get("stderr", ""))
            sys.exit(reply["code"])
        """)
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    def set_output(stdout="", code=0, stderr=""):
        reply.write_text(json.dumps({"stdout": stdout, "code": code, "stderr": stderr}))

    def calls():
        return (
            [json.loads(line) for line in record.read_text().splitlines()]
            if record.exists()
            else []
        )

    return script, set_output, calls


@pytest.fixture
def snap(tmp_path):
    """A snapshot as ``store`` leaves it: its config and the LMDB file Olaf writes."""
    home = tmp_path / "snap"
    (home / ".olaf" / "db").mkdir(parents=True)
    (home / ".olaf" / "olaf_config.json").write_text("{}")
    (home / ".olaf" / "db" / "data.mdb").write_bytes(b"")
    return home


def test_the_readme_record_maps_to_the_evaluation_fields(olaf):
    [m] = olaf.parse_matches(json.dumps(README_RECORD, indent=2))
    assert m["query_offset_s"] == pytest.approx(1.936)
    assert m["ref_start_s"] == pytest.approx(70.864)
    assert m["ref_key"] == "chuquimamani-condori-call-your-name"
    assert m["match_count"] == 41


def test_concatenated_query_objects_add_each_fragment_offset(olaf):
    # --fragmented and multi-file queries print one pretty-printed object per query.
    out = "\n".join(
        json.dumps(o, indent=2)
        for o in (
            query_object(0.0, match("a", 20, 1.0, 5.0)),
            query_object(30.0, match("b", 30, 2.5, 7.0)),
        )
    )
    assert [(m["ref_key"], m["query_offset_s"]) for m in olaf.parse_matches(out)] == [
        ("a", 1.0),
        ("b", 32.5),
    ]


@pytest.mark.parametrize("out", ["", "not json", '{"query_offset": 0.0}'])
def test_output_without_a_query_object_is_an_error(olaf, out):
    with pytest.raises(olaf.OlafError):
        olaf.parse_matches(out)


@pytest.mark.parametrize(
    ("counts", "floor", "expected"),
    [
        ([("weak", 9), ("strong", 40), ("middle", 20)], 12, "strong"),
        ([("weak", 9), ("noise", 6)], 12, None),  # Phase 1: stray matches sat at 6-10
        ([("edge", 12)], 12, "edge"),  # the floor is inclusive
        ([("weak", 9)], 0, "weak"),
        ([], 12, None),
    ],
)
def test_recognize_keeps_the_strongest_match_at_or_above_the_floor(
    olaf, fake_olaf, snap, counts, floor, expected
):
    script, set_output, _ = fake_olaf
    set_output(json.dumps(query_object(0.0, *(match(p, c) for p, c in counts))))
    recognizer = olaf.OlafRecognizer(snap, olaf_bin=str(script), min_match_count=floor)
    result = recognizer.recognize("clip.wav")
    assert (result or {}).get("ref_key") == expected


def test_the_default_floor_is_twelve(olaf, tmp_path):
    assert olaf.DEFAULT_MIN_MATCH_COUNT == 12
    assert olaf.OlafRecognizer(tmp_path / "snap").min_match_count == 12


def test_recognize_runs_query_under_the_snapshot_home(olaf, fake_olaf, snap):
    script, set_output, calls = fake_olaf
    set_output(json.dumps(README_RECORD))
    result = olaf.OlafRecognizer(snap, olaf_bin=str(script)).recognize("/clips/clip.wav")
    assert calls() == [
        {"argv": ["query", "--format", "json", "/clips/clip.wav"], "home": str(snap)}
    ]
    assert result == {
        "artist": "",
        "song": "chuquimamani-condori-call-your-name",
        "album": "",
        "label": "",
        "source": "local",
        "confidence": 41.0,
        "query_offset_s": pytest.approx(1.936),
        "ref_start_s": pytest.approx(70.864),
        "ref_key": "chuquimamani-condori-call-your-name",
    }


def test_a_lookup_names_the_reference(olaf, fake_olaf, snap):
    script, set_output, _ = fake_olaf
    set_output(json.dumps(README_RECORD))
    tags = {
        "chuquimamani-condori-call-your-name": {
            "artist": "Chuquimamani-Condori",
            "song": "Call Your Name",
            "album": "Edits",
            "label": "self-released",
        }
    }
    result = olaf.OlafRecognizer(snap, olaf_bin=str(script), lookup=tags.get).recognize("clip.wav")
    assert (result["artist"], result["song"], result["album"], result["label"]) == (
        "Chuquimamani-Condori",
        "Call Your Name",
        "Edits",
        "self-released",
    )


def test_store_gives_the_snapshot_home_its_own_olaf_config(olaf, fake_olaf, tmp_path):
    # Without ~/.olaf/olaf_config.json, Olaf falls back to a config beside the binary,
    # which could point db_folder anywhere; the adapter's own config prevents that.
    script, _, _ = fake_olaf
    home = tmp_path / "snap"
    olaf.OlafRecognizer(home, olaf_bin=str(script)).store([("/a.mp3", "id-a")])
    config = json.loads((home / ".olaf" / "olaf_config.json").read_text())
    assert config["db_folder"] == "~/.olaf/db/"
    assert config["cache_folder"] == "~/.olaf/cache/"


def test_store_leaves_an_existing_snapshot_config_alone(olaf, fake_olaf, tmp_path):
    script, _, _ = fake_olaf
    config = tmp_path / "snap" / ".olaf" / "olaf_config.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"db_folder": "~/.olaf/db/", "verbose": true}')
    olaf.OlafRecognizer(tmp_path / "snap", olaf_bin=str(script)).store([("/a.mp3", "id-a")])
    assert config.read_text() == '{"db_folder": "~/.olaf/db/", "verbose": true}'


def test_store_passes_path_identifier_pairs_with_ids(olaf, fake_olaf, tmp_path):
    script, _, calls = fake_olaf
    home = tmp_path / "snap"
    olaf.OlafRecognizer(home, olaf_bin=str(script)).store([("/a.mp3", "id-a"), ("/b.mp3", "id-b")])
    assert calls() == [
        {"argv": ["store", "--with-ids", "/a.mp3", "id-a", "/b.mp3", "id-b"], "home": str(home)}
    ]


def test_store_with_nothing_to_store_runs_nothing(olaf, fake_olaf, tmp_path):
    script, _, calls = fake_olaf
    olaf.OlafRecognizer(tmp_path / "snap", olaf_bin=str(script)).store([])
    assert calls() == []


@pytest.mark.parametrize("action", ["query", "store"])
def test_a_failing_olaf_raises_with_its_stderr(olaf, fake_olaf, snap, action):
    script, set_output, _ = fake_olaf
    set_output(code=1, stderr="error: FileNotFound")
    recognizer = olaf.OlafRecognizer(snap, olaf_bin=str(script))
    with pytest.raises(olaf.OlafError, match="FileNotFound"):
        if action == "query":
            recognizer.recognize("missing.wav")
        else:
            recognizer.store([("missing.mp3", "id")])


def test_the_binary_comes_from_the_setting(fresh_recognizer, tmp_path):
    fresh_recognizer(
        WXDU_SHAZAM_SECRET="not-a-real-secret", STREAM_SLEUTH_OLAF_BIN="/opt/olaf/bin/olaf"
    )
    olaf = importlib.import_module("stream_sleuth.recognizers.olaf")
    assert olaf.OlafRecognizer(tmp_path / "snap").olaf_bin == "/opt/olaf/bin/olaf"


def test_the_binary_defaults_to_olaf_on_path(olaf, tmp_path):
    assert olaf.OlafRecognizer(tmp_path / "snap").olaf_bin == "olaf"


def test_the_adapter_is_a_recognizer(olaf, tmp_path):
    base = importlib.import_module("stream_sleuth.recognizers.base")
    assert isinstance(olaf.OlafRecognizer(tmp_path / "snap"), base.Recognizer)


def test_index_build_stores_the_pairs_under_the_snapshot_home(
    fresh_recognizer, fake_olaf, tmp_path
):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    cli = importlib.import_module("stream_sleuth.cli")
    script, _, calls = fake_olaf
    home = tmp_path / "snap"
    argv = [
        "index",
        "build",
        "--home",
        str(home),
        "--olaf-bin",
        str(script),
        "/a.mp3",
        "id-a",
        "/b.mp3",
        "id-b",
    ]
    assert cli.main(argv) == 0
    assert calls() == [
        {"argv": ["store", "--with-ids", "/a.mp3", "id-a", "/b.mp3", "id-b"], "home": str(home)}
    ]


def test_index_build_refuses_an_unpaired_path(fresh_recognizer, tmp_path, capsys):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    cli = importlib.import_module("stream_sleuth.cli")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["index", "build", "--home", str(tmp_path), "/a.mp3", "id-a", "/b.mp3"])
    assert exit_info.value.code == 2
    assert "pairs" in capsys.readouterr().err


def test_a_match_missing_a_field_is_an_olaf_error(olaf):
    with pytest.raises(olaf.OlafError):
        olaf.parse_matches(json.dumps({"query_offset": 0.0, "matches": [{"match_count": 30}]}))


def test_a_hung_query_times_out_instead_of_stalling_the_loop(olaf, fake_olaf, snap):
    script, set_output, _ = fake_olaf
    script.write_text(
        script.read_text().replace("sys.exit(reply", "import time; time.sleep(5); sys.exit(reply")
    )
    set_output(json.dumps(query_object(0.0)))
    recognizer = olaf.OlafRecognizer(snap, olaf_bin=str(script), query_timeout_s=0.5)
    with pytest.raises(olaf.OlafError, match="timed out"):
        recognizer.recognize("clip.wav")


def test_a_failure_reported_only_on_stdout_still_names_its_reason(olaf, fake_olaf, tmp_path):
    # Olaf prints its argument-parsing errors to stdout, leaving stderr empty.
    script, set_output, _ = fake_olaf
    set_output(code=2, stdout="Unknown option '-x.wav' for 'olaf store'")
    with pytest.raises(olaf.OlafError, match="Unknown option"):
        olaf.OlafRecognizer(tmp_path / "snap", olaf_bin=str(script)).store([("-x.wav", "id")])


@pytest.mark.parametrize(
    "state",
    ["absent", "empty", "failed store", "no config"],
)
def test_querying_a_snapshot_without_an_index_is_an_error_and_creates_nothing(
    olaf, fake_olaf, tmp_path, state
):
    # A mistyped snapshot path must fail, not score as an empty index's 0% recall.
    script, set_output, calls = fake_olaf
    set_output(json.dumps(query_object(0.0)))
    home = tmp_path / "snap"
    if state == "empty":
        home.mkdir()
    elif state == "failed store":  # Olaf leaves its db folder but no data.mdb
        (home / ".olaf" / "db").mkdir(parents=True)
        (home / ".olaf" / "olaf_config.json").write_text("{}")
    elif state == "no config":  # Olaf would fall back to a config beside its binary
        (home / ".olaf" / "db").mkdir(parents=True)
        (home / ".olaf" / "db" / "data.mdb").write_bytes(b"")
    before = sorted(tmp_path.rglob("*"))
    with pytest.raises(olaf.OlafError, match="no Olaf index"):
        olaf.OlafRecognizer(home, olaf_bin=str(script)).recognize("clip.wav")
    assert calls() == []
    assert sorted(tmp_path.rglob("*")) == before


_MISCASED_HOME = Path(str(Path.home()).swapcase())
_FIRMLINKED_HOME = Path("/System/Volumes/Data" + str(Path.home()))

REFUSED_HOMES = [
    pytest.param("snap", "absolute", id="relative"),
    pytest.param(".", "absolute", id="dot"),
    pytest.param("~/snap", "absolute", id="literal-tilde"),
    pytest.param(str(Path.home()), "home directory", id="real-home"),
    pytest.param(
        str(_MISCASED_HOME),
        "home directory",
        id="miscased-home",
        marks=pytest.mark.skipif(not _MISCASED_HOME.exists(), reason="case-sensitive filesystem"),
    ),
    pytest.param(
        str(_FIRMLINKED_HOME),
        "home directory",
        id="firmlinked-home",
        marks=pytest.mark.skipif(not _FIRMLINKED_HOME.exists(), reason="no macOS data firmlink"),
    ),
    pytest.param(str(CHECKOUT), "inside the checkout", id="checkout"),
    pytest.param(str(CHECKOUT / "snap"), "inside the checkout", id="in-checkout"),
    pytest.param(str(CHECKOUT / "tests" / ".." / "snap"), "inside the checkout", id="dotdot"),
]


@pytest.mark.parametrize(("home", "reason"), REFUSED_HOMES)
def test_a_snapshot_outside_the_data_area_is_refused(olaf, home, reason):
    with pytest.raises(olaf.OlafError, match=reason):
        olaf.OlafRecognizer(home)


def test_an_existing_snapshot_is_accepted_when_home_does_not_exist(olaf, tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "no-such-home"))
    assert olaf.OlafRecognizer(snap).home == snap


def test_a_snapshot_reached_through_a_symlink_into_the_checkout_is_refused(olaf, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(CHECKOUT)
    with pytest.raises(olaf.OlafError, match="inside the checkout") as exc_info:
        olaf.OlafRecognizer(link / "snap")
    # By name: the fixture imports the package afresh, so the class is not the test module's.
    assert type(exc_info.value.__cause__).__name__ == "DataPathError"
    assert not (CHECKOUT / "snap").exists()


@pytest.mark.parametrize(("home", "reason"), REFUSED_HOMES)
def test_index_build_refuses_the_snapshot_and_creates_nothing(
    fresh_recognizer, fake_olaf, tmp_path, monkeypatch, capsys, home, reason
):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    cli = importlib.import_module("stream_sleuth.cli")
    # Even if the refusal regresses, nothing may be written into the checkout or ~.
    monkeypatch.setattr(cli.OlafRecognizer, "store", lambda *_: pytest.fail("stored"))
    script, _, calls = fake_olaf
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    argv = ["index", "build", "--home", home, "--olaf-bin", str(script), "/a.mp3", "id-a"]
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
    assert reason in capsys.readouterr().err
    assert calls() == []
    assert sorted(tmp_path.rglob("*")) == before
    assert not (CHECKOUT / "snap").exists()


def test_a_large_store_is_split_so_no_argv_grows_past_the_limit(olaf, fake_olaf, tmp_path):
    script, _, calls = fake_olaf
    pairs = [(f"/pool/{i}.mp3", f"id-{i}") for i in range(2 * olaf.STORE_BATCH + 1)]
    olaf.OlafRecognizer(tmp_path / "snap", olaf_bin=str(script)).store(pairs)
    batches = [c["argv"] for c in calls()]
    assert [len(b) for b in batches] == [
        2 + 2 * olaf.STORE_BATCH,
        2 + 2 * olaf.STORE_BATCH,
        2 + 2,
    ]
    assert all(b[:2] == ["store", "--with-ids"] for b in batches)
    assert [a for b in batches for a in b[2:]] == [a for pair in pairs for a in pair]


@pytest.mark.parametrize("action", ["query", "store"])
def test_a_missing_binary_is_an_olaf_error(olaf, snap, tmp_path, action):
    recognizer = olaf.OlafRecognizer(snap, olaf_bin=str(tmp_path / "no-such-olaf"))
    with pytest.raises(olaf.OlafError, match="cannot run"):
        if action == "query":
            recognizer.recognize("clip.wav")
        else:
            recognizer.store([("/a.mp3", "id")])


def test_a_timed_out_query_kills_the_decoder_olaf_started(olaf, fake_olaf, snap, tmp_path):
    # Olaf decodes with an ffmpeg child; a timeout must not leave it running.
    script, set_output, _ = fake_olaf
    child_pid = tmp_path / "child.pid"
    script.write_text(
        script.read_text().replace(
            "sys.exit(reply",
            "import subprocess, time\n"
            "child = subprocess.Popen(['sleep', '30'])\n"
            f"open({str(child_pid)!r}, 'w').write(str(child.pid))\n"
            "time.sleep(30); sys.exit(reply",
        )
    )
    set_output(json.dumps(query_object(0.0)))
    recognizer = olaf.OlafRecognizer(snap, olaf_bin=str(script), query_timeout_s=1.0)
    with pytest.raises(olaf.OlafError, match="timed out"):
        recognizer.recognize("clip.wav")
    pid = int(child_pid.read_text())
    for _ in range(50):
        if not _alive(pid):
            break
        time.sleep(0.05)
    assert not _alive(pid)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child of the fake (reparented to init) is reaped promptly; a zombie
    # would still answer kill(0), so check its state too.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(state.stdout.strip()) and not state.stdout.strip().startswith("Z")


def hang(script):
    """Make the fake ``olaf`` sleep, so a query outlives its timeout."""
    script.write_text(
        script.read_text().replace("sys.exit(reply", "import time; time.sleep(5); sys.exit(reply")
    )


@pytest.mark.parametrize("error", [PermissionError, ProcessLookupError])
def test_a_kill_that_races_the_childs_exit_still_raises_olaf_error(
    olaf, fake_olaf, snap, monkeypatch, error
):
    # On macOS killpg raises when Olaf exits just as the timeout fires.
    script, set_output, _ = fake_olaf
    hang(script)
    set_output(json.dumps(query_object(0.0)))

    def refuse(*_):
        raise error

    monkeypatch.setattr(olaf.os, "killpg", refuse)
    recognizer = olaf.OlafRecognizer(snap, olaf_bin=str(script), query_timeout_s=0.3)
    with pytest.raises(olaf.OlafError, match="timed out"):
        recognizer.recognize("clip.wav")


def test_an_interrupted_query_kills_the_process_group_before_re_raising(
    olaf, fake_olaf, snap, monkeypatch
):
    script, set_output, _ = fake_olaf
    hang(script)
    set_output(json.dumps(query_object(0.0)))
    killed = []
    real_killpg, real_communicate = olaf.os.killpg, subprocess.Popen.communicate

    def killpg(pid, sig):
        killed.append((pid, sig))
        real_killpg(pid, sig)

    def interrupted(self, *args, **kwargs):
        if killed:
            return real_communicate(self, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(olaf.os, "killpg", killpg)
    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        olaf.OlafRecognizer(snap, olaf_bin=str(script)).recognize("clip.wav")
    assert [sig for _, sig in killed] == [signal.SIGKILL]


@pytest.mark.parametrize("action", ["query", "store"])
def test_olaf_never_inherits_the_terminal_as_stdin(olaf, fake_olaf, snap, monkeypatch, action):
    # Olaf's ffmpeg child reads stdin unless told not to, and can leave echo off.
    script, _, _ = fake_olaf
    seen = []
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        seen.append(kwargs.get("stdin"))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(olaf.subprocess, "Popen", spy)
    recognizer = olaf.OlafRecognizer(snap, olaf_bin=str(script))
    try:
        if action == "query":
            recognizer.recognize("clip.wav")
        else:
            recognizer.store([("/a.mp3", "id")])
    except olaf.OlafError:
        pass  # the fake prints no query result; only how it was started matters
    assert seen == [subprocess.DEVNULL]


def test_a_lookup_missing_a_key_falls_back_for_that_key(olaf, fake_olaf, snap):
    script, set_output, _ = fake_olaf
    set_output(json.dumps(README_RECORD))
    tags = {"chuquimamani-condori-call-your-name": {"artist": "Chuquimamani-Condori"}}
    result = olaf.OlafRecognizer(snap, olaf_bin=str(script), lookup=tags.get).recognize("c.wav")
    assert (result["artist"], result["song"], result["album"], result["label"]) == (
        "Chuquimamani-Condori",
        "chuquimamani-condori-call-your-name",
        "",
        "",
    )


def test_index_build_reports_an_olaf_failure_in_one_line(
    fresh_recognizer, fake_olaf, tmp_path, capsys
):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    cli = importlib.import_module("stream_sleuth.cli")
    script, set_output, _ = fake_olaf
    set_output(code=1, stderr="File is not an audio file")
    argv = ["index", "build", "--home", str(tmp_path / "snap"), "--olaf-bin", str(script)]
    with pytest.raises(SystemExit) as exit_info:
        cli.main([*argv, "/track.aiff", "id"])
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "File is not an audio file" in err
    assert "Traceback" not in err


def test_store_reports_an_uncreatable_snapshot_as_olaf_error(olaf, fake_olaf, tmp_path):
    script, _, calls = fake_olaf
    (tmp_path / "file").write_text("")
    recognizer = olaf.OlafRecognizer(tmp_path / "file" / "snap", olaf_bin=str(script))
    with pytest.raises(olaf.OlafError, match="cannot create"):
        recognizer.store([("/a.mp3", "id")])
    assert calls() == []


def test_store_reports_an_unsearchable_parent_as_olaf_error(olaf, fake_olaf, tmp_path):
    script, _, calls = fake_olaf
    locked = tmp_path / "locked"
    locked.mkdir()
    recognizer = olaf.OlafRecognizer(locked / "snap", olaf_bin=str(script))
    locked.chmod(0)
    try:
        with pytest.raises(olaf.OlafError, match="cannot create"):
            recognizer.store([("/a.mp3", "id")])
    finally:
        locked.chmod(stat.S_IRWXU)
    assert calls() == []


def test_index_build_reports_an_uncreatable_snapshot_in_one_line(
    fresh_recognizer, fake_olaf, tmp_path, capsys
):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    cli = importlib.import_module("stream_sleuth.cli")
    script, _, _ = fake_olaf
    (tmp_path / "file").write_text("")
    argv = ["index", "build", "--home", str(tmp_path / "file" / "snap"), "--olaf-bin", str(script)]
    with pytest.raises(SystemExit) as exit_info:
        cli.main([*argv, "/a.mp3", "id"])
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "cannot create" in err
    assert "Traceback" not in err


def test_the_usage_line_is_the_invocation_that_works(fresh_recognizer, capsys):
    fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    cli = importlib.import_module("stream_sleuth.cli")
    with pytest.raises(SystemExit):
        cli.main(["index", "build", "--help"])
    assert "usage: python -m stream_sleuth.cli index build" in capsys.readouterr().out
