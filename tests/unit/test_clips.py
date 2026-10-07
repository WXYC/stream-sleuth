"""Clip addresses, the grid, and the cut's ffmpeg arguments, against the recording ffmpeg stub."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from evaluation import clips
from stream_sleuth.paths import CHECKOUT, DataPathError
from tests.ffmpeg_stub import FfmpegStub

HOUR = "2026/08/12/202608121600.mp3"


@pytest.mark.parametrize(
    ("length_s", "hour_s", "points", "last_offset"),
    [
        (6, 3600, 240, 3585),
        (12, 3600, 240, 3585),
        (20, 3600, 239, 3570),
        (20, 3600.0, 239, 3570),
        (12, 3596.5, 239, 3570),
        (6, 3596.5, 240, 3585),
        (20, 59.97, 3, 30),
    ],
)
def test_grid_holds_exactly_the_offsets_whose_clip_ends_by_the_hours_end(
    length_s, hour_s, points, last_offset
):
    grid = clips.grid(HOUR, length_s, hour_s=hour_s)
    assert len(grid) == points
    assert grid[0].offset_s == 0
    assert grid[-1].offset_s == last_offset
    assert all(a.offset_s + length_s <= hour_s for a in grid)
    assert grid[-1].offset_s + clips.GRID_S + length_s > hour_s
    assert {a.offset_s % clips.GRID_S for a in grid} == {0}
    assert {(a.hour_key, a.length_s, a.profile) for a in grid} == {(HOUR, length_s, "128k")}


def test_grid_takes_the_hour_length_and_profile():
    grid = clips.grid(HOUR, 12, profile="320k", hour_s=60)
    assert [a.offset_s for a in grid] == [0, 15, 30, 45]
    assert {a.profile for a in grid} == {"320k"}


def test_grid_of_an_hour_shorter_than_one_clip_is_empty():
    assert clips.grid(HOUR, 20, hour_s=19.9) == []


@pytest.mark.parametrize(
    ("address", "key"),
    [
        (clips.ClipAddress(HOUR, 0, 12), f"{HOUR}#0+12@128k"),
        (clips.ClipAddress(HOUR, 45, 12), f"{HOUR}#45+12@128k"),
        (clips.ClipAddress(HOUR, 3585, 6, "320k"), f"{HOUR}#3585+6@320k"),
        (clips.ClipAddress("hours/Jessica Pratt at night.mp3", 15, 20), "hours/Jessica Pratt at night.mp3#15+20@128k"),
        (clips.ClipAddress("Hermanos Gutiérrez.mp3", 30, 6), "Hermanos Gutiérrez.mp3#30+6@128k"),
    ],
)  # fmt: skip
def test_one_address_has_one_key_and_parse_inverts_it(address, key):
    assert address.key == key
    assert str(address) == key
    assert clips.ClipAddress.parse(key) == address
    assert clips.ClipAddress.parse(key).key == key


@pytest.mark.parametrize("hour_s", [3600, 3596.5, 59.97])
@pytest.mark.parametrize("length_s", clips.CAPTURE_LENGTHS_S)
@pytest.mark.parametrize("profile", clips.PROFILES)
def test_every_grid_key_round_trips(hour_s, length_s, profile):
    for address in clips.grid(HOUR, length_s, profile, hour_s):
        assert clips.ClipAddress.parse(address.key) == address
        assert clips.ClipAddress.parse(address.key).key == address.key


@pytest.mark.parametrize(
    ("hour_key", "offset_s", "length_s", "profile", "error"),
    [
        pytest.param(HOUR, 7, 12, "128k", ValueError, id="off-grid"),
        pytest.param(HOUR, -15, 12, "128k", ValueError, id="negative"),
        pytest.param(HOUR, 0, 10, "128k", ValueError, id="unknown-length"),
        pytest.param(HOUR, 0, 12, "64k", ValueError, id="unknown-profile"),
        pytest.param(HOUR, 15.0, 12, "128k", TypeError, id="float-offset"),
        pytest.param(HOUR, True, 12, "128k", TypeError, id="bool-offset"),
        pytest.param(HOUR, "15", 12, "128k", TypeError, id="str-offset"),
        pytest.param(HOUR, 15, 12.0, "128k", TypeError, id="float-length"),
        pytest.param(b"h.mp3", 15, 12, "128k", TypeError, id="bytes-hour"),
        pytest.param("", 15, 12, "128k", ValueError, id="empty-hour"),
        pytest.param("a\nb.mp3", 15, 12, "128k", ValueError, id="newline-hour"),
        pytest.param("a\tb.mp3", 15, 12, "128k", ValueError, id="tab-hour"),
        pytest.param("a#b.mp3", 15, 12, "128k", ValueError, id="hash-hour"),
        pytest.param("a+b.mp3", 15, 12, "128k", ValueError, id="plus-hour"),
        pytest.param("a@b.mp3", 15, 12, "128k", ValueError, id="at-hour"),
    ],
)
def test_address_rejects_what_has_no_canonical_key(hour_key, offset_s, length_s, profile, error):
    with pytest.raises(error):
        clips.ClipAddress(hour_key, offset_s, length_s, profile)


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("no-separators.mp3", id="no-separators"),
        pytest.param(f"{HOUR}#x+12@128k", id="letter-offset"),
        pytest.param(f"{HOUR}#0+12", id="no-profile"),
        pytest.param(f"{HOUR}#015+12@128k", id="leading-zero-offset"),
        pytest.param(f"{HOUR}#00+12@128k", id="double-zero-offset"),
        pytest.param(f"{HOUR}#15+012@128k", id="leading-zero-length"),
        pytest.param(f"{HOUR}#15.0+12@128k", id="decimal-offset"),
        pytest.param(f"{HOUR}#+15+12@128k", id="signed-offset"),
        pytest.param(f"{HOUR}#-15+12@128k", id="negative-offset"),
        pytest.param(f"{HOUR}# 15+12@128k", id="spaced-offset"),
        pytest.param(f"{HOUR}#١٥+12@128k", id="arabic-indic-digits"),
        pytest.param(f"{HOUR}#１５+12@128k", id="fullwidth-digits"),
        pytest.param(f"{HOUR}#15+12@128k\n", id="trailing-newline"),
        pytest.param(f"{HOUR}#15+12@128K", id="upper-profile"),
        pytest.param(f"{HOUR}#15+12@64k", id="unknown-profile"),
        pytest.param(f"{HOUR}#7+12@128k", id="off-grid"),
        pytest.param(f"{HOUR}#15+10@128k", id="unknown-length"),
        pytest.param("#15+12@128k", id="empty-hour"),
        pytest.param("a#b.mp3#15+12@128k", id="hash-in-hour"),
        pytest.param("a\nb.mp3#15+12@128k", id="newline-in-hour"),
    ],
)
def test_parse_rejects_every_key_that_is_not_canonical(key):
    with pytest.raises(ValueError):
        clips.ClipAddress.parse(key)


@pytest.fixture
def stub(tmp_path, monkeypatch):
    root = tmp_path / "stub"
    root.mkdir()
    stub = FfmpegStub(root).install()
    monkeypatch.setenv("PATH", f"{stub.bin_dir}{os.pathsep}{os.environ['PATH']}")
    return stub


@pytest.fixture
def hour(tmp_path, monkeypatch):
    """An hour file the stub never reads, measured as exactly 3,600 s."""
    path = tmp_path / "hour.mp3"
    path.write_bytes(b"")
    monkeypatch.setattr(clips, "hour_duration", lambda p: 3600.0)
    return path


@pytest.fixture
def work(tmp_path):
    path = tmp_path / "work"
    path.mkdir()
    return path


def test_cut_seeks_takes_the_length_and_encodes_cbr_mp3(stub, hour, work):
    with clips.cut(clips.ClipAddress(HOUR, 45, 12, "320k"), hour, work) as clip:
        assert clip.suffix == ".mp3"
        assert clip.exists()
    (argv,) = stub.calls()
    assert argv[argv.index("-ss") + 1] == "45"
    assert argv.index("-ss") < argv.index("-i")
    assert argv[argv.index("-i") + 1] == f"file:{hour}"
    assert argv[argv.index("-t") + 1] == "12"
    assert argv[argv.index("-c:a") + 1] == "libmp3lame"
    assert argv[argv.index("-b:a") + 1] == "320k"


def test_cut_decodes_back_to_a_mono_16khz_wav_when_asked(stub, hour, work):
    with clips.cut(clips.ClipAddress(HOUR, 0, 6), hour, work, wav=True) as clip:
        assert clip.suffix == ".wav"
    encode, decode = stub.calls()
    assert encode[-1].endswith(".mp3")
    assert decode[decode.index("-i") + 1] == f"file:{encode[-1]}"
    assert decode[decode.index("-ac") + 1] == "1"
    assert decode[decode.index("-ar") + 1] == "16000"


@pytest.mark.parametrize("wav", [False, True])
def test_every_ffmpeg_call_reads_no_stdin(stub, hour, work, monkeypatch, wav):
    seen = []
    real_run = subprocess.run

    def recording_run(*args, **kwargs):
        seen.append(kwargs.get("stdin"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(clips.subprocess, "run", recording_run)
    with clips.cut(clips.ClipAddress(HOUR, 0, 6), hour, work, wav=wav):
        pass
    assert seen == [subprocess.DEVNULL] * (2 if wav else 1)


def test_cut_leaves_nothing_behind_after_exit_or_exception(stub, hour, work):
    with clips.cut(clips.ClipAddress(HOUR, 0, 12), hour, work, wav=True):
        pass
    assert list(work.iterdir()) == []
    with pytest.raises(RuntimeError), clips.cut(clips.ClipAddress(HOUR, 0, 12), hour, work):
        raise RuntimeError("recognizer failed")
    assert list(work.iterdir()) == []


@pytest.mark.parametrize(
    ("wav", "failing_call"),
    [pytest.param(False, 1, id="encode"), pytest.param(True, 1, id="encode-before-decode"),
     pytest.param(True, 2, id="decode")],
)  # fmt: skip
def test_cut_raises_and_cleans_up_when_ffmpeg_fails(stub, hour, work, wav, failing_call):
    stub.fail(failing_call)
    with (
        pytest.raises(clips.ClipError) as raised,
        clips.cut(clips.ClipAddress(HOUR, 0, 12), hour, work, wav=wav),
    ):
        pass
    assert len(stub.calls()) == failing_call
    assert "simulated failure" in str(raised.value)
    assert "b'" not in str(raised.value)
    assert list(work.iterdir()) == []


@pytest.mark.parametrize(
    ("duration", "offset_s", "length_s"),
    [
        pytest.param(3600.0, 3585, 20, id="crosses-the-end"),
        pytest.param(3600.0, 3600, 6, id="starts-at-the-end"),
        pytest.param(3600.0, 99990, 6, id="starts-past-the-end"),
        pytest.param(3596.5, 3585, 12, id="crosses-a-short-hour"),
        pytest.param(60.0, 45, 20, id="crosses-a-minute"),
        pytest.param(0.0, 0, 6, id="empty-hour"),
    ],
)
def test_cut_refuses_a_clip_past_the_hours_end_before_running_ffmpeg(
    stub, tmp_path, work, monkeypatch, duration, offset_s, length_s
):
    hour = tmp_path / "short.mp3"
    hour.write_bytes(b"")
    monkeypatch.setattr(clips, "hour_duration", lambda p: duration)
    with (
        pytest.raises(clips.ClipError, match="past the end"),
        clips.cut(clips.ClipAddress(HOUR, offset_s, length_s), hour, work),
    ):
        pytest.fail("cut yielded a clip past the hour's end")
    assert stub.calls() == []
    assert list(work.iterdir()) == []


@pytest.mark.parametrize(
    ("duration", "offset_s", "length_s"),
    [(3590.0, 3570, 20), (3600.0, 3570, 20), (3597.0, 3585, 12), (3596.5, 3585, 6), (60.0, 45, 12)],
)
def test_cut_accepts_a_clip_that_ends_by_the_hours_end(
    stub, tmp_path, work, monkeypatch, duration, offset_s, length_s
):
    hour = tmp_path / "hour.mp3"
    hour.write_bytes(b"")
    monkeypatch.setattr(clips, "hour_duration", lambda p: duration)
    with clips.cut(clips.ClipAddress(HOUR, offset_s, length_s), hour, work) as clip:
        assert clip.exists()


@pytest.mark.parametrize(
    "work_dir",
    [CHECKOUT, CHECKOUT / "data" / "clips", Path("clips")],
    ids=["checkout", "in-checkout", "relative"],
)
def test_cut_refuses_a_checkout_or_relative_work_dir_before_creating_anything(
    stub, hour, monkeypatch, work_dir
):
    monkeypatch.chdir(CHECKOUT)
    before = set(CHECKOUT.iterdir())
    with (
        pytest.raises(DataPathError),
        clips.cut(clips.ClipAddress(HOUR, 0, 6), hour, work_dir),
    ):
        pytest.fail("cut yielded a clip in the checkout")
    assert set(CHECKOUT.iterdir()) == before
    assert stub.calls() == []


def _progress(seconds: str) -> bytes:
    return (
        f"out_time_us={seconds}\nprogress=continue\nout_time_us={seconds}\nprogress=end\n".encode()
    )


def test_hour_duration_decodes_once_per_version_of_the_file(tmp_path, monkeypatch):
    path = tmp_path / "hour.mp3"
    path.write_bytes(b"first")
    calls = []

    def fake_ffmpeg(*args):
        calls.append(args)
        return _progress(str(3_599_973_878 if len(calls) == 1 else 3_600_000_000))

    monkeypatch.setattr(clips, "_ffmpeg", fake_ffmpeg)
    assert clips.hour_duration(path) == 3599.973878
    assert clips.hour_duration(path) == 3599.973878
    assert len(calls) == 1
    assert f"file:{path.resolve()}" in calls[0]
    path.write_bytes(b"second version")
    assert clips.hour_duration(path) == 3600.0
    assert len(calls) == 2


@pytest.mark.parametrize("stdout", [b"", _progress("N/A"), b"progress=end\n"])
def test_hour_duration_raises_when_ffmpeg_reports_no_audio(tmp_path, monkeypatch, stdout):
    path = tmp_path / "hour.mp3"
    path.write_bytes(b"x")
    monkeypatch.setattr(clips, "_ffmpeg", lambda *args: stdout)
    with pytest.raises(clips.ClipError):
        clips.hour_duration(path)


def test_hour_duration_raises_for_a_missing_hour(tmp_path):
    with pytest.raises(clips.ClipError):
        clips.hour_duration(tmp_path / "absent.mp3")
