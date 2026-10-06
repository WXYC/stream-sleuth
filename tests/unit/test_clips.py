"""Clip addresses, the grid, and the cut's ffmpeg arguments, against the recording ffmpeg stub."""

from __future__ import annotations

import os

import pytest

from evaluation import clips
from tests.ffmpeg_stub import FfmpegStub

HOUR = "2026/08/12/202608121600.mp3"


@pytest.mark.parametrize(
    ("length_s", "points", "last_offset"),
    [(6, 240, 3585), (12, 240, 3585), (20, 239, 3570)],
)
def test_grid_fits_every_clip_inside_the_hour(length_s, points, last_offset):
    grid = clips.grid(HOUR, length_s)
    assert len(grid) == points
    assert grid[0].offset_s == 0
    assert grid[-1].offset_s == last_offset
    assert grid[-1].offset_s + length_s <= 3600
    assert {a.offset_s % clips.GRID_S for a in grid} == {0}
    assert {(a.hour_key, a.length_s, a.profile) for a in grid} == {(HOUR, length_s, "128k")}


def test_grid_takes_the_hour_length_and_profile():
    grid = clips.grid(HOUR, 12, profile="320k", hour_s=60)
    assert [a.offset_s for a in grid] == [0, 15, 30, 45]
    assert {a.profile for a in grid} == {"320k"}


@pytest.mark.parametrize(
    "address",
    [
        clips.ClipAddress(HOUR, 0, 12),
        clips.ClipAddress(HOUR, 3585, 6, "320k"),
        clips.ClipAddress("hours/Jessica Pratt @ night#2.mp3", 15, 20),
    ],
)
def test_address_round_trips_through_its_string_key(address):
    assert clips.ClipAddress.parse(str(address)) == address


def test_address_string_key_is_stable():
    assert str(clips.ClipAddress(HOUR, 45, 12)) == f"{HOUR}#45+12@128k"


@pytest.mark.parametrize(
    ("offset_s", "length_s", "profile"),
    [(7, 12, "128k"), (-15, 12, "128k"), (0, 10, "128k"), (0, 12, "64k")],
)
def test_address_rejects_off_grid_offsets_unknown_lengths_and_profiles(offset_s, length_s, profile):
    with pytest.raises(ValueError):
        clips.ClipAddress(HOUR, offset_s, length_s, profile)


@pytest.mark.parametrize("key", ["no-separators.mp3", f"{HOUR}#x+12@128k", f"{HOUR}#0+12"])
def test_parse_rejects_malformed_keys(key):
    with pytest.raises(ValueError):
        clips.ClipAddress.parse(key)


@pytest.fixture
def stub(tmp_path, monkeypatch):
    root = tmp_path / "stub"
    root.mkdir()
    stub = FfmpegStub(root).install()
    monkeypatch.setenv("PATH", f"{stub.bin_dir}{os.pathsep}{os.environ['PATH']}")
    return stub


def test_cut_seeks_takes_the_length_and_encodes_cbr_mp3(stub, tmp_path):
    hour = tmp_path / "hour.mp3"
    hour.write_bytes(b"")
    work = tmp_path / "work"
    work.mkdir()
    with clips.cut(clips.ClipAddress(HOUR, 45, 12, "320k"), hour, work) as clip:
        assert clip.suffix == ".mp3"
        assert clip.exists()
    (argv,) = stub.calls()
    assert argv[argv.index("-ss") + 1] == "45"
    assert argv.index("-ss") < argv.index("-i")
    assert argv[argv.index("-i") + 1] == str(hour)
    assert argv[argv.index("-t") + 1] == "12"
    assert argv[argv.index("-c:a") + 1] == "libmp3lame"
    assert argv[argv.index("-b:a") + 1] == "320k"


def test_cut_decodes_back_to_a_mono_16khz_wav_when_asked(stub, tmp_path):
    hour = tmp_path / "hour.mp3"
    hour.write_bytes(b"")
    work = tmp_path / "work"
    work.mkdir()
    with clips.cut(clips.ClipAddress(HOUR, 0, 6), hour, work, wav=True) as clip:
        assert clip.suffix == ".wav"
    encode, decode = stub.calls()
    assert encode[-1].endswith(".mp3")
    assert decode[decode.index("-i") + 1] == encode[-1]
    assert decode[decode.index("-ac") + 1] == "1"
    assert decode[decode.index("-ar") + 1] == "16000"


def test_cut_leaves_nothing_behind_after_exit_or_exception(stub, tmp_path):
    hour = tmp_path / "hour.mp3"
    hour.write_bytes(b"")
    work = tmp_path / "work"
    work.mkdir()
    with clips.cut(clips.ClipAddress(HOUR, 0, 12), hour, work, wav=True):
        pass
    assert list(work.iterdir()) == []
    with pytest.raises(RuntimeError), clips.cut(clips.ClipAddress(HOUR, 0, 12), hour, work):
        raise RuntimeError("recognizer failed")
    assert list(work.iterdir()) == []


def test_cut_raises_and_cleans_up_when_ffmpeg_fails(stub, tmp_path):
    hour = tmp_path / "hour.mp3"
    hour.write_bytes(b"")
    work = tmp_path / "work"
    work.mkdir()
    stub.fail(1)
    with pytest.raises(clips.ClipError), clips.cut(clips.ClipAddress(HOUR, 0, 12), hour, work):
        pass
    assert list(work.iterdir()) == []
