"""Cutting real clips from a synthetic tone hour with a real ffmpeg."""

from __future__ import annotations

import json
import subprocess

import pytest

from evaluation import clips

pytestmark = pytest.mark.ffmpeg

HOUR_S = 60
MP3_FRAME_S = 1152 / 44100


@pytest.fixture(scope="module")
def tone_hour(tmp_path_factory):
    path = tmp_path_factory.mktemp("archive") / "tone.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={HOUR_S}:sample_rate=44100",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "0",
            str(path),
        ],
        check=True,
    )
    return path


def _probe(path):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=channels,sample_rate,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(out.stdout)
    return info["format"], info["streams"][0]


def _decoded_duration(path, tmp_path):
    # Older ffprobe builds count the encoder's priming and frame padding in an
    # MP3's container duration; decoding honors the gapless header, and decoded
    # audio is what a recognizer hears.
    wav = tmp_path / "decoded.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), str(wav)], check=True)
    fmt, _ = _probe(wav)
    return float(fmt["duration"])


@pytest.mark.parametrize(("length_s", "profile"), [(6, "128k"), (12, "128k"), (20, "320k")])
def test_cut_has_the_length_and_constant_bitrate(tone_hour, tmp_path, length_s, profile):
    address = clips.ClipAddress("tone.mp3", 30, length_s, profile)
    work = tmp_path / "work"
    work.mkdir()
    with clips.cut(address, tone_hour, work) as clip:
        _, stream = _probe(clip)
        duration = _decoded_duration(clip, tmp_path)
    assert abs(duration - length_s) <= MP3_FRAME_S
    assert int(stream["bit_rate"]) == int(profile.removesuffix("k")) * 1000


def test_cut_decodes_to_a_mono_16khz_wav(tone_hour, tmp_path):
    with clips.cut(clips.ClipAddress("tone.mp3", 15, 12), tone_hour, tmp_path, wav=True) as clip:
        fmt, stream = _probe(clip)
    assert (stream["channels"], stream["sample_rate"]) == (1, "16000")
    assert abs(float(fmt["duration"]) - 12) <= MP3_FRAME_S


def test_cut_deletes_the_clip_after_exit_and_after_an_exception(tone_hour, tmp_path):
    with clips.cut(clips.ClipAddress("tone.mp3", 0, 6), tone_hour, tmp_path, wav=True) as clip:
        assert clip.stat().st_size > 0
    assert list(tmp_path.iterdir()) == []
    with (
        pytest.raises(RuntimeError),
        clips.cut(clips.ClipAddress("tone.mp3", 0, 6), tone_hour, tmp_path),
    ):
        raise RuntimeError("recognizer failed")
    assert list(tmp_path.iterdir()) == []
