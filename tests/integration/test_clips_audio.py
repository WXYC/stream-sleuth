"""Cutting real clips from a synthetic hour with a real ffmpeg.

The synthetic "hour" is 60 s long: 30 s of silence, then 30 s of a 440 Hz tone,
so where a clip's tone starts shows where the cut actually seeked to.
"""

from __future__ import annotations

import json
import subprocess
from array import array

import pytest

from evaluation import clips

pytestmark = pytest.mark.ffmpeg

HOUR_S = 60
STEP_S = 30
MP3_FRAME_S = 1152 / 44100
RATE = 44100
# The tone's peak is about 4,100; silence through the encoder stays near 0.
LOUD = 1000


@pytest.fixture(scope="module", params=[["-q:a", "0"], ["-b:a", "128k"]], ids=["vbr", "cbr"])
def step_hour(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("archive") / "step.mp3"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-f", "lavfi", "-i", f"anullsrc=r={RATE}:cl=mono:d={STEP_S}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={HOUR_S - STEP_S}:sample_rate={RATE}",
            "-filter_complex", "[0][1]concat=n=2:v=0:a=1",
            "-c:a", "libmp3lame", *request.param, str(path),
        ],
        check=True,
    )  # fmt: skip
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


def _samples(path):
    # Decoding honors the MP3 gapless header (older ffprobe builds count the
    # encoder's priming and padding in the container duration), and decoded
    # audio is what a recognizer hears.
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(RATE),
         "-f", "s16le", "-"],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return array("h", out.stdout)


def _onset_s(samples):
    """Where the tone starts in a clip, in seconds, or None if the clip is silent."""
    loud = next((i for i, s in enumerate(samples) if abs(s) > LOUD), None)
    return None if loud is None else loud / RATE


def test_hour_duration_is_the_decoded_length(step_hour):
    assert abs(clips.hour_duration(step_hour) - HOUR_S) <= MP3_FRAME_S


@pytest.mark.parametrize(("length_s", "profile"), [(6, "128k"), (12, "128k"), (20, "320k")])
def test_cut_has_the_length_and_constant_bitrate(step_hour, tmp_path, length_s, profile):
    address = clips.ClipAddress("step.mp3", 30, length_s, profile)
    with clips.cut(address, step_hour, tmp_path) as clip:
        _, stream = _probe(clip)
        duration = len(_samples(clip)) / RATE
    assert abs(duration - length_s) <= MP3_FRAME_S
    assert int(stream["bit_rate"]) == int(profile.removesuffix("k")) * 1000


@pytest.mark.parametrize(
    ("offset_s", "length_s", "onset_s"),
    [
        pytest.param(0, 20, None, id="all-silence"),
        pytest.param(15, 12, None, id="silence-up-to-the-step"),
        pytest.param(15, 20, STEP_S - 15, id="crosses-the-step"),
        pytest.param(30, 6, 0.0, id="starts-at-the-step"),
    ],
)
def test_cut_seeks_to_the_address_offset(step_hour, tmp_path, offset_s, length_s, onset_s):
    with clips.cut(clips.ClipAddress("step.mp3", offset_s, length_s), step_hour, tmp_path) as clip:
        onset = _onset_s(_samples(clip))
    if onset_s is None:
        assert onset is None
    else:
        assert onset is not None
        assert abs(onset - onset_s) <= MP3_FRAME_S


def test_cut_decodes_to_a_mono_16khz_wav(step_hour, tmp_path):
    with clips.cut(clips.ClipAddress("step.mp3", 15, 12), step_hour, tmp_path, wav=True) as clip:
        fmt, stream = _probe(clip)
    assert (stream["channels"], stream["sample_rate"]) == (1, "16000")
    assert abs(float(fmt["duration"]) - 12) <= MP3_FRAME_S


@pytest.mark.parametrize(("offset_s", "length_s"), [(45, 20), (60, 6), (75, 6), (99990, 6)])
def test_cut_refuses_a_clip_past_the_end_of_a_real_hour(step_hour, tmp_path, offset_s, length_s):
    with (
        pytest.raises(clips.ClipError, match="past the end"),
        clips.cut(clips.ClipAddress("step.mp3", offset_s, length_s), step_hour, tmp_path),
    ):
        pytest.fail("cut yielded a clip past the hour's end")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("length_s", clips.CAPTURE_LENGTHS_S)
def test_every_grid_clip_of_a_measured_hour_cuts_to_its_full_length(step_hour, tmp_path, length_s):
    grid = clips.grid("step.mp3", length_s, hour_s=clips.hour_duration(step_hour))
    assert grid[-1].offset_s + length_s <= HOUR_S < grid[-1].offset_s + clips.GRID_S + length_s
    with clips.cut(grid[-1], step_hour, tmp_path) as clip:
        assert abs(len(_samples(clip)) / RATE - length_s) <= MP3_FRAME_S


def test_cut_deletes_the_clip_after_exit_and_after_an_exception(step_hour, tmp_path):
    with clips.cut(clips.ClipAddress("step.mp3", 0, 6), step_hour, tmp_path, wav=True) as clip:
        assert clip.stat().st_size > 0
    assert list(tmp_path.iterdir()) == []
    with (
        pytest.raises(RuntimeError),
        clips.cut(clips.ClipAddress("step.mp3", 0, 6), step_hour, tmp_path),
    ):
        raise RuntimeError("recognizer failed")
    assert list(tmp_path.iterdir()) == []
