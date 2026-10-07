"""``tests.audio.render``: the one place a test builds a synthetic-audio ffmpeg command line.

A fake ``subprocess.run`` records the command, so these run without an encoder; the
``ffmpeg``- and ``olaf``-marked tests that call ``render`` exercise the real thing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests import audio

FORMAT = audio.FORMAT
PREFIX = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict]]:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    return calls


def test_a_bare_duration_renders_a_440_hz_sine(
    commands: list[tuple[list[str], dict]], tmp_path: Path
) -> None:
    out = tmp_path / "tone.mp3"

    assert audio.render(out, 10) == out

    cmd, kwargs = commands[0]
    assert cmd == [*PREFIX, "-f", FORMAT, "-i", "sine=frequency=440:duration=10", str(out)]
    assert kwargs["check"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL


def test_output_arguments_go_between_the_inputs_and_the_path(
    commands: list[tuple[list[str], dict]], tmp_path: Path
) -> None:
    out = tmp_path / "tone.mp3"

    audio.render(out, ("anoisesrc=c=pink:seed=11:a=0.5", 60), args=["-ac", "1", "-ar", "16000"])

    assert commands[0][0] == [
        *PREFIX,
        "-f", FORMAT, "-i", "anoisesrc=c=pink:seed=11:a=0.5:duration=60",
        "-ac", "1", "-ar", "16000",
        str(out),
    ]  # fmt: skip


def test_two_or_more_segments_are_concatenated(
    commands: list[tuple[list[str], dict]], tmp_path: Path
) -> None:
    out = tmp_path / "step.mp3"

    audio.render(
        out,
        ("anullsrc=r=44100:cl=mono", 30),
        ("sine=frequency=440:sample_rate=44100", 30),
        ("sine=frequency=880:sample_rate=44100", 5),
        args=["-c:a", "libmp3lame"],
    )

    assert commands[0][0] == [
        *PREFIX,
        "-f", FORMAT, "-i", "anullsrc=r=44100:cl=mono:duration=30",
        "-f", FORMAT, "-i", "sine=frequency=440:sample_rate=44100:duration=30",
        "-f", FORMAT, "-i", "sine=frequency=880:sample_rate=44100:duration=5",
        "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1",
        "-c:a", "libmp3lame",
        str(out),
    ]  # fmt: skip


@pytest.mark.parametrize(
    "source", ["anoisesrc=d=60:c=pink", "sine=duration=60", "anullsrc=r=8000:d=60"]
)
def test_a_source_that_already_sets_its_duration_gets_no_second_one(
    commands: list[tuple[list[str], dict]], tmp_path: Path, source: str
) -> None:
    audio.render(tmp_path / "x.wav", (source, 60))

    assert commands[0][0][commands[0][0].index("-i") + 1] == source


def test_a_failing_ffmpeg_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(audio.subprocess, "run", fail)

    with pytest.raises(subprocess.CalledProcessError):
        audio.render(tmp_path / "x.wav", 1)
