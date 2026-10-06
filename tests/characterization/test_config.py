"""Pins the ``WXDU_*`` settings: their defaults, and the refusal to run without a secret."""

from __future__ import annotations

import pytest


def test_defaults_when_only_the_secret_is_set(fresh_recognizer):
    recognizer = fresh_recognizer(WXDU_SHAZAM_SECRET="x")

    assert recognizer.STREAM_URL == "https://stream.wxdu.art/wxdu192.mp3"
    assert recognizer.API_URL == "https://api.wxdu.art/api/shazam"
    assert (recognizer.INTERVAL, recognizer.INTERVAL_GAP) == (23, 4)
    assert (recognizer.CAPTURE_FAST, recognizer.CAPTURE_SLOW) == (6, 12)
    assert recognizer.VERBOSE is False


@pytest.mark.parametrize(
    ("value", "verbose"),
    [("1", True), ("true", True), ("YES", True), ("0", False), ("", False), ("on", False)],
)
def test_verbose_accepts_one_true_or_yes(fresh_recognizer, value, verbose):
    recognizer = fresh_recognizer(WXDU_SHAZAM_SECRET="x", WXDU_VERBOSE=value)

    assert recognizer.VERBOSE is verbose


def test_refuses_to_run_without_a_secret(fresh_recognizer, capsys, ingest_server, tmp_path):
    from tests.ffmpeg_stub import ffmpeg_calls

    recognizer = fresh_recognizer(WXDU_SHAZAM_API=ingest_server.url)

    with pytest.raises(SystemExit) as exit_info:
        recognizer.main()

    assert exit_info.value.code == 1
    out = capsys.readouterr()
    assert out.err == "WXDU_SHAZAM_SECRET is not set; refusing to run.\n"
    assert out.out == ""
    assert ingest_server.requests == []
    assert ffmpeg_calls(tmp_path) == []
