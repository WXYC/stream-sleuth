"""Pins ``recognizer.main()``'s loop: what it captures, posts, logs, and pauses for.

Each test scripts one Shazam response per cycle; the run stops after that many
loop pauses (see ``conftest.py``). Capture lengths are read from the ``-t``
argument the stub ``ffmpeg`` received, and pauses are the sentinel sleeps.
"""

from __future__ import annotations

import os
import re

import pytest

from tests.characterization import shazam_responses as r
from tests.characterization.conftest import INTERVAL, INTERVAL_GAP, STREAM_URL

HIT, GAP = INTERVAL, INTERVAL_GAP
STAMP = r"\[\d\d:\d\d:\d\d\] "


def stripped(lines: list[str]) -> list[str]:
    """Log lines without their ``[HH:MM:SS]`` prefix, after checking it is there."""
    for line in lines:
        assert re.match(STAMP, line), line
    return [re.sub(f"^{STAMP}", "", line) for line in lines]


def test_startup_line_names_the_stream_cadence_and_endpoint(run_main, ingest_server):
    run = run_main([r.NO_MATCH])

    assert run.stdout[0] == (
        f"stream-sleuth: sampling {STREAM_URL} "
        f"(hit: 6s cap / {HIT}s pause, gap: 12s cap / {GAP}s pause) -> {ingest_server.url}"
    )


def test_capture_and_pause_follow_hits_and_misses(run_main):
    run = run_main(
        [r.JUANA_MOLINA, r.JUANA_MOLINA, r.NO_MATCH, r.NO_MATCH, r.JESSICA_PRATT, r.JESSICA_PRATT]
    )

    # Starts speedy; a miss escalates to the slow capture and the short pause,
    # which hold until the next hit.
    assert run.capture_seconds == [6, 6, 6, 12, 12, 6]
    assert run.pauses == [HIT, HIT, GAP, GAP, HIT, HIT]


def test_posts_only_when_the_song_changes(run_main):
    same_song_other_case = r.match("JUANA MOLINA", "La Paradoja", album="DOGA")
    run = run_main(
        [r.JUANA_MOLINA, same_song_other_case, r.NO_MATCH, r.JUANA_MOLINA, r.JESSICA_PRATT]
    )

    # The change key is (artist, song) lowercased, and a miss does not reset it.
    assert [p.json()["artist"] for p in run.posts] == ["Juana Molina", "Jessica Pratt"]
    assert run.posts[0].json() == {
        "artist": "Juana Molina",
        "song": "la paradoja",
        "album": "DOGA",
        "label": "Sonamos",
    }


def test_quiet_log_reports_posts_and_the_first_miss_only(run_main):
    run = run_main([r.JUANA_MOLINA, r.JUANA_MOLINA, r.NO_MATCH, r.NO_MATCH, r.JESSICA_PRATT])

    assert stripped(run.stdout[1:]) == [
        "posted (201): Juana Molina - la paradoja",
        "no match, extending capture to 12s",
        "posted (201): Jessica Pratt - Back, Baby",
    ]
    assert run.stderr == []


def test_verbose_log_reports_every_cycle(run_main):
    run = run_main(
        [r.JUANA_MOLINA, r.JUANA_MOLINA, r.NO_MATCH, r.NO_MATCH, r.JESSICA_PRATT],
        WXDU_VERBOSE="1",
    )

    assert stripped(run.stdout[1:]) == [
        "posted (201): Juana Molina - la paradoja",
        "still playing: Juana Molina - la paradoja",
        "no match, extending capture to 12s",
        "no match (12s)",
        "posted (201): Jessica Pratt - Back, Baby",
    ]


def test_a_match_with_an_empty_title_counts_as_a_miss(run_main):
    run = run_main([r.match("Jessica Pratt", ""), r.JESSICA_PRATT])

    assert run.capture_seconds == [6, 12]
    assert run.pauses == [GAP, HIT]
    assert [p.json()["song"] for p in run.posts] == ["Back, Baby"]


def test_a_failed_post_is_logged_and_retried_on_the_next_cycle(run_main):
    run = run_main([r.JUANA_MOLINA, r.JUANA_MOLINA], post_statuses=[500])

    assert [p.json()["song"] for p in run.posts] == ["la paradoja", "la paradoja"]
    assert stripped(run.stderr) == ["error: HTTP Error 500: Internal Server Error"]
    assert stripped(run.stdout[1:]) == ["posted (201): Juana Molina - la paradoja"]
    # The error leaves the capture state alone: still speedy, still the hit pause.
    assert run.capture_seconds == [6, 6]
    assert run.pauses == [HIT, HIT]


@pytest.mark.parametrize(
    ("responses", "captures", "pauses"),
    [
        pytest.param(
            [RuntimeError("shazam broke"), r.JUANA_MOLINA], [6, 6], [HIT, HIT], id="while-speedy"
        ),
        pytest.param(
            [r.NO_MATCH, RuntimeError("shazam broke"), r.JUANA_MOLINA],
            [6, 12, 12],
            [GAP, GAP, HIT],
            id="while-in-a-gap",
        ),
    ],
)
def test_an_error_keeps_the_current_capture_state(run_main, responses, captures, pauses):
    run = run_main(responses)

    assert run.capture_seconds == captures
    assert run.pauses == pauses
    assert stripped(run.stderr) == ["error: shazam broke"]


def test_a_failed_capture_is_logged_and_the_loop_continues(run_main, ffmpeg_stub):
    ffmpeg_stub.fail(1)
    # The failed capture never reaches Shazam, so only one response is consumed;
    # the extra entry is never read but sets the run to two cycles.
    run = run_main([r.JUANA_MOLINA, r.JUANA_MOLINA])

    [error] = stripped(run.stderr)
    assert error.startswith("error: Command '['ffmpeg'")
    assert error.endswith("returned non-zero exit status 1.")
    assert run.capture_seconds == [6, 6]
    assert run.pauses == [HIT, HIT]
    assert [p.json()["song"] for p in run.posts] == ["la paradoja"]


def test_each_capture_invokes_ffmpeg_with_the_same_arguments(run_main):
    run = run_main([r.NO_MATCH, r.JUANA_MOLINA])

    for argv, seconds in zip(run.ffmpeg_argv, [6, 12], strict=True):
        *head, output = argv
        assert head == [
            "-y", "-loglevel", "error",
            "-i", STREAM_URL,
            "-t", str(seconds),
            "-ac", "1", "-ar", "16000",
            "-f", "wav",
        ]  # fmt: skip
        assert output.endswith(".wav")


def test_each_capture_file_is_removed_after_recognition(run_main, fake_shazam):
    run = run_main([r.NO_MATCH, r.JUANA_MOLINA])

    outputs = [argv[-1] for argv in run.ffmpeg_argv]
    assert fake_shazam.paths == outputs
    assert len(set(outputs)) == 2
    assert not any(os.path.exists(path) for path in outputs)
