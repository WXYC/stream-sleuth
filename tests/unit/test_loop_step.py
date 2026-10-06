"""The loop's decisions as a pure function, and the driver that carries them out.

``step()`` is checked transition by transition with no I/O; the characterization
tests pin the same behavior end to end through ``recognizer.main()``. The driver
tests use in-memory parts and stop the endless loop from the injected sleep.
"""

from __future__ import annotations

import time

import pytest

from stream_sleuth.loop import Action, Cadence, State, pause, run, step

CADENCE = Cadence(fast=6, slow=12, interval=23, interval_gap=4)
JUANA = {"artist": "Juana Molina", "song": "la paradoja", "album": "DOGA", "label": "Sonamos"}
JUANA_SHOUTED = {**JUANA, "artist": "JUANA MOLINA", "song": "La Paradoja"}
PRATT = {"artist": "Jessica Pratt", "song": "Back, Baby", "album": "", "label": ""}
NO_TITLE = {**PRATT, "song": ""}
JUANA_KEY = ("juana molina", "la paradoja")


@pytest.mark.parametrize(
    ("state", "track", "expected_state", "expected_action"),
    [
        pytest.param(
            State(window=6), JUANA, State(6, JUANA_KEY), Action(emit=JUANA), id="new-song-posts"
        ),
        pytest.param(
            State(12, ("jessica pratt", "back, baby")),
            JUANA,
            State(6, JUANA_KEY),
            Action(emit=JUANA),
            id="a-hit-in-a-gap-returns-to-speedy",
        ),
        pytest.param(
            State(6, JUANA_KEY),
            JUANA_SHOUTED,
            State(6, JUANA_KEY),
            Action(message="still playing: JUANA MOLINA - La Paradoja", verbose_only=True),
            id="same-song-any-case-does-not-post",
        ),
        pytest.param(
            State(6, JUANA_KEY),
            None,
            State(12, JUANA_KEY),
            Action(message="no match, extending capture to 12s"),
            id="first-miss-escalates-and-keeps-the-key",
        ),
        pytest.param(
            State(12, JUANA_KEY),
            None,
            State(12, JUANA_KEY),
            Action(message="no match (12s)", verbose_only=True),
            id="repeat-miss-stays-slow",
        ),
        pytest.param(
            State(6),
            NO_TITLE,
            State(12),
            Action(message="no match, extending capture to 12s"),
            id="empty-title-is-a-miss",
        ),
    ],
)
def test_step(state, track, expected_state, expected_action):
    assert step(state, track, CADENCE) == (expected_state, expected_action)


@pytest.mark.parametrize(("window", "expected"), [(6, 23), (12, 4)])
def test_pause_is_short_only_in_the_slow_capture_state(window, expected):
    assert pause(State(window), CADENCE) == expected


class Stop(BaseException):
    pass


class Exhausted(BaseException):
    """The scripted parts ran out: the driver did not stop where the test expected.

    A ``BaseException``, so ``run()``'s ``except Exception`` cannot swallow it and
    loop forever.
    """


class Parts:
    """A source, recognizer, and output scripted cycle by cycle, recording what they did."""

    def __init__(self, results, emit_errors=()):
        self.results = list(results)
        self.emit_errors = list(emit_errors)
        self.captures: list[int] = []
        self.emitted: list[dict] = []

    def capture(self, path, seconds):
        self.captures.append(seconds)

    def recognize(self, wav_path):
        if not self.results:
            raise Exhausted
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def emit(self, track):
        error = self.emit_errors.pop(0) if self.emit_errors else None
        if error:
            raise error
        self.emitted.append(track)
        return 201


def drive(parts, cycles, sleep=None):
    pauses = []

    def record(seconds):
        pauses.append(seconds)
        if len(pauses) == cycles:
            raise Stop

    with pytest.raises(Stop):
        run(parts, parts, parts, CADENCE, verbose=False, sleep=sleep or record)
    return pauses


def test_a_failed_emit_keeps_the_previous_state_so_the_next_cycle_retries():
    parts = Parts([JUANA, JUANA], emit_errors=[RuntimeError("HTTP Error 500")])

    pauses = drive(parts, 2)

    assert parts.emitted == [JUANA]
    assert parts.captures == [6, 6]
    assert pauses == [23, 23]


def test_a_successful_emit_is_kept_even_if_logging_it_fails(monkeypatch):
    # The old loop set last_key straight after post() returned, so a failing log
    # write must not make the next cycle post the same song again.
    from stream_sleuth import loop

    def log(message, **kwargs):
        if message.startswith("posted"):
            raise OSError("stdout is gone")

    monkeypatch.setattr(loop, "_log", log)
    parts = Parts([JUANA, JUANA])

    drive(parts, 2)

    assert parts.emitted == [JUANA]


def test_a_recognizer_error_keeps_the_current_capture_state():
    parts = Parts([None, RuntimeError("shazam broke"), JUANA])

    pauses = drive(parts, 3)

    assert parts.captures == [6, 12, 12]
    assert pauses == [4, 4, 23]


def test_the_driver_resolves_time_sleep_at_call_time(monkeypatch):
    seen = []

    def patched(seconds):
        seen.append(seconds)
        raise Stop

    monkeypatch.setattr(time, "sleep", patched)
    with pytest.raises(Stop):
        run(Parts([JUANA]), Parts([JUANA]), Parts([]), CADENCE, verbose=False)

    assert seen == [23]
