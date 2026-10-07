"""The adaptive loop: capture, recognize, post on change, pause.

``step()`` decides, as a pure function of the state and one recognition result;
``run()`` carries the decisions out against a source, a recognizer, and an output.
"""

import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from .config import (
    API_SECRET,
    API_URL,
    CAPTURE_FAST,
    CAPTURE_SLOW,
    INTERVAL,
    INTERVAL_GAP,
    STREAM_URL,
    VERBOSE,
)
from .outputs import HttpPostOutput, Output
from .recognizers.base import Identification, Recognizer
from .recognizers.shazam import ShazamRecognizer
from .sources import IcecastSource, Source


@dataclass(frozen=True)
class Cadence:
    """Capture lengths and pauses, in seconds (``WXDU_CAPTURE_*``, ``WXDU_INTERVAL*``)."""

    fast: int
    slow: int
    interval: int
    interval_gap: int


@dataclass(frozen=True)
class State:
    """The capture length to use next, and the (artist, song) last posted."""

    window: int
    last_key: tuple[str, str] | None = None


@dataclass(frozen=True)
class Action:
    """What one cycle does: emit a track, or log a line (some only when verbose)."""

    emit: Mapping[str, object] | None = None
    message: str | None = None
    verbose_only: bool = False


def step(state: State, track: Identification | None, cadence: Cadence) -> tuple[State, Action]:
    """Decide the next state and the cycle's action from one recognition result.

    The returned state applies only once the action succeeds: the driver keeps the
    old state if emitting raises, so a failed post is retried on the next cycle.
    """
    if track and track["song"]:
        key = (track["artist"].lower(), track["song"].lower())
        if key != state.last_key:
            # Only post when the song changes, so the DB is a clean log
            # of distinct tracks rather than a duplicate every cycle.
            return State(cadence.fast, key), Action(emit=track)
        # Any hit (new or the same track still playing) means we're
        # confident about what's on air -- keep captures speedy.
        message = f"still playing: {track['artist']} - {track['song']}"
        return State(cadence.fast, key), Action(message=message, verbose_only=True)
    # Miss: can't identify the current audio (usually a song change
    # we haven't caught yet). Lengthen the capture to improve the
    # odds, and stay there until something hits.
    missed = State(cadence.slow, state.last_key)
    if state.window != cadence.slow:
        return missed, Action(message=f"no match, extending capture to {cadence.slow}s")
    return missed, Action(message=f"no match ({cadence.slow}s)", verbose_only=True)


def pause(state: State, cadence: Cadence) -> int:
    """Shorter pause while we're in the miss/gap state, so we catch the next
    track sooner; relaxed pause once a track is identified."""
    return cadence.interval_gap if state.window == cadence.slow else cadence.interval


def identify_once(seconds):
    return _identify(IcecastSource(), ShazamRecognizer(), seconds)


def _identify(source: Source, recognizer: Recognizer, seconds: int) -> Identification | None:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        source.capture(tmp.name, seconds)
        return recognizer.recognize(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _log(message: str, **kwargs: Any) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True, **kwargs)


def run(
    source: Source,
    recognizer: Recognizer,
    output: Output,
    cadence: Cadence,
    *,
    verbose: bool,
    sleep: Callable[[float], None] | None = None,
) -> NoReturn:
    """Carry out ``step()``'s decisions forever.

    ``sleep`` defaults to ``time.sleep`` looked up on every pause, never bound at
    import, so a patch of ``time.sleep`` made after import is still observed.
    """
    state = State(cadence.fast)  # start speedy
    while True:
        try:
            next_state, action = step(state, _identify(source, recognizer, state.window), cadence)
            if action.emit is not None:
                status = output.emit(action.emit)
                # Posted: record the key before logging, as the old loop did, so a
                # failed log write cannot make the next cycle post the same song
                # again; the window changes only once the cycle completes.
                state = State(state.window, next_state.last_key)
                track = action.emit
                _log(f"posted ({status}): {track['artist']} - {track['song']}")
            elif action.message and (verbose or not action.verbose_only):
                _log(action.message)
            state = next_state
        except Exception as e:  # noqa: BLE001 - keep the loop alive through any error
            _log(f"error: {e}", file=sys.stderr)

        (sleep or time.sleep)(pause(state, cadence))


def main():
    if not API_SECRET:
        print("WXDU_SHAZAM_SECRET is not set; refusing to run.", file=sys.stderr)
        sys.exit(1)

    print(
        f"stream-sleuth: sampling {STREAM_URL} "
        f"(hit: {CAPTURE_FAST}s cap / {INTERVAL}s pause, "
        f"gap: {CAPTURE_SLOW}s cap / {INTERVAL_GAP}s pause) -> {API_URL}",
        flush=True,
    )
    cadence = Cadence(CAPTURE_FAST, CAPTURE_SLOW, INTERVAL, INTERVAL_GAP)
    run(IcecastSource(), ShazamRecognizer(), HttpPostOutput(), cadence, verbose=VERBOSE)
