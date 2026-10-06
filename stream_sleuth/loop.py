"""The adaptive loop: capture, recognize, post on change, pause."""

import asyncio
import os
import sys
import tempfile
import time

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
from .outputs import post
from .recognizers.shazam import _recognize, parse
from .sources import capture


def identify_once(seconds):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        capture(tmp.name, seconds)
        return parse(asyncio.run(_recognize(tmp.name)))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


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
    last_key = None
    window = CAPTURE_FAST  # start speedy

    while True:
        try:
            track = identify_once(window)
            if track and track["song"]:
                key = (track["artist"].lower(), track["song"].lower())
                if key != last_key:
                    # Only post when the song changes, so the DB is a clean log
                    # of distinct tracks rather than a duplicate every cycle.
                    status = post(track)
                    last_key = key
                    print(
                        f"[{time.strftime('%H:%M:%S')}] posted ({status}): "
                        f"{track['artist']} - {track['song']}",
                        flush=True,
                    )
                elif VERBOSE:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] still playing: "
                        f"{track['artist']} - {track['song']}",
                        flush=True,
                    )
                # Any hit (new or the same track still playing) means we're
                # confident about what's on air -- keep captures speedy.
                window = CAPTURE_FAST
            else:
                # Miss: can't identify the current audio (usually a song change
                # we haven't caught yet). Lengthen the capture to improve the
                # odds, and stay there until something hits.
                if window != CAPTURE_SLOW:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] no match, "
                        f"extending capture to {CAPTURE_SLOW}s",
                        flush=True,
                    )
                elif VERBOSE:
                    print(f"[{time.strftime('%H:%M:%S')}] no match ({CAPTURE_SLOW}s)", flush=True)
                window = CAPTURE_SLOW
        except Exception as e:  # noqa: BLE001 - keep the loop alive through any error
            print(f"[{time.strftime('%H:%M:%S')}] error: {e}", file=sys.stderr, flush=True)

        # Shorter pause while we're in the miss/gap state, so we catch the next
        # track sooner; relaxed pause once a track is identified.
        time.sleep(INTERVAL_GAP if window == CAPTURE_SLOW else INTERVAL)
