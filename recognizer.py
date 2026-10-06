#!/usr/bin/env python3
"""WXDU stream Shazam recognizer.

Runs continuously on the WXDU iMac. Every INTERVAL seconds it samples a few
seconds of the live stream, identifies the track with Shazam, and -- when the
song changes -- POSTs it to the wxdu API, which stores it in
plmanager.shazamplaying. Adrenalin's playlist entry page surfaces the 5 most
recent as clickable buttons for live DJs.

Config is via environment variables (see .env.example / the launchd plist):

  WXDU_STREAM_URL    stream to sample      (default: 192 kbps stream)
  WXDU_SHAZAM_API    ingest endpoint URL   (default: https://api.wxdu.art/api/shazam)
  WXDU_SHAZAM_SECRET shared secret         (REQUIRED; matches the API's SHAZAM_INGEST_SECRET)
  WXDU_INTERVAL      pause between tries    (default: 23; while getting hits)
  WXDU_INTERVAL_GAP  pause between tries    (default: 4;  during a miss/gap, to
                                             catch the next track sooner)
  WXDU_CAPTURE_FAST  speedy capture seconds (default: 6;  used while getting hits)
  WXDU_CAPTURE_SLOW  longer capture seconds (default: 12; used after a miss)
  WXDU_VERBOSE       log every cycle        (default: off; set 1 to see each tick,
                                             incl. same-song hits and repeat misses)

Capture length adapts: it starts speedy (WXDU_CAPTURE_FAST). Any hit -- a new
song or the same one still playing -- keeps it speedy. A miss (can't identify
what's on air) escalates to WXDU_CAPTURE_SLOW to improve the odds, staying there
until a hit lands, then dropping back to speedy.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

from shazamio import Shazam

STREAM_URL = os.environ.get("WXDU_STREAM_URL", "https://stream.wxdu.art/wxdu192.mp3")
API_URL = os.environ.get("WXDU_SHAZAM_API", "https://api.wxdu.art/api/shazam")
API_SECRET = os.environ.get("WXDU_SHAZAM_SECRET", "")
INTERVAL = int(os.environ.get("WXDU_INTERVAL", "23"))
INTERVAL_GAP = int(os.environ.get("WXDU_INTERVAL_GAP", "4"))
CAPTURE_FAST = int(os.environ.get("WXDU_CAPTURE_FAST", "6"))
CAPTURE_SLOW = int(os.environ.get("WXDU_CAPTURE_SLOW", "12"))
# When set (1/true/yes), log every cycle -- including same-song hits and repeat
# misses -- so you can watch it tick. Off by default to keep the log quiet.
VERBOSE = os.environ.get("WXDU_VERBOSE", "").lower() in ("1", "true", "yes")


def capture(path, seconds):
    """Grab `seconds` of the stream into a small mono 16kHz wav via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            STREAM_URL,
            "-t",
            str(seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            path,
        ],
        check=True,
        timeout=seconds + 25,
        stdin=subprocess.DEVNULL,
    )


def parse(out):
    """Pull artist/song/album/label out of Shazam's response, or None on no match."""
    track = out.get("track") if isinstance(out, dict) else None
    if not track:
        return None
    result = {
        "artist": track.get("subtitle", "") or "",
        "song": track.get("title", "") or "",
        "album": "",
        "label": "",
    }
    for section in track.get("sections", []) or []:
        for md in section.get("metadata", []) or []:
            key = (md.get("title") or "").strip().lower()
            val = md.get("text", "") or ""
            if key == "album" and not result["album"]:
                result["album"] = val
            elif key == "label" and not result["label"]:
                result["label"] = val
    return result


def post(track):
    """POST a recognized track to the wxdu API ingest endpoint."""
    body = json.dumps(track).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Ingest-Secret": API_SECRET,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status


async def _recognize(path):
    return await Shazam().recognize(path)


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


if __name__ == "__main__":
    main()
