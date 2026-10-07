#!/usr/bin/env python3
"""WXDU stream Shazam recognizer.

Runs continuously on the WXDU iMac. Every INTERVAL seconds it samples a few
seconds of the live stream, identifies the track with Shazam, and -- when the
song changes -- POSTs it to the wxdu API, which stores it in
plmanager.shazamplaying. Adrenalin's playlist entry page surfaces the 5 most
recent as clickable buttons for live DJs.

Config is via environment variables (see .env.example / the launchd plist).
Each WXDU_<NAME> below is also readable as STREAM_SLEUTH_<NAME>, which wins
when both are set; STREAM_SLEUTH_OUTPUT=jsonl writes to a local file instead
of POSTing (see the README's Config section):

  WXDU_STREAM_URL    stream to sample      (default: 192 kbps stream)
  WXDU_SHAZAM_API    ingest endpoint URL   (default: https://api.wxdu.art/api/shazam)
  WXDU_SHAZAM_SECRET shared secret         (required for http output; matches the API's SHAZAM_INGEST_SECRET)
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

# The code lives in the stream_sleuth package; this module keeps
# `python recognizer.py`, WXDU's launchd job, and its names working.
from stream_sleuth.config import (
    API_SECRET,
    API_URL,
    CAPTURE_FAST,
    CAPTURE_SLOW,
    INTERVAL,
    INTERVAL_GAP,
    STREAM_URL,
    VERBOSE,
)
from stream_sleuth.loop import identify_once, main
from stream_sleuth.outputs import post
from stream_sleuth.recognizers.shazam import parse
from stream_sleuth.sources import capture

__all__ = [
    "API_SECRET",
    "API_URL",
    "CAPTURE_FAST",
    "CAPTURE_SLOW",
    "INTERVAL",
    "INTERVAL_GAP",
    "STREAM_URL",
    "VERBOSE",
    "capture",
    "identify_once",
    "main",
    "parse",
    "post",
]

if __name__ == "__main__":
    main()
