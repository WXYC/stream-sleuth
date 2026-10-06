"""Settings, read from the environment once, at import (see ``recognizer.py``'s docstring).

Each setting is ``STREAM_SLEUTH_<NAME>``, with WXDU's original ``WXDU_<NAME>`` kept as
a supported alias. The new name wins when both are set; an empty new name counts as
unset, so a blank line copied from ``.env.example`` never hides the alias.
"""

import os


def _env(name: str, default: str) -> str:
    return os.environ.get(f"STREAM_SLEUTH_{name}") or os.environ.get(f"WXDU_{name}", default)


STREAM_URL = _env("STREAM_URL", "https://stream.wxdu.art/wxdu192.mp3")
API_URL = _env("SHAZAM_API", "https://api.wxdu.art/api/shazam")
API_SECRET = _env("SHAZAM_SECRET", "")
INTERVAL = int(_env("INTERVAL", "23"))
INTERVAL_GAP = int(_env("INTERVAL_GAP", "4"))
CAPTURE_FAST = int(_env("CAPTURE_FAST", "6"))
CAPTURE_SLOW = int(_env("CAPTURE_SLOW", "12"))
# When set (1/true/yes), log every cycle -- including same-song hits and repeat
# misses -- so you can watch it tick. Off by default to keep the log quiet.
VERBOSE = _env("VERBOSE", "").lower() in ("1", "true", "yes")
# Where identifications go: "http" (POST to SHAZAM_API, WXDU's default) or "jsonl"
# (append to OUTPUT_PATH). New settings, so they have no WXDU_ alias.
OUTPUT = os.environ.get("STREAM_SLEUTH_OUTPUT") or "http"
OUTPUT_PATH = os.environ.get("STREAM_SLEUTH_OUTPUT_PATH", "")
