"""Settings, read from the environment once, at import (see ``recognizer.py``'s docstring)."""

import os

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
