# stream-sleuth

Continuously identifies what's playing on the WXDU live stream using Shazam, and
feeds the results to the playlist manager so live DJs get one-click "now playing
on the stream" suggestions.

Runs on the WXDU iMac (modern macOS). Every ~25s it samples a few seconds of the
stream, recognizes the track with [`shazamio`](https://github.com/shazamio/ShazamIO),
and — when the song changes — POSTs it to the wxdu API, which stores it in
`plmanager.shazamplaying`. Adrenalin's playlist entry page reads the 5 most
recent rows and renders them as soft-blue buttons.

```
recognizer.py  ──HTTPS POST /api/shazam (X-Ingest-Secret)──▶  wxdu API  ──▶  MySQL
                                                                              │
                                    adrenalin playlist page  ◀──SELECT last 5─┘
```

## Setup (macOS)

```bash
# 1. ffmpeg (used to capture the stream)
brew install ffmpeg

# 2. python deps in a venv (Python 3.10–3.12; shazamio does not install on 3.13+, so on a newer default python3 use e.g. python3.12 -m venv venv)
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# 3. config
cp .env.example .env
#   - set STREAM_SLEUTH_SHAZAM_SECRET to the SAME value as the API's SHAZAM_INGEST_SECRET
#     (generate one with: openssl rand -hex 32); an existing WXDU_SHAZAM_SECRET still works
#   - or, for WXYC: cp examples/wxyc.env .env (writes JSONL locally, no secret needed)

# 4. test by hand
./run.sh
```

You should see lines like:

```
stream-sleuth: sampling https://stream.wxdu.art/wxdu192.mp3 (hit: 6s cap / 23s pause, gap: 12s cap / 4s pause) -> https://api.wxdu.art/api/shazam
[14:03:21] posted (201): Björk - Hunter
```

`no match` cycles are normal (talk breaks, obscure/local releases not in Shazam's
database) — the tool just tries again next interval.

## Run it as a background service (launchd)

```bash
# edit com.wxdu.stream-sleuth.plist: set the two /PATH/TO paths and the secret
cp com.wxdu.stream-sleuth.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.wxdu.stream-sleuth.plist

# logs:
tail -f /tmp/stream-sleuth.out.log /tmp/stream-sleuth.err.log

# stop / reload:
launchctl unload ~/Library/LaunchAgents/com.wxdu.stream-sleuth.plist
```

`KeepAlive` restarts it if it ever exits; `RunAtLoad` starts it at login.

## Config

All via environment; `.env.example` documents each setting with WXDU's values, and `examples/wxyc.env` is WXYC's. The tool emits a track only when it *changes*, so the output stays a clean log of distinct songs rather than a duplicate every cycle.

Every setting is `STREAM_SLEUTH_<NAME>`. WXDU's original `WXDU_<NAME>` spelling is a supported alias, so an existing `.env` or launchd plist keeps working unchanged. When both are set, the `STREAM_SLEUTH_` name wins; an empty one counts as unset.

| Setting | Alias | Default | Meaning |
|---|---|---|---|
| `STREAM_SLEUTH_STREAM_URL` | `WXDU_STREAM_URL` | WXDU's 192 kbps stream | Stream to sample |
| `STREAM_SLEUTH_OUTPUT` | (none) | `http` | `http` posts each new song to the API; `jsonl` appends it to a local file |
| `STREAM_SLEUTH_OUTPUT_PATH` | (none) | (none) | The JSONL file; required when the output is `jsonl` |
| `STREAM_SLEUTH_SHAZAM_API` | `WXDU_SHAZAM_API` | `https://api.wxdu.art/api/shazam` | Ingest endpoint for the `http` output |
| `STREAM_SLEUTH_SHAZAM_SECRET` | `WXDU_SHAZAM_SECRET` | (none) | Shared secret; required for the `http` output |
| `STREAM_SLEUTH_INTERVAL` | `WXDU_INTERVAL` | `23` | Pause between tries while getting hits, seconds |
| `STREAM_SLEUTH_INTERVAL_GAP` | `WXDU_INTERVAL_GAP` | `4` | Pause between tries during a miss, seconds |
| `STREAM_SLEUTH_CAPTURE_FAST` | `WXDU_CAPTURE_FAST` | `6` | Capture length while getting hits, seconds |
| `STREAM_SLEUTH_CAPTURE_SLOW` | `WXDU_CAPTURE_SLOW` | `12` | Capture length after a miss, seconds |
| `STREAM_SLEUTH_VERBOSE` | `WXDU_VERBOSE` | off | `1`, `true`, or `yes` logs every cycle |

The recognizer refuses to start (message on stderr, exit 1) when the chosen output is incomplete: `http` without a secret, or `jsonl` without a path. A JSONL record is the identification (`artist`, `song`, `album`, `label`) plus `emitted_at`.

## Development

Contributors use [`uv`](https://docs.astral.sh/uv/) and the committed `uv.lock`; the deploy above is unchanged and still uses `requirements.txt`.

```bash
uv sync --locked --extra dev --extra eval   # picks Python 3.12 from .python-version
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy . --ignore-missing-imports
```

The package supports Python 3.10 through 3.12: `shazamio` pins `shazamio-core`, whose macOS wheels stop at 3.12, and its dependency `pydub` imports `audioop`, which Python 3.13 removed. See `CLAUDE.md` for the test and data conventions.

## Evaluation harness

`evaluation/` is the research harness for measuring recognition accuracy against a station's archived broadcasts. It is never installed and the live recognizer never imports it; its dependencies are the `eval` extra.

Its data and settings live outside the checkout:

- **`STREAM_SLEUTH_DATA_DIR`** (default `~/.local/share/stream-sleuth/`) holds audio, indexes, exports, and results. Nothing under it is ever committed.
- **`$STREAM_SLEUTH_DATA_DIR/eval.env`** holds the harness's settings, including the reference pool's key pair. Keep it at mode `600`.

Load `eval.env` into **one process at a time**, never into your interactive shell:

```bash
uv run --extra eval --env-file "${STREAM_SLEUTH_DATA_DIR:-$HOME/.local/share/stream-sleuth}/eval.env" python -m evaluation.<module> ...
# or, in a subshell:
( set -a; . "${STREAM_SLEUTH_DATA_DIR:-$HOME/.local/share/stream-sleuth}/eval.env"; set +a; uv run --extra eval python -m evaluation.<module> ... )
```

`run.sh` exports its own environment into the recognizer, so a `./run.sh` started from a shell that had sourced `eval.env` would carry the harness's credentials into the live process.

Every S3 client the harness uses comes from `evaluation/s3_readonly.py`, which refuses any operation other than `ListObjectsV2`, `GetObject`, and `HeadObject`. That refusal guards against mistakes in the harness's own code; it is not a substitute for read-only credentials. It reads:

| Variable | Meaning |
|---|---|
| `STREAM_SLEUTH_ARCHIVE_BUCKET` | The broadcast archive's bucket, read by `archive_bucket()`. |
| `STREAM_SLEUTH_ARCHIVE_AWS_PROFILE` | Named AWS profile for the broadcast archive; unset means the default credential chain. Use a read-only profile where one exists. |
| `STREAM_SLEUTH_POOL_ENDPOINT`, `_BUCKET`, `_KEY_ID`, `_SECRET` | The reference pool's S3-compatible store. Each one unset or empty falls back to the same suffix under `DIGITAL_ARCHIVE_STORE_AZURACAST_`, the names WXYC's Backend-Service uses. Prefer a read-only key scoped to the bucket: the fallback key can write. |
| `STREAM_SLEUTH_POOL_PREFIXES` | Comma-separated key prefixes that make up the reference pool, e.g. `rotation/Heavy/,rotation/Medium/`. Read by `evaluation/pool.py`. |

### Reference pool

`evaluation/pool.py` builds the reference pool without mirroring it. `inventory()` lists every object under the prefixes, and `summarize()` counts objects and bytes per prefix and format before anything is fetched. `stream()` then fetches one audio file at a time into the staging directory its caller passes, named by the SHA-1 of its key, reads its tags into the `pool.db` its caller opens, hands it to a consumer (the index build), and deletes it whether the consumer succeeds or fails. The module reads no data-directory setting, so callers pass paths under `$STREAM_SLEUTH_DATA_DIR`, e.g. `$STREAM_SLEUTH_DATA_DIR/pool-staging/` and `$STREAM_SLEUTH_DATA_DIR/pool.db`, never the checkout. A rerun skips files already indexed and retries only failed ones. Files without an artist or a title tag are indexed anyway and counted as `untagged` in the counts `stream()` returns and logs. Supported formats are mp3, aac, wav, flac, and m4a/mp4, each read with its own `mutagen` reader; anything else is counted and skipped.

### Broadcast archive

`evaluation/archive.py` is WXYC-specific. WXYC's archive holds one MP3 per hour, keyed `YYYY/MM/DD/YYYYMMDDHH00.mp3` by the hour's America/New_York local time, not UTC. `hour_key()` maps a timezone-aware instant to its key and returns `None` for the fall-back hour, which is recorded twice under one key with the later recording overwriting the earlier; `hour_start()` is its inverse. `fetch()` downloads an hour to `<archive_dir>/<key>` through a `.part` file, keeps an existing file of the right size, and never overwrites one of any other size; callers pass `$STREAM_SLEUTH_DATA_DIR/archive` as `archive_dir`, never a path inside the checkout. `fetch_all()` skips and reports per-hour failures and stops on anything that would fail every hour, such as expired credentials.

## Notes

- Only ASCII/UTF-8 metadata is sent; the API stores it in a `utf8mb4` table.
- The shared secret is the only auth on the ingest endpoint — keep `.env` and the
  plist readable only by the service account. Optionally also restrict the API's
  `/api/shazam` route to this iMac's static IP at the Apache/nginx layer.
