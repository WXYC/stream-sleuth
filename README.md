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
#   - or, for WXYC: cp examples/wxyc.env .env and mkdir -p ~/.local/share/stream-sleuth
#     (appends JSONL there, outside the checkout; no secret needed)

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
| `STREAM_SLEUTH_OUTPUT_PATH` | (none) | (none) | The JSONL file, an absolute path outside the checkout (e.g. under `~/.local/share/stream-sleuth/`); required when the output is `jsonl` |
| `STREAM_SLEUTH_SHAZAM_API` | `WXDU_SHAZAM_API` | `https://api.wxdu.art/api/shazam` | Ingest endpoint for the `http` output |
| `STREAM_SLEUTH_SHAZAM_SECRET` | `WXDU_SHAZAM_SECRET` | (none) | Shared secret; required for the `http` output |
| `STREAM_SLEUTH_INTERVAL` | `WXDU_INTERVAL` | `23` | Pause between tries while getting hits, seconds |
| `STREAM_SLEUTH_INTERVAL_GAP` | `WXDU_INTERVAL_GAP` | `4` | Pause between tries during a miss, seconds |
| `STREAM_SLEUTH_CAPTURE_FAST` | `WXDU_CAPTURE_FAST` | `6` | Capture length while getting hits, seconds |
| `STREAM_SLEUTH_CAPTURE_SLOW` | `WXDU_CAPTURE_SLOW` | `12` | Capture length after a miss, seconds |
| `STREAM_SLEUTH_VERBOSE` | `WXDU_VERBOSE` | off | `1`, `true`, or `yes` logs every cycle |
| `STREAM_SLEUTH_OLAF_BIN` | (none) | `olaf` on `PATH` | The Olaf binary for the local recognizer (below) |

The recognizer refuses to start (message on stderr, exit 1) when the chosen output is incomplete: `http` without a secret (the message names the `WXDU_SHAZAM_SECRET` alias, as it always has), or `jsonl` without an absolute path outside the checkout that it can append to (it creates the file, not its directory). A JSONL record is the identification (`artist`, `song`, `album`, `label`) plus `emitted_at`.

## Local recognizer (optional)

`stream_sleuth/recognizers/olaf.py` matches audio against a station's own reference files with [Olaf](https://github.com/JorenSix/Olaf) (AGPL-3.0, compatible with this repo's GPL-3.0), run as a subprocess. It is not a pip dependency, and nothing uses it unless a station turns it on. Build the pinned commit with Zig 0.16.0:

```bash
git clone https://github.com/JorenSix/Olaf.git && cd Olaf
git checkout a98d8c03cfd447011d402718ca2d10b2bb467eb0
zig build -Doptimize=ReleaseFast          # Zig 0.16.0; also fetches Olaf's zigzag dependency
export STREAM_SLEUTH_OLAF_BIN="$PWD/zig-out/bin/olaf"
```

Olaf decodes with `ffmpeg`. Each index is a *snapshot* directory that the adapter passes to Olaf as `HOME`, so snapshots never share a database and `~/.olaf` is never touched. Keep snapshots under `$STREAM_SLEUTH_DATA_DIR/olaf/<snapshot>/`; the adapter refuses a relative path, the home directory itself, and any path inside the checkout. Only `index build` creates a snapshot; querying one that holds no index is an error, so a mistyped path fails rather than matching nothing. Fill one with:

```bash
uv run python -m stream_sleuth.cli index build --home "$STREAM_SLEUTH_DATA_DIR/olaf/rotation" track.mp3 some-identifier [more.mp3 another-id ...]
```

Olaf reports the identifier as the match's `ref_key`. `evaluation.olaf_snapshot.build_snapshot(name, objects)` builds a snapshot from the reference pool: it streams each object with `evaluation.pool.stream()`, runs one `olaf store` per staged file under its stage id, and writes the snapshot's own `pool.db` at `$STREAM_SLEUTH_DATA_DIR/olaf/<name>/pool.db` beside its `HOME`. One `pool.db` per snapshot, because `stream()` skips every key a `pool.db` already marks `indexed`, so a second snapshot over the first's would index nothing. `evaluation.pool.tag_lookup(db)` maps a stage id to the station's tags (`title` as `song`, NULL tags as `""`, `label` always `""`) and goes to `OlafRecognizer(lookup=...)`; a reference with no title tag, or a stage id that is not an indexed row, has an empty song, which the loop treats as a miss. `recognizer_identity(name, min_match_count)` returns `olaf@<OLAF_COMMIT>, snapshot=<name>, min=<floor>`, the identity stored results are filed under; the floor is part of it because matches below it are dropped before anything is stored. A snapshot the evaluation harness scores must use each reference's pool stage id (`evaluation.pool.stage_id(object_key)`) as its identifier, or its matches cannot be joined to `pool.db`.

Matches below a `match_count` of 12 are ignored: in WXYC's first test, real songs scored 17 to 178 on a 12 s clip and stray matches 6 to 10.

## Development

Contributors use [`uv`](https://docs.astral.sh/uv/) and the committed `uv.lock`; the deploy above is unchanged and still uses `requirements.txt`.

```bash
uv sync --locked --extra dev --extra eval   # picks Python 3.12 from .python-version
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy . --ignore-missing-imports
uv run pytest -m "ffmpeg"    # needs ffmpeg on PATH
uv run pytest -m "olaf"      # needs ffmpeg and STREAM_SLEUTH_OLAF_BIN (see Local recognizer)
```

The package supports Python 3.10 through 3.12: `shazamio` pins `shazamio-core`, whose macOS wheels stop at 3.12, and its dependency `pydub` imports `audioop`, which Python 3.13 removed. See `CLAUDE.md` for the test and data conventions.

## Evaluation harness

`evaluation/` is the research harness for measuring recognition accuracy against a station's archived broadcasts. It is never installed and the live recognizer never imports it; its dependencies are the `eval` extra.

Its data and settings live outside the checkout:

- **`STREAM_SLEUTH_DATA_DIR`** (default `~/.local/share/stream-sleuth/`) holds audio, indexes, exports, and results. Nothing under it is ever committed. `stream_sleuth.paths.data_dir()` is the one place the setting and its default are resolved; it refuses a relative path or one inside the checkout.
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

`evaluation/pool.py` builds the reference pool without mirroring it. `inventory()` lists every object under the prefixes, and `summarize()` counts objects and bytes per prefix and format before anything is fetched. `stream()` then fetches one audio file at a time into the staging directory its caller passes, named by the SHA-1 of its key, reads its tags into the `pool.db` its caller opens, hands it to a consumer (the index build), and deletes it whether the consumer succeeds or fails. The module reads no data-directory setting, so callers pass paths under `data_dir()`; the Olaf snapshot build (below) passes `data_dir() / "olaf" / <snapshot> / "pool.db"` and `.../staging`. `stream()` and `open_pool_db()` raise `DataPathError` for a relative path or one inside the checkout. A rerun skips files already indexed and retries only failed ones. Files without an artist or a title tag are indexed anyway and counted as `untagged` in the counts `stream()` returns and logs. Supported formats are mp3, aac, wav, flac, and m4a/mp4, each read with its own `mutagen` reader; anything else is counted and skipped.

### Broadcast archive

`evaluation/archive.py` is WXYC-specific. WXYC's archive holds one MP3 per hour, keyed `YYYY/MM/DD/YYYYMMDDHH00.mp3` by the hour's America/New_York local time, not UTC. `hour_key()` maps a timezone-aware instant to its key and returns `None` for the fall-back hour, which is recorded twice under one key with the later recording overwriting the earlier; `hour_start()` is its inverse. `fetch()` downloads an hour to `<archive_dir>/<key>` through a `.part` file, keeps an existing file of the right size, and never overwrites one of any other size; callers pass `data_dir() / "archive"` as `archive_dir`; `fetch()` and `fetch_all()` raise `DataPathError` for a relative path or one inside the checkout, before any request. `fetch_all()` skips and reports per-hour failures and stops on anything that would fail every hour, such as expired credentials.

### Plays

`evaluation/corpus.py` is WXYC-specific. Its join keys come from `evaluation/names.py`, a station-neutral module that holds the tag-normalization rules (`fold`, `album_key`, `fuzzy`, and the version-qualifier rules) and imports nothing from the WXYC modules. It reads the two CSVs that `evaluation/sql/flowsheet-export.sql` writes (run in a read-only `psql` session into a dated export directory, never over an earlier one) and the reference pool's `pool.db`. `Flowsheet.load()` orders rows by `(add_time, id)`, computes `ETL_STOP`, and labels each show's `play_order` as single-writer (with a reorder flag) or unreliable. `hour_stats()` summarizes every archive hour, and `select_corpus()` chooses the evaluation corpus from it. Eligible DJ hours have at least 8 track rows, a median gap of at least 90 s between logged tracks (shorter means batch logging), and fall outside the August 2026 gap-import days. Hours are banded by their Eastern start: overnight 00:00–05:59, daytime 06:00–17:59, evening 18:00–23:59. The corpus is 20 hours: 12 `canonical` hours with the highest in-pool share (band quotas 5 daytime, 4 evening, 3 overnight), 4 `canonical` hours with an in-pool share of at most 10% ranked by track rows (2, 1, 1), and 4 contrast hours from 2022–2024 (the best in-pool share from each year, then the best remaining). The contrast hours are capped at one per broadcast (`show_id`) and one per recurring slot. A weekly program is one `show_id` per broadcast and the export has no DJ column, so a recurring program is approximated by its slot: the Eastern year, weekday, and hour of the show's first track row. The slot is a proxy for distinct programs, not a guarantee, and it does not identify DJs. A program whose first track lands in a neighboring hour on some week gets a different slot that week, a program that keeps its slot across years can supply one contrast hour per year, a DJ can hold two slots, and a slot can change hands within a year. An hour belongs to the show and slot of every track row in it, so an hour spanning two uses up both, and if too few qualify the shortfall is reported rather than the caps relaxed. There is no talk-hour group: WXYC plays music every hour, and its `talkset` rows are DJ mic breaks inside music hours, which each play's `talk_rows` still counts. Hours holding a reorder-flagged show rank last. A band that cannot fill its quota is filled from the same group's other bands, never from the `etl` era, and every shortfall is reported in the result. `select_hours()` keeps the Phase 1 rule, in-pool count to a target. `write_plays()` writes `plays.jsonl` for a list of hour keys, with each matched play's pool file format as `pool_format`, and refuses to overwrite an existing file, a relative path, or one inside the checkout (`DataPathError`, before anything is created). It opens `pool.db` read-only, so a mistyped path fails instead of creating an empty database. It reads no environment setting, so it needs no `eval.env`.

Freeze the selection once, before any query is spent, with `select`. It runs `hour_stats()` and `select_corpus()` and writes `hours.txt` (every selected hour key, one per line, in selection order: the shape `--hours` and the Shazam CLI read), `subset.txt` (the four-hour subset), and `selection.json` (each hour's group, band, subset membership, and the `in_pool` and `track_rows` counts that ranked it, every shortfall, the export directory's name, and the absolute resolved `pool.db` path: a record for people and the report, not a boundary artifact). The subset is plan section 5.2's: the top-ranked `canonical-high` hour and the top-ranked one from a different band, the top-ranked `canonical-low` hour, and the top-ranked contrast hour, each by the corpus's own ranking (reorder-flagged hours last). A subset position with no candidate is a `subset/<group>` shortfall in `selection.json` and is never filled from another group. `select` refuses to overwrite any of the three files (and then creates none of the others), and refuses an `--out-dir` that is relative or inside the checkout before it reads the export or the pool; `--out-dir` defaults to the data directory. Then write the plays for the frozen hours:

```sh
DATA="${STREAM_SLEUTH_DATA_DIR:-$HOME/.local/share/stream-sleuth}"
# select ranks against $DATA/pool.db: the Shazam leg freezes the selection before the
# Olaf snapshot (and its own pool.db) exists, so that is the only basis there is yet.
uv run --extra eval python -m evaluation.corpus select \
    --export "$DATA/exports/<date>" --pool-db "$DATA/pool.db" --out-dir "$DATA"
# plays are written from the snapshot's pool.db, the one the study scores against.
uv run --extra eval python -m evaluation.corpus \
    --export "$DATA/exports/<date>" --pool-db "$DATA/olaf/<snapshot>/pool.db" \
    --selection "$DATA/selection.json" --out "$DATA/plays.jsonl"
```

Run order: `select` (before the first Shazam request) and the Shazam legs (see Shazam below), then the snapshot build, then this plays command with `--selection`, then the Olaf legs. `select` is never re-run once the Shazam leg has started. **The plays command's warning that `selection.json` was made from another `pool.db` is expected** when plays come from the snapshot's `pool.db`: `selection.json` records the `$DATA/pool.db` that ranked it. It is followed by the basis check's own warning, which names every hour whose `track_rows` or `in_pool` moved from the counts `select` recorded; read it, because those hours' ranking basis is not the snapshot's.

`--selection` reads the hours and their labels from `selection.json`; it and `--hours` are mutually exclusive. Labels come only from `--selection`: `--hours` takes any file of hour keys, such as `hours.txt` or `subset.txt`, but it is unlabelled, so its records carry `group: null` and `subset: false` even for the subset hours. To write plays for just the four subset hours with their labels, add `--subset-only` to a `--selection` run (it is an error without `--selection`). Every play record, carryover plays included, carries `group` (`canonical-high`, `canonical-low`, `contrast`, or `null` for a `--hours` run), `band` (`overnight`, `daytime`, `evening`; the hour key's own band for a `--hours` run), and `subset` (true for the four subset hours of a `--selection` run, else false), so the report can slice recall by group, band, and subset. A `selection.json` with no `hours` object, a label outside those sets, a `band` that is not the hour key's own band (a label's `band` is stamped on every play of the hour, so it must equal the hour key's own), or a non-boolean `subset` (or a file that is not JSON, such as `hours.txt`) exits with one line naming the file and the problem, before anything is written. A `selection.json` made from a different export directory or `pool.db` than `--export` and `--pool-db` still runs, with a warning, because a newer export may cover the same hours. After writing, a `--selection` run recomputes each written hour's non-carryover track count and in-pool count and logs one warning naming every hour whose counts differ from the ones `select` recorded, with both values (the join rules, the pool, or the export moved the ranking basis; a warning, never a refusal, and the hours written do not change). A `selection.json` made before counts were recorded runs with one warning that its basis cannot be checked. `--out` is checked (relative or inside the checkout is refused) before the export or the pool is read.

### Clips

`evaluation/clips.py` is station-neutral: it reads hour files only. A clip is addressed by `(hour key, grid offset, capture length, codec profile)`, and recognizer results are stored under its key, `<hour key>#<offset>+<length>@<profile>` (for example `2026/08/12/202608121600.mp3#45+12@128k`). The offset and length are plain decimal integers, so every address has exactly one key and `ClipAddress.parse()` accepts that key and no other spelling (no leading zeros, decimals, signs, or non-ASCII digits). An hour key must be non-empty, printable, and free of the separators `#`, `+`, and `@`.

The grid is every 15 s from 0, keeping exactly the offsets whose clip ends at or before the hour's end: a full 3,600 s hour holds 240 clips of 6 s or 12 s and 239 of 20 s, since a 20 s clip at 3,585 s would end at 3,605 s. `hour_duration()` measures an hour file by decoding it (about 2 s per hour, cached per file version); pass it as `grid(..., hour_s=hour_duration(path))` for an hour that may be short. `hour_addresses(keys, archive_dir, length_s, profile)` does that for a list of hour keys, in order: an hour that is missing or unreadable is logged and skipped, so every grid leg builds its addresses the same way.

`cut()` is a context manager. It refuses a work directory that is relative or inside the checkout (`DataPathError`, before creating anything), and refuses with `ClipError` any address whose clip would run past the hour file's decoded end, rather than yield a short or empty clip. Otherwise it cuts the clip in its own temporary directory under the work directory, re-encodes it to constant-bitrate MP3 at the profile's rate (`128k`, the live mount, or `320k`), optionally decodes that to the mono 16 kHz WAV the live capture produces, and deletes it on exit, including when the caller raises or ffmpeg fails. Clips are never kept: an address is a recipe over a retained hour file.

### Shazam

`evaluation/shazam_eval.py` is station-neutral. It queries Shazam once per clip address and appends each answer to a JSONL store that is never cleared or rewritten. Shazam is unofficial and rate-limits, so every request is throttled before it is sent: at most `STREAM_SLEUTH_SHAZAM_RATE_PER_DAY` per UTC day (default 500), at least `STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S` apart (default 20). The count, the last request time, and any stop are kept in a state file, and each request is counted there before it is sent, so a restart cannot exceed the day's budget or send sooner than the interval allows. **One run per state file:** a run holds an exclusive lock on `<state>.lock` until it ends, and a second run on the same state file is refused at once (`ThrottleBusyError`, naming the file) before it decodes or sends anything. The budget is the account's, so every leg (the cross-check, the full corpus, each subset) uses the same `--state` and runs in turn; giving a leg its own `--state` gives it a second daily budget. A state file whose last request is more than the interval in the future (a clock stepped back, or a file written by a clock running ahead) is refused with one `FutureStateError` (a `ValueError`) naming the file and both times, and never slept on: the CLI still logs its `stopped: refused` line and the out-of-retries summary, prints the message (wait, or correct the clock; deleting the state file resets the day's count and clears any stop, so it is not the fix), and exits 1; a shorter wait that still exceeds the interval is logged before sleeping. A 429 stops the run for the rest of the UTC day, even when its body fails to arrive; a 403 does the same with the stop reason `forbidden`, after that one request. So do 20 non-scoring outcomes in a row (`MAX_FAILURE_STREAK`), so a systemic failure other than those cannot spend the whole daily budget. An address is retried at most `MAX_RETRIES` (3) times after its first non-scoring outcome (a 429 or 403 is the day's, not the address's, and does not count); after that it is not queried, and is listed in the run's end summary so a person can look. An address is queried once per run however many times its hour is listed (a repeated hour key in `--hours` is gridded once, with a warning), and an address whose latest answer was a 429 or 403 is queried after the rest, so it cannot open every day's queue. A 2xx whose body is not JSON, or JSON of an unexpected shape, is a `decode_error`. A record torn by a crash mid-write is logged and skipped when the store is read, and the next record starts on a new line. Each answer is stored as `matched` or `no_match` (scoring outcomes, never queried again), or `rate_limited`, `server_error` (a 403 is stored as one, with its status), or `decode_error` (queried again by a later run, up to the retry cap). Each record carries the four wire fields, Shazam's match offset (`offset_s`), the HTTP status, and the recognizer identity `shazam@<shazamio version>, segment=<capture length>`; the whole clip is fingerprinted.

Each hour's addresses come from its decoded length (`clips.hour_addresses()`, see Clips), so a short hour is queried only at the offsets that fit. A missing or unreadable hour, or a clip that cannot be cut, is logged with its hour or address and skipped: it is never stored and spends no query, so one bad hour does not stop the day's run. The result store, the throttle state, and the clip work directory are refused before any request if relative or inside the checkout (`DataPathError`); `--store`, `--state`, and `--work-dir` default to `shazam/results.jsonl`, `shazam/throttle.json`, and `clips/` under the data directory.

```sh
uv run --extra eval --env-file "$STREAM_SLEUTH_DATA_DIR/eval.env" python -m evaluation.shazam_eval \
  --hours "$STREAM_SLEUTH_DATA_DIR/hours.txt" --archive-dir "$STREAM_SLEUTH_DATA_DIR/archive"
```

Rerun the same command on later days; it skips every address with a scoring outcome.

### Running the legs

`evaluation/run.py` runs the study's Shazam legs from the frozen `selection.json`, in this order: 12 s on the full corpus, then 6 s, 20 s, and 12 s @320k on the four-hour subset (the hours with `subset: true`). It chooses no hours itself: `hours.txt` and `subset.txt` are conveniences for people, and every leg reads `selection.json`. All legs run inside one `Throttle` on `shazam/throttle.json`, so they share one daily budget (`STREAM_SLEUTH_SHAZAM_RATE_PER_DAY` and `STREAM_SLEUTH_SHAZAM_MIN_INTERVAL_S`, read by `shazam_eval.budget_from_env()` for this command and `shazam_eval` alike). A leg that stops the day (`rate_limited`, `forbidden`, `daily_cap`, or `failure_streak`) leaves the later legs `not started`, and the end-of-run report names each leg's outcome, then the requests sent and the addresses out of retries. A future-dated throttle state (see Shazam above) is logged and reported as that leg's `refused`, the later legs are `not started`, and the command exits 1. Resume, retries, and stop reasons are `shazam_eval`'s. The two 12 s legs file results under one recognizer identity, so a reader tells them apart by the address's `@128k` or `@320k`. It refuses to start with less than 5 GiB free in the clip work directory, and checks `--work-dir`, `--archive-dir`, `--store`, and `--state` (defaults under the data directory) before any request. A `selection.json` that is missing, is not JSON (such as `hours.txt` passed by mistake), or is not an object; whose `hours` is not a non-empty object keyed by hour; with an hour whose `subset` is not `true` or `false` (a hand-edited `"false"` would be truthy); or with no hour whose `subset` is `true` (the subset legs would report `done` having sent nothing) exits with one line naming the file, and the hour when one is at fault, before anything is written.

```sh
uv run --extra eval --env-file "$DATA/eval.env" python -m evaluation.run            # every leg
uv run --extra eval --env-file "$DATA/eval.env" python -m evaluation.run --legs 12s  # one leg
```

## Notes

- Only ASCII/UTF-8 metadata is sent; the API stores it in a `utf8mb4` table.
- The shared secret is the only auth on the ingest endpoint — keep `.env` and the
  plist readable only by the service account. Optionally also restrict the API's
  `/api/shazam` route to this iMac's static IP at the Apache/nginx layer.
