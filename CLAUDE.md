# stream-sleuth

Live-stream song recognition for college radio. `recognizer.py` captures a few seconds of a station's Icecast stream with `ffmpeg`, identifies it with the unofficial Shazam client `shazamio`, and POSTs song changes to the station's API so live DJs get one-click "now playing" suggestions. WXDU runs it in production on an iMac (`run.sh`, `com.wxdu.stream-sleuth.plist`); its behavior must not change unless a PR says so on purpose.

## Interpreter

`requires-python = ">=3.10,<3.13"`, and `.python-version` pins 3.12.

- The floor is what `shazamio` enforces.
- The cap exists because `shazamio` pins `shazamio-core==1.1.2`, whose macOS wheels cover cp39–cp312 only, and because its dependency `pydub` imports `audioop`, which was removed in 3.13. A machine whose default `python3` is 3.13 or later cannot build WXDU's venv with `python3 -m venv venv`; use `uv`, which honors `.python-version`.
- WXDU's deploy runs `venv/bin/python recognizer.py` from `requirements.txt` and never pip-installs this package, so the cap cannot affect it.

## Install and checks

`uv.lock` is committed, and every install goes through it:

```sh
uv sync --locked --extra dev --extra eval
uv run ruff check .
uv run ruff format --check .
uv run mypy . --ignore-missing-imports
uv run pytest
```

These are exactly the CI jobs; run them before every push. Do not document or use `pip install -e`, which drifts from the lock. `--locked` fails instead of silently rewriting `uv.lock` when `pyproject.toml` has changed; run `uv lock` deliberately and commit the result. `requirements.txt` is WXDU's install and is the one file not driven by the lock. It is not pinned (`shazamio>=0.8`), so a WXDU venv rebuild can pick up a newer `shazamio` than the lock tests.

## Tests

- `testpaths = ["tests"]`, `pythonpath = ["."]`. Every test directory has an `__init__.py`.
- pytest markers name **infrastructure CI must provision** (a binary, a service, an external API), following the org scheme in [`WXYC/wiki` `patterns/test-patterns.md`](https://github.com/WXYC/wiki/blob/main/patterns/test-patterns.md); `slow` is a cost dimension only. Directories say tier, markers say infrastructure.
- **A marker lands in the PR that adds the first test using it, never earlier**: declared in `pyproject.toml`, excluded in `addopts` as `not <marker>`, and given a same-named CI job, all at once. pytest exits 5 when a job collects nothing, so an empty job is a red build.
- CI quotes the expression, `pytest -m "<marker>"`; the marker-sync check only recognizes a quoted `-m` argument.
- The default CI job runs plain `pytest` and never names a subdirectory, so every unmarked test runs.
- `recognizer.py` reads every `WXDU_*` variable into a module constant at import. A test that sets the environment must delete every `WXDU_*` and `STREAM_SLEUTH_*` variable, remove `recognizer` and every `stream_sleuth*` entry from `sys.modules`, set its own values, and import afresh. Never `importlib.reload`. See `tests/test_smoke.py`.
- Fixtures are synthetic. Example artists are ones a freeform college station actually plays (Juana Molina, Jessica Pratt, Chuquimamani-Condori, Hermanos Gutiérrez), never mainstream ones.

## Data

Audio, reference-pool listings and object keys, playlist exports, credentials, and result stores never enter the repo. Research data lives under `$STREAM_SLEUTH_DATA_DIR` (default `~/.local/share/stream-sleuth/`), outside the checkout. `data/`, `eval.env`, and `.env.eval` are in `.gitignore` as a second guard only; nothing relies on them.

## S3 and credentials

- **Every S3 client comes from `evaluation.s3_readonly`** (`archive_client()`, `pool_client()`), whose `before-call.s3` handler refuses any operation but `ListObjectsV2`, `GetObject`, and `HeadObject`. `tests/import_scan.py` enforces it over all first-party code: an AST scan forbids importing `boto3`, `botocore`, or `s3transfer` anywhere else, and a textual scan forbids S3 write and presign method names and dynamic imports. The only exemptions are the factory and `tests/unit/test_s3_readonly.py`; do not add more.
- Guard tests use moto's `mock_aws`, never `botocore.stub.Stubber`, which answers before the guard runs and would make the tests pass without it. A test against a custom endpoint sets `MOTO_S3_CUSTOM_ENDPOINTS`, or moto lets botocore reach the real host.
- The harness reads `os.environ` only. Its settings live in `$STREAM_SLEUTH_DATA_DIR/eval.env`, outside the checkout, and are loaded into one process with `uv run --extra eval --env-file …` or a subshell, **never `source`d into an interactive shell**: `run.sh` exports its whole environment into the recognizer.

## Formatting

`recognizer.py` is excluded from `ruff format --check` until characterization tests pin its behavior; the reformat then lands as its own commit that passes the same tests. `ruff check` already covers it.

## WXYC fork

This section applies to the `WXYC/stream-sleuth` fork and is dropped from anything offered upstream to `landmarco/stream-sleuth`.

- Every PR is created with `gh pr create --repo WXYC/stream-sleuth --base <base>`, so a stacked PR can never default to the upstream repo. Merge with rebase only.
- **`spike/` branches are local-only and are never pushed.** This fork is public.
- The `marker-sync` CI job calls `WXYC/wxyc-etl/.github/workflows/check-ci-marker-sync.yml@gha/v1`; read that repo's tag-stability policy before changing the call.
- The viability-study plan lives in the private `wxyc-workspace` repo at `plans/stream-sleuth/plan.md`; the public summary is the epic, WXYC/stream-sleuth#1.
