"""A self-hosted fingerprint index: Olaf (https://github.com/JorenSix/Olaf), run as a subprocess.

Each index is a *snapshot* with its own directory, which the adapter passes to Olaf as
``HOME``, so two snapshots (or a test's ``tmp_path``) never share a database and the
real ``~/.olaf`` is never touched. References are stored with ``olaf store --with-ids``,
so every match names the caller's identifier rather than a file path that may be gone.

A snapshot lives under ``$STREAM_SLEUTH_DATA_DIR/olaf/<snapshot>/``; the caller passes
that path, since this module reads no data-directory setting. It must be absolute, and
neither the real home (whose ``~/.olaf`` is never touched) nor inside the checkout.
Only ``store`` creates a snapshot: querying one with no index raises ``OlafError``, so a
mistyped path fails instead of scoring as an empty index.

Olaf reports only that identifier. ``lookup`` maps it to the four wire keys (a station's
tag database, say); without one, the identifier is the song and the other keys are empty.
"""

import json
import os
import signal
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from ..config import OLAF_BIN
from ..paths import DataPathError, require_outside_checkout
from .base import EvalIdentification, Identification, Recognizer

# Phase 1 of the viability study (2026-10-06): over 2,400 12 s clips, Olaf's own
# thresholds passed stray matches at match_count 6-10 while real songs scored 17-178;
# a floor of 12 removed the noise and lost no identification.
DEFAULT_MIN_MATCH_COUNT = 12

# A query takes milliseconds; a hung one must not stall the live loop forever.
DEFAULT_QUERY_TIMEOUT_S = 60.0

# Written into each snapshot so Olaf never falls back to a config beside its binary.
SNAPSHOT_CONFIG = {"db_folder": "~/.olaf/db/", "cache_folder": "~/.olaf/cache/"}

SNAPSHOT_CONFIG_PATH = Path(".olaf", "olaf_config.json")

# The LMDB file Olaf's first successful store creates in that db_folder.
SNAPSHOT_INDEX = Path(".olaf", "db", "data.mdb")

# Pairs per ``olaf store`` call: a whole reference pool on one argv would pass ARG_MAX.
STORE_BATCH = 200


class OlafError(RuntimeError):
    """Olaf failed or printed no query result, or the snapshot directory is unusable."""


class OlafRecognizer(Recognizer):
    """Identifies a clip against one Olaf snapshot, and fills that snapshot with ``store``.

    ``home`` is the snapshot directory: absolute, not the home directory, and outside
    the checkout, else ``OlafError``. ``olaf_bin`` defaults to ``STREAM_SLEUTH_OLAF_BIN``.
    A match below ``min_match_count`` is ignored. ``lookup`` maps an identifier to the
    four wire keys; any key it leaves out falls back to the identifier-as-song default.
    A query running past ``query_timeout_s`` is killed with its decoder and raises
    ``OlafError``.
    """

    def __init__(
        self,
        home: str | Path,
        olaf_bin: str | None = None,
        min_match_count: int = DEFAULT_MIN_MATCH_COUNT,
        lookup: Callable[[str], Identification | None] | None = None,
        query_timeout_s: float = DEFAULT_QUERY_TIMEOUT_S,
    ) -> None:
        self.home = _snapshot_home(home)
        self.olaf_bin = olaf_bin or OLAF_BIN
        self.min_match_count = min_match_count
        self.lookup = lookup
        self.query_timeout_s = query_timeout_s

    def recognize(self, wav_path: str) -> EvalIdentification | None:
        """The strongest match at or above the floor, or None; ``OlafError`` on failure.

        The snapshot must already hold its config and an index: a query never creates one.
        """
        if not all((self.home / f).is_file() for f in (SNAPSHOT_CONFIG_PATH, SNAPSHOT_INDEX)):
            raise OlafError(f"no Olaf index in {self.home}; fill it with `index build` first")
        matches = [
            m
            for m in parse_matches(
                self._run("query", "--format", "json", wav_path, timeout=self.query_timeout_s)
            )
            if m["match_count"] >= self.min_match_count
        ]
        if not matches:
            return None
        best = max(matches, key=lambda m: m["match_count"])
        found = self.lookup(best["ref_key"]) if self.lookup else None
        names = {"artist": "", "song": best["ref_key"], "album": "", "label": "", **(found or {})}
        return {
            "artist": names["artist"],
            "song": names["song"],
            "album": names["album"],
            "label": names["label"],
            "source": "local",
            "confidence": float(best["match_count"]),
            "query_offset_s": best["query_offset_s"],
            "ref_start_s": best["ref_start_s"],
            "ref_key": best["ref_key"],
        }

    def store(self, items: Iterable[tuple[str, str]]) -> None:
        """Index ``(audio path, identifier)`` pairs, ``STORE_BATCH`` per Olaf call.

        Creates the snapshot and its config if absent. Olaf skips identifiers it already
        holds, so rerunning after an ``OlafError`` stores only what is missing.
        """
        pairs = list(items)
        if not pairs:
            return
        config = self.home / SNAPSHOT_CONFIG_PATH
        try:
            if not config.exists():
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_text(json.dumps(SNAPSHOT_CONFIG, indent=2) + "\n")
        except OSError as exc:
            raise OlafError(f"cannot create the snapshot in {self.home}: {exc.strerror}") from exc
        for i in range(0, len(pairs), STORE_BATCH):
            batch = [arg for pair in pairs[i : i + STORE_BATCH] for arg in pair]
            self._run("store", "--with-ids", *batch)

    def _run(self, *args: str, timeout: float | None = None) -> str:
        # A timed run gets its own process group, so a timeout also kills the ffmpeg
        # Olaf decodes with; an untimed store stays in the caller's, so Ctrl-C reaches it.
        try:
            proc = subprocess.Popen(
                [self.olaf_bin, *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,  # Olaf's ffmpeg must not read the terminal
                text=True,
                env={**os.environ, "HOME": str(self.home)},
                start_new_session=timeout is not None,
            )
        except OSError as exc:
            raise OlafError(f"cannot run {self.olaf_bin}: {exc.strerror}") from exc
        with proc:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except BaseException as exc:
                # A timeout or an interrupt must not leave Olaf and its ffmpeg running.
                if timeout is not None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (PermissionError, ProcessLookupError):
                        proc.kill()  # Olaf exited just as the kill fired
                if not isinstance(exc, subprocess.TimeoutExpired):
                    raise
                proc.communicate()
                raise OlafError(f"olaf {args[0]} timed out after {timeout} s") from exc
        if proc.returncode != 0:
            # Olaf prints argument-parsing errors to stdout, not stderr.
            reason = stderr.strip() or stdout.strip()
            raise OlafError(f"olaf {args[0]} exited {proc.returncode}: {reason[-500:]}")
        return stdout


def _snapshot_home(home: str | Path) -> Path:
    path = Path(home)
    try:
        resolved = require_outside_checkout(path)
    except DataPathError as exc:
        raise OlafError(f"the snapshot directory: {exc}") from exc
    # By file identity, not spelling: a mis-cased or firmlinked home is still the home.
    if resolved.exists() and os.path.samefile(resolved, Path.home()):
        raise OlafError(
            f"the snapshot directory {path} is the home directory, whose ~/.olaf is never touched"
        )
    return path


def parse_matches(output: str) -> list[dict]:
    """Every match in ``olaf query --format json`` output, with clip-relative offsets.

    Olaf prints one pretty-printed object per query (several for ``--fragmented`` or
    several files), each with its fragment's ``query_offset``; a match begins at
    ``query_offset + query_start`` seconds into the clip.
    """
    decoder = json.JSONDecoder()
    matches: list[dict] = []
    pos, found = 0, False
    while (pos := _skip_space(output, pos)) < len(output):
        try:
            obj, pos = decoder.raw_decode(output, pos)
        except json.JSONDecodeError as exc:
            raise OlafError(f"unreadable olaf output: {output[:200]!r}") from exc
        if not isinstance(obj, dict) or "matches" not in obj:
            raise OlafError(f"not a query result: {output[:200]!r}")
        found = True
        try:
            matches += [
                {
                    "match_count": m["match_count"],
                    "query_offset_s": obj.get("query_offset", 0.0) + m["query_start"],
                    "ref_start_s": m["reference_start"],
                    "ref_key": m["path"],
                }
                for m in obj["matches"]
            ]
        except (KeyError, TypeError) as exc:
            raise OlafError(f"incomplete match in olaf output: {exc!r}") from exc
    if not found:
        raise OlafError("olaf printed no query result")
    return matches


def _skip_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos
