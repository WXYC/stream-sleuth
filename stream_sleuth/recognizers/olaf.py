"""A self-hosted fingerprint index: Olaf (https://github.com/JorenSix/Olaf), run as a subprocess.

Each index is a *snapshot* with its own directory, which the adapter passes to Olaf as
``HOME``, so two snapshots (or a test's ``tmp_path``) never share a database and the
real ``~/.olaf`` is never touched. References are stored with ``olaf store --with-ids``,
so every match names the caller's identifier rather than a file path that may be gone.

Olaf reports only that identifier. ``lookup`` maps it to the four wire keys (a station's
tag database, say); without one, the identifier is the song and the other keys are empty.
"""

import json
import os
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from ..config import OLAF_BIN
from .base import EvalIdentification, Identification, Recognizer

# Phase 1 of the viability study (2026-10-06): over 2,400 12 s clips, Olaf's own
# thresholds passed stray matches at match_count 6-10 while real songs scored 17-178;
# a floor of 12 removed the noise and lost no identification.
DEFAULT_MIN_MATCH_COUNT = 12

# Written into each snapshot so Olaf never falls back to a config beside its binary.
SNAPSHOT_CONFIG = {"db_folder": "~/.olaf/db/", "cache_folder": "~/.olaf/cache/"}


class OlafError(RuntimeError):
    """Olaf exited non-zero or printed output that is not a query result."""


class OlafRecognizer(Recognizer):
    def __init__(
        self,
        home: str | Path,
        olaf_bin: str | None = None,
        min_match_count: int = DEFAULT_MIN_MATCH_COUNT,
        lookup: Callable[[str], Identification | None] | None = None,
    ) -> None:
        self.home = Path(home)
        self.olaf_bin = olaf_bin or OLAF_BIN
        self.min_match_count = min_match_count
        self.lookup = lookup

    def recognize(self, wav_path: str) -> EvalIdentification | None:
        matches = [
            m
            for m in parse_matches(self._run("query", "--format", "json", wav_path))
            if m["match_count"] >= self.min_match_count
        ]
        if not matches:
            return None
        best = max(matches, key=lambda m: m["match_count"])
        names = (self.lookup(best["ref_key"]) if self.lookup else None) or {
            "artist": "",
            "song": best["ref_key"],
            "album": "",
            "label": "",
        }
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
        """Index ``(audio path, identifier)`` pairs; Olaf skips identifiers it already holds."""
        args = [arg for pair in items for arg in pair]
        if args:
            self._run("store", "--with-ids", *args)

    def _run(self, *args: str) -> str:
        config = self.home / ".olaf" / "olaf_config.json"
        if not config.exists():
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(json.dumps(SNAPSHOT_CONFIG, indent=2) + "\n")
        proc = subprocess.run(
            [self.olaf_bin, *args],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(self.home)},
        )
        if proc.returncode != 0:
            raise OlafError(
                f"olaf {args[0]} exited {proc.returncode}: {proc.stderr.strip()[-500:]}"
            )
        return proc.stdout


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
        for m in obj["matches"]:
            matches.append(
                {
                    "match_count": m["match_count"],
                    "query_offset_s": obj.get("query_offset", 0.0) + m["query_start"],
                    "ref_start_s": m["reference_start"],
                    "ref_key": m["path"],
                }
            )
    if not found:
        raise OlafError("olaf printed no query result")
    return matches


def _skip_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos
