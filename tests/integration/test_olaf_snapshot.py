"""A snapshot built from a synthetic pool, queried with cuts of a synthetic hour.

Needs ``STREAM_SLEUTH_OLAF_BIN`` (or ``olaf`` on PATH) and ``ffmpeg``; the ``olaf`` CI job
provides both. Seeded pink noise stands in for music. The pool is a moto bucket holding
two references: one tagged the way the station's files are, one with no tags at all.
"""

from __future__ import annotations

import os
import subprocess

import pytest
from moto import mock_aws
from mutagen.id3 import TALB, TIT2, TPE1
from mutagen.wave import WAVE

from evaluation import pool
from evaluation.olaf_snapshot import build_snapshot
from stream_sleuth.loop import Cadence, State, step
from stream_sleuth.recognizers.olaf import OlafRecognizer, snapshot_dir
from tests.unit.test_s3_readonly import seed_objects

pytestmark = pytest.mark.olaf

# Read before aws_isolated_env clears every STREAM_SLEUTH_* variable.
OLAF_BIN = os.environ.get("STREAM_SLEUTH_OLAF_BIN") or "olaf"

ENDPOINT = "https://pool.example.test"
BUCKET = "synthetic-pool"
TAGGED = "rotation/Heavy/juana-molina/doga/01-la-paradoja.wav"
UNTAGGED = "rotation/Light/jessica-pratt/02-back-baby.wav"
CADENCE = Cadence(fast=6, slow=12, interval=23, interval_gap=4)


def noise(seed):
    return ["-f", "lavfi", "-i", f"anoisesrc=d=60:c=pink:seed={seed}:a=0.5"]


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


@pytest.fixture
def synthetic_pool(monkeypatch, aws_isolated_env, tmp_path):
    aws_isolated_env.use_pool(ENDPOINT, BUCKET)
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(tmp_path / "data"))
    references = {}
    for key, seed in ((TAGGED, 11), (UNTAGGED, 22)):
        path = tmp_path / f"{seed}.wav"
        ffmpeg(*noise(seed), "-ac", "1", "-ar", "16000", str(path))
        references[key] = path
    wav = WAVE(references[TAGGED])
    wav.add_tags()
    for frame in (TPE1(encoding=3, text="Juana Molina"), TIT2(encoding=3, text="la paradoja"),
                  TALB(encoding=3, text="DOGA")):  # fmt: skip
        wav.tags.add(frame)
    wav.save()
    with mock_aws():
        seed_objects(ENDPOINT, BUCKET, {k: p.read_bytes() for k, p in references.items()})
        yield


def cut_of_the_hour(tmp_path, start):
    """A 12 s, 128 kbps cut starting ``start`` s into an hour of the two references back to back."""
    clip = tmp_path / f"clip-{start}.mp3"
    ffmpeg(*noise(11), *noise(22), "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1",
           "-ac", "1", "-ar", "16000", "-ss", str(start), "-t", "12",
           "-c:a", "libmp3lame", "-b:a", "128k", str(clip))  # fmt: skip
    return clip


def test_a_match_carries_the_references_tags_and_an_untagged_one_is_a_miss(
    synthetic_pool, tmp_path
):
    counts = build_snapshot("rotation", pool.inventory(["rotation/"]), olaf_bin=OLAF_BIN)
    assert counts == {"indexed": 2, "untagged": 1}
    home = snapshot_dir("rotation")
    db = pool.open_pool_db(home / "pool.db")
    recognizer = OlafRecognizer(home, olaf_bin=OLAF_BIN, lookup=pool.tag_lookup(db))

    tagged = recognizer.recognize(str(cut_of_the_hour(tmp_path, 20)))
    untagged = recognizer.recognize(str(cut_of_the_hour(tmp_path, 80)))

    assert tagged is not None and untagged is not None
    assert tagged["ref_key"] == pool.stage_id(TAGGED)
    assert untagged["ref_key"] == pool.stage_id(UNTAGGED)
    action = step(State(CADENCE.fast), tagged, CADENCE)[1]
    assert action.emit is not None
    assert [action.emit[k] for k in ("artist", "song", "album", "label")] == [
        "Juana Molina",
        "la paradoja",
        "DOGA",
        "",
    ]
    assert step(State(CADENCE.fast), untagged, CADENCE)[1].emit is None
