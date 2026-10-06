"""Each per-format tag reader against a real file: ffmpeg renders, mutagen tags.

ffmpeg's muxers do not write the tag chunks these readers expect (a WAV's RIFF
``id3 `` chunk especially), so every fixture is rendered untagged by ffmpeg and
then tagged with mutagen, the way the station's files were tagged.
"""

from __future__ import annotations

import subprocess

import pytest
from mutagen.easyid3 import EasyID3
from mutagen.easymp4 import EasyMP4
from mutagen.flac import FLAC
from mutagen.id3 import TALB, TIT2, TPE1, TPE2, TRCK
from mutagen.mp3 import EasyMP3
from mutagen.wave import WAVE

from evaluation import pool

pytestmark = pytest.mark.ffmpeg

TAGS = {
    "artist": "Hermanos Gutiérrez",
    "albumartist": "Hermanos Gutiérrez",
    "album": "Sonido Cósmico",
    "title": "Low Sun",
    "tracknumber": "4",
}
ID3_FRAMES = {
    "artist": TPE1,
    "albumartist": TPE2,
    "album": TALB,
    "title": TIT2,
    "tracknumber": TRCK,
}


def _easy(tags):
    for name, value in TAGS.items():
        tags[name] = value
    tags.save()


def _tag_aac(path):
    tags = EasyID3()
    for name, value in TAGS.items():
        tags[name] = value
    tags.save(path)


def _tag_wav(path):
    wav = WAVE(path)
    wav.add_tags()
    for name, frame in ID3_FRAMES.items():
        wav.tags.add(frame(encoding=3, text=TAGS[name]))
    wav.save()


# format -> (file extension, ffmpeg output arguments, tagger)
FORMATS = {
    "mp3": (".mp3", ["-c:a", "libmp3lame", "-b:a", "128k"], lambda p: _easy(EasyMP3(p))),
    "aac": (".aac", ["-c:a", "aac", "-b:a", "128k", "-f", "adts"], _tag_aac),
    "wav": (".wav", ["-c:a", "pcm_s16le"], _tag_wav),
    "flac": (".flac", ["-c:a", "flac"], lambda p: _easy(FLAC(p))),
    "mp4": (".m4a", ["-c:a", "aac", "-b:a", "128k"], lambda p: _easy(EasyMP4(p))),
}


@pytest.mark.parametrize("fmt", FORMATS)
def test_each_reader_returns_the_tags_and_stream_info(tmp_path, fmt):
    extension, args, tag = FORMATS[fmt]
    path = tmp_path / f"fixture{extension}"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-ac",
            "2",
            *args,
            str(path),
        ],
        check=True,
    )
    tag(path)

    assert pool.FORMATS[extension] == fmt
    tags = pool.read_tags(path, fmt)

    assert {k: tags[k] for k in ("artist", "album_artist", "album", "title", "track_number")} == {
        "artist": TAGS["artist"],
        "album_artist": TAGS["albumartist"],
        "album": TAGS["album"],
        "title": TAGS["title"],
        "track_number": TAGS["tracknumber"],
    }
    assert tags["duration_s"] == pytest.approx(3.0, abs=0.2)
    assert tags["bitrate_kbps"] > 0


def test_an_untagged_file_reads_as_no_tags(tmp_path):
    path = tmp_path / "untagged.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=2",
            "-c:a",
            "libmp3lame",
            str(path),
        ],
        check=True,
    )

    tags = pool.read_tags(path, "mp3")

    assert tags["artist"] is None and tags["title"] is None
    assert tags["duration_s"] == pytest.approx(2.0, abs=0.2)
