"""A recording ``ffmpeg`` stub on ``PATH``, modeled on library-metadata-lookup's ``tests/curl_stub.py``.

``recognizer.capture()`` runs ``ffmpeg`` as a subprocess to grab a few seconds of
the stream. The characterization suite runs the real ``capture()`` and the real
``subprocess.run`` against this stub instead, so the exact arguments are pinned
without a network or an encoder.

The stub writes a short, valid mono 16 kHz WAV to the output path (its last
argument), and records **one numbered argv file per call**. Numbered records are
what make the 6 s to 12 s capture escalation assertable as distinct calls rather
than one blurred final state.

Lives at ``tests/`` root because it is a helper module, not a suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

_STUB = """\
#!{python}
import pathlib
import sys
import wave

calls = pathlib.Path({calls!r})
calls.mkdir(exist_ok=True)
n = len(list(calls.glob("*.argv"))) + 1
(calls / f"{{n}}.argv").write_text("".join(arg + "\\n" for arg in sys.argv[1:]))

fail = pathlib.Path({fail!r})
if fail.exists() and str(n) in fail.read_text().split():
    sys.stderr.write("ffmpeg stub: simulated failure\\n")
    sys.exit(1)

with wave.open(sys.argv[-1], "wb") as out:
    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(16000)
    out.writeframes(b"\\x00\\x00" * 1600)
"""


def install_ffmpeg_stub(tmp_path: Path) -> Path:
    """Write a fake ``ffmpeg`` into ``tmp_path/bin`` and return that directory.

    Put the returned directory first on ``PATH``. Call *n* writes
    ``tmp_path/ffmpeg_calls/<n>.argv``, one argument per line. To make call *n*
    exit 1 after recording (as a failed capture would), list *n* in
    ``tmp_path/ffmpeg_fail`` with :func:`fail_ffmpeg_calls`.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "ffmpeg"
    stub.write_text(
        _STUB.format(
            python=sys.executable,
            calls=str(tmp_path / "ffmpeg_calls"),
            fail=str(tmp_path / "ffmpeg_fail"),
        )
    )
    stub.chmod(0o755)
    return bin_dir


def fail_ffmpeg_calls(tmp_path: Path, *call_numbers: int) -> None:
    """Make the listed 1-based stub calls exit 1 after recording their argv."""
    (tmp_path / "ffmpeg_fail").write_text(" ".join(str(n) for n in call_numbers))


def ffmpeg_calls(tmp_path: Path) -> list[list[str]]:
    """Read back the stub's argv records, in call order."""
    calls_dir = tmp_path / "ffmpeg_calls"
    if not calls_dir.is_dir():
        return []
    files = sorted(calls_dir.glob("*.argv"), key=lambda p: int(p.stem))
    return [f.read_text().splitlines() for f in files]
