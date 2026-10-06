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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class FfmpegStub:
    """A fake ``ffmpeg`` kept under ``root``: the executable, its call records, its failure list.

    Call :meth:`install`, then put :attr:`bin_dir` first on ``PATH``. Call *n*
    writes ``root/ffmpeg_calls/<n>.argv``, one argument per line.
    """

    root: Path

    @property
    def bin_dir(self) -> Path:
        return self.root / "bin"

    @property
    def _calls_dir(self) -> Path:
        return self.root / "ffmpeg_calls"

    @property
    def _fail_file(self) -> Path:
        return self.root / "ffmpeg_fail"

    def install(self) -> FfmpegStub:
        """Write the executable into :attr:`bin_dir` and return ``self``."""
        self.bin_dir.mkdir(exist_ok=True)
        stub = self.bin_dir / "ffmpeg"
        stub.write_text(
            _STUB.format(
                python=sys.executable, calls=str(self._calls_dir), fail=str(self._fail_file)
            )
        )
        stub.chmod(0o755)
        return self

    def fail(self, *call_numbers: int) -> None:
        """Make the listed 1-based calls exit 1 after recording their argv, as a failed capture would."""
        self._fail_file.write_text(" ".join(str(n) for n in call_numbers))

    def calls(self) -> list[list[str]]:
        """Read back the argv records, in call order."""
        files = sorted(self._calls_dir.glob("*.argv"), key=lambda p: int(p.stem))
        return [f.read_text().splitlines() for f in files]
