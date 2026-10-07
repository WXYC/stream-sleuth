"""Test doubles for the Shazam leg, shared by every suite that drives it.

``FakeShazam`` is a localhost server, so no test contacts Shazam: pass its ``url`` as
``CountingClient``'s required ``base_url``. ``write_tone`` writes the synthetic audio
``shazamio-core`` fingerprints offline, so ``Shazam.recognize`` runs its real path up to
the HTTP request.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import threading
import wave
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class FakeShazam:
    """A localhost server that answers each POST with the next scripted (status, body, type).

    A fourth element overrides the declared Content-Length, so a body can end short.
    """

    def __init__(self, responses: list[tuple]) -> None:
        self.responses = list(responses)
        self.requests: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append(self.path)
                status, body, ctype, *length = outer.responses.pop(0)
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(length[0] if length else len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def json_response(status: int, body: object) -> tuple[int, bytes, str]:
    return status, json.dumps(body).encode(), "application/json"


HTML_429 = (429, b"<html><body>Too Many Requests</body></html>", "text/html")


def write_tone(path: Path, seconds: int = 12) -> None:
    """A mono 16 kHz WAV of two tones."""
    rate = 16000
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack(
                    "<h",
                    int(
                        8000 * math.sin(2 * math.pi * 440 * i / rate)
                        + 3000 * math.sin(2 * math.pi * 1234 * i / rate)
                    ),
                )
                for i in range(rate * seconds)
            )
        )


class FakeClock:
    """Wall clock plus a recording async sleep that advances it."""

    def __init__(self, start: float) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        await asyncio.sleep(0)  # yield, as a real sleep would, before time moves on
        self.now += seconds


def clock_at(iso: str) -> FakeClock:
    return FakeClock(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp())
