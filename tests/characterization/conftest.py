"""The harness that pins WXDU's recognizer behavior at seams the refactor does not move.

Four seams, all outside ``recognizer.py`` itself:

1. ``shazamio.Shazam.recognize`` is patched on the third-party class.
2. A recording fake ``ffmpeg`` (``tests/ffmpeg_stub.py``) is first on ``PATH``.
3. A localhost HTTP server is the ingest API (``WXDU_SHAZAM_API``).
4. The global ``time.sleep`` is patched, but only for the loop's own pauses.
   ``capture()`` runs ``ffmpeg`` with a timeout, and CPython's
   ``Popen.wait(timeout=...)`` polls with its own short ``time.sleep`` calls. So
   the interval settings are distinctive sentinels, and only sleeps of exactly
   those lengths are recorded; every other duration goes to the real
   ``time.sleep``. After the requested number of loop pauses the patch raises
   :class:`StopLoop`, which subclasses ``BaseException`` so no
   ``except Exception`` in the loop can swallow it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import shazamio

from tests.ffmpeg_stub import ffmpeg_calls, install_ffmpeg_stub

# Distinctive pause lengths, so a recorded sleep can only be the loop's own.
INTERVAL = 1001
INTERVAL_GAP = 1003
SENTINELS = {INTERVAL, INTERVAL_GAP}

STREAM_URL = "https://stream.example.test/live.mp3"
SECRET = "characterization-secret"


class StopLoop(BaseException):
    """Raised by the patched sleep to end ``main()``'s infinite loop between cycles."""


@dataclass
class IngestRequest:
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


@dataclass
class IngestServer:
    """A localhost stand-in for the station's ingest API."""

    url: str
    requests: list[IngestRequest] = field(default_factory=list)
    # Status codes to answer with, in order; once exhausted every answer is 201.
    statuses: list[int] = field(default_factory=list)


@pytest.fixture
def ingest_server() -> Iterator[IngestServer]:
    server_state = IngestServer(url="")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server's naming
            length = int(self.headers.get("Content-Length", "0"))
            server_state.requests.append(
                IngestRequest(self.path, dict(self.headers.items()), self.rfile.read(length))
            )
            status = server_state.statuses.pop(0) if server_state.statuses else 201
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_state.url = f"http://127.0.0.1:{httpd.server_address[1]}/api/shazam"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield server_state
    finally:
        httpd.shutdown()
        httpd.server_close()


@dataclass
class FakeShazam:
    """Answers ``Shazam.recognize`` from a script of responses (or exceptions)."""

    responses: list[Any] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)


@pytest.fixture
def fake_shazam(monkeypatch: pytest.MonkeyPatch) -> FakeShazam:
    fake = FakeShazam()

    async def recognize(self: shazamio.Shazam, data: Any, *args: Any, **kwargs: Any) -> Any:
        fake.paths.append(str(data))
        response = fake.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(shazamio.Shazam, "recognize", recognize)
    return fake


@dataclass
class LoopSleeps:
    """The loop's own pauses, in order, and how many to allow before stopping."""

    pauses: list[float] = field(default_factory=list)
    stop_after: int = 0


@pytest.fixture
def loop_sleeps(monkeypatch: pytest.MonkeyPatch) -> LoopSleeps:
    recorder = LoopSleeps()
    real_sleep = time.sleep

    def sleep(seconds: float) -> None:
        if seconds in SENTINELS:
            recorder.pauses.append(seconds)
            if len(recorder.pauses) >= recorder.stop_after:
                raise StopLoop
            return
        real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", sleep)
    return recorder


@dataclass
class LoopRun:
    """Everything one ``main()`` run produced."""

    stdout: list[str]
    stderr: list[str]
    pauses: list[float]
    capture_seconds: list[int]
    ffmpeg_argv: list[list[str]]
    posts: list[IngestRequest]


@pytest.fixture
def run_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fresh_recognizer: Callable[..., ModuleType],
    ingest_server: IngestServer,
    fake_shazam: FakeShazam,
    loop_sleeps: LoopSleeps,
) -> Callable[..., LoopRun]:
    """Run ``recognizer.main()`` for one cycle per scripted Shazam response.

    Keyword arguments become extra ``WXDU_*`` environment variables (for example
    ``WXDU_VERBOSE="1"``). ``post_statuses`` scripts the ingest API's answers.
    """
    monkeypatch.setenv("PATH", f"{install_ffmpeg_stub(tmp_path)}{os.pathsep}{os.environ['PATH']}")
    for proxy in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy"):
        monkeypatch.delenv(proxy, raising=False)

    def _run(
        responses: list[Any], *, post_statuses: list[int] | None = None, **env: str
    ) -> LoopRun:
        recognizer = fresh_recognizer(
            WXDU_STREAM_URL=STREAM_URL,
            WXDU_SHAZAM_API=ingest_server.url,
            WXDU_SHAZAM_SECRET=SECRET,
            WXDU_INTERVAL=str(INTERVAL),
            WXDU_INTERVAL_GAP=str(INTERVAL_GAP),
            **env,
        )
        fake_shazam.responses = list(responses)
        ingest_server.statuses = list(post_statuses or [])
        loop_sleeps.stop_after = len(responses)
        with pytest.raises(StopLoop):
            recognizer.main()
        out = capsys.readouterr()
        argv = ffmpeg_calls(tmp_path)
        return LoopRun(
            stdout=out.out.splitlines(),
            stderr=out.err.splitlines(),
            pauses=list(loop_sleeps.pauses),
            capture_seconds=[int(a[a.index("-t") + 1]) for a in argv],
            ffmpeg_argv=argv,
            posts=list(ingest_server.requests),
        )

    return _run
