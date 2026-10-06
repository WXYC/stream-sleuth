"""Station-neutral settings: ``STREAM_SLEUTH_<NAME>``, with ``WXDU_<NAME>`` as a supported alias."""

from __future__ import annotations

import importlib

import pytest

# (name suffix, constant, raw new value, raw alias value, parsed new value, parsed alias value)
SETTINGS = [
    (
        "STREAM_URL",
        "STREAM_URL",
        "https://audio-mp3.ibiblio.org/wxyc.mp3",
        "https://alias.example/a.mp3",
        "https://audio-mp3.ibiblio.org/wxyc.mp3",
        "https://alias.example/a.mp3",
    ),
    (
        "SHAZAM_API",
        "API_URL",
        "https://ingest.example/new",
        "https://ingest.example/alias",
        "https://ingest.example/new",
        "https://ingest.example/alias",
    ),
    ("SHAZAM_SECRET", "API_SECRET", "new-secret", "alias-secret", "new-secret", "alias-secret"),
    ("INTERVAL", "INTERVAL", "30", "31", 30, 31),
    ("INTERVAL_GAP", "INTERVAL_GAP", "5", "6", 5, 6),
    ("CAPTURE_FAST", "CAPTURE_FAST", "7", "8", 7, 8),
    ("CAPTURE_SLOW", "CAPTURE_SLOW", "14", "15", 14, 15),
    ("VERBOSE", "VERBOSE", "1", "0", True, False),
]


@pytest.mark.parametrize(
    ("env", "expect"),
    [
        (lambda n, new, alias: {f"STREAM_SLEUTH_{n}": new}, "new"),
        (lambda n, new, alias: {f"WXDU_{n}": alias}, "alias"),
        (lambda n, new, alias: {f"STREAM_SLEUTH_{n}": new, f"WXDU_{n}": alias}, "new"),
        (lambda n, new, alias: {f"STREAM_SLEUTH_{n}": "", f"WXDU_{n}": alias}, "alias"),
    ],
    ids=[
        "new-name-alone",
        "wxdu-alias-alone",
        "new-name-wins-over-alias",
        "empty-new-name-falls-back",
    ],
)
@pytest.mark.parametrize(("name", "constant", "new", "alias", "new_value", "alias_value"), SETTINGS)
def test_each_setting_reads_the_new_name_then_the_wxdu_alias(
    fresh_recognizer, env, expect, name, constant, new, alias, new_value, alias_value
):
    recognizer = fresh_recognizer(**env(name, new, alias))

    assert getattr(recognizer, constant) == (new_value if expect == "new" else alias_value)


class LoopStarted(BaseException):
    """Raised instead of running the real loop, so a test never captures or posts."""


@pytest.fixture
def start(fresh_recognizer, monkeypatch):
    """Import afresh with ``env``, then call ``main()`` with ``run`` replaced by a recorder."""

    def _start(**env):
        recognizer = fresh_recognizer(**env)
        loop = importlib.import_module("stream_sleuth.loop")
        calls = []

        def fake_run(source, recognizer_, output, cadence, **kwargs):
            calls.append(output)
            raise LoopStarted

        monkeypatch.setattr(loop, "run", fake_run)
        try:
            recognizer.main()
        except LoopStarted:
            pass
        return calls

    return _start


def test_the_wxdu_default_still_refuses_without_a_secret(start, capsys):
    with pytest.raises(SystemExit) as exit_info:
        start()

    assert exit_info.value.code == 1
    out = capsys.readouterr()
    assert out.err == "WXDU_SHAZAM_SECRET is not set; refusing to run.\n"
    assert out.out == ""


def test_http_output_with_a_secret_posts(start):
    calls = start(STREAM_SLEUTH_SHAZAM_SECRET="not-a-real-secret")
    outputs = importlib.import_module("stream_sleuth.outputs")

    assert [type(o) for o in calls] == [outputs.HttpPostOutput]


def test_jsonl_output_needs_no_secret(start, tmp_path, capsys):
    path = tmp_path / "emissions.jsonl"
    calls = start(STREAM_SLEUTH_OUTPUT="jsonl", STREAM_SLEUTH_OUTPUT_PATH=str(path))
    outputs = importlib.import_module("stream_sleuth.outputs")

    assert [type(o) for o in calls] == [outputs.JsonlOutput]
    assert calls[0].path == path
    assert capsys.readouterr().out.rstrip().endswith(f"-> {path}")


@pytest.mark.parametrize(
    ("env", "message"),
    [
        (
            {"STREAM_SLEUTH_OUTPUT": "jsonl"},
            "STREAM_SLEUTH_OUTPUT_PATH is not set; refusing to run.\n",
        ),
        (
            {"STREAM_SLEUTH_OUTPUT": "carrier-pigeon"},
            "STREAM_SLEUTH_OUTPUT must be http or jsonl, not 'carrier-pigeon'; refusing to run.\n",
        ),
    ],
)
def test_refuses_an_incomplete_or_unknown_output(start, capsys, env, message):
    with pytest.raises(SystemExit) as exit_info:
        start(**env)

    assert exit_info.value.code == 1
    assert capsys.readouterr().err == message
