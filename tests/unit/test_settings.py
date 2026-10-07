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


# VERBOSE is a boolean that defaults to off, so one (new, alias) pair cannot make every
# row above fail in the direction it exists for; each case here differs from the default
# or from the value the wrong precedence would give.
@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"STREAM_SLEUTH_VERBOSE": "1"}, True),
        ({"WXDU_VERBOSE": "1"}, True),
        ({"STREAM_SLEUTH_VERBOSE": "0", "WXDU_VERBOSE": "1"}, False),
        ({"STREAM_SLEUTH_VERBOSE": "", "WXDU_VERBOSE": "1"}, True),
    ],
    ids=[
        "new-name-alone",
        "wxdu-alias-alone",
        "new-name-wins-over-alias",
        "empty-new-name-falls-back",
    ],
)
def test_verbose_reads_the_new_name_then_the_wxdu_alias(fresh_recognizer, env, expected):
    assert fresh_recognizer(**env).VERBOSE is expected


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


@pytest.mark.parametrize(
    "output",
    [{}, {"STREAM_SLEUTH_OUTPUT": "http"}, {"STREAM_SLEUTH_OUTPUT": ""}],
    ids=["unset", "http", "empty-counts-as-unset"],
)
def test_http_output_with_a_secret_posts(start, output):
    calls = start(STREAM_SLEUTH_SHAZAM_SECRET="not-a-real-secret", **output)
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
            {"STREAM_SLEUTH_OUTPUT": "jsonl", "STREAM_SLEUTH_OUTPUT_PATH": "wxyc-emissions.jsonl"},
            "STREAM_SLEUTH_OUTPUT_PATH must be an absolute path, not 'wxyc-emissions.jsonl';"
            " refusing to run.\n",
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


def test_refuses_a_jsonl_path_whose_directory_is_missing(start, tmp_path, capsys):
    missing = tmp_path / "missing"

    with pytest.raises(SystemExit) as exit_info:
        start(STREAM_SLEUTH_OUTPUT="jsonl", STREAM_SLEUTH_OUTPUT_PATH=str(missing / "e.jsonl"))

    assert exit_info.value.code == 1
    assert capsys.readouterr().err == (
        f"STREAM_SLEUTH_OUTPUT_PATH's directory {missing} does not exist; refusing to run.\n"
    )
