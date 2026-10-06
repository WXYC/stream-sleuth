"""Station-neutral settings: ``STREAM_SLEUTH_<NAME>``, with ``WXDU_<NAME>`` as a supported alias."""

from __future__ import annotations

import importlib

import pytest

# (name suffix, constant, value under the new name, value under the alias, expected values)
SETTINGS = [
    (
        "STREAM_URL",
        "STREAM_URL",
        "https://audio-mp3.ibiblio.org/wxyc.mp3",
        "https://alias.example/a.mp3",
        None,
    ),
    ("SHAZAM_API", "API_URL", "https://ingest.example/new", "https://ingest.example/alias", None),
    ("SHAZAM_SECRET", "API_SECRET", "new-secret", "alias-secret", None),
    ("INTERVAL", "INTERVAL", "30", "31", (30, 31)),
    ("INTERVAL_GAP", "INTERVAL_GAP", "5", "6", (5, 6)),
    ("CAPTURE_FAST", "CAPTURE_FAST", "7", "8", (7, 8)),
    ("CAPTURE_SLOW", "CAPTURE_SLOW", "14", "15", (14, 15)),
    ("VERBOSE", "VERBOSE", "1", "0", (True, False)),
]


def _expected(new, alias, parsed):
    return parsed if parsed else (new, alias)


@pytest.mark.parametrize(("name", "constant", "new", "alias", "parsed"), SETTINGS)
def test_new_name_alone(fresh_recognizer, name, constant, new, alias, parsed):
    recognizer = fresh_recognizer(**{f"STREAM_SLEUTH_{name}": new})

    assert getattr(recognizer, constant) == _expected(new, alias, parsed)[0]


@pytest.mark.parametrize(("name", "constant", "new", "alias", "parsed"), SETTINGS)
def test_wxdu_alias_alone(fresh_recognizer, name, constant, new, alias, parsed):
    recognizer = fresh_recognizer(**{f"WXDU_{name}": alias})

    assert getattr(recognizer, constant) == _expected(new, alias, parsed)[1]


@pytest.mark.parametrize(("name", "constant", "new", "alias", "parsed"), SETTINGS)
def test_new_name_wins_when_both_are_set(fresh_recognizer, name, constant, new, alias, parsed):
    recognizer = fresh_recognizer(**{f"STREAM_SLEUTH_{name}": new, f"WXDU_{name}": alias})

    assert getattr(recognizer, constant) == _expected(new, alias, parsed)[0]


@pytest.mark.parametrize(("name", "constant", "new", "alias", "parsed"), SETTINGS)
def test_an_empty_new_name_falls_back_to_the_alias(
    fresh_recognizer, name, constant, new, alias, parsed
):
    recognizer = fresh_recognizer(**{f"STREAM_SLEUTH_{name}": "", f"WXDU_{name}": alias})

    assert getattr(recognizer, constant) == _expected(new, alias, parsed)[1]


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
