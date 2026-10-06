"""Smoke test: the recognizer module imports and reads its configuration.

This is the one test PR 0 ships so the default CI job collects something from
the start (pytest exits 5 when a job collects nothing). The characterization
suite that pins WXDU's runtime behavior arrives in its own PR.
"""

import importlib
import os
import sys

import pytest


@pytest.fixture
def fresh_recognizer(monkeypatch):
    """Import ``recognizer`` afresh from a clean ``WXDU_*`` environment.

    ``recognizer.py`` reads every ``WXDU_*`` variable into a module constant at
    import time, so a test that sets the environment needs a fresh import.
    """

    def _import(**env):
        for name in list(sys.modules):
            if name == "recognizer" or name.startswith("stream_sleuth"):
                monkeypatch.delitem(sys.modules, name)
        for name in [n for n in os.environ if n.startswith(("WXDU_", "STREAM_SLEUTH_"))]:
            monkeypatch.delenv(name)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return importlib.import_module("recognizer")

    return _import


def test_reads_configuration_from_the_environment(fresh_recognizer):
    recognizer = fresh_recognizer(
        WXDU_STREAM_URL="https://audio-mp3.ibiblio.org/wxyc.mp3",
        WXDU_SHAZAM_SECRET="not-a-real-secret",
        WXDU_INTERVAL="30",
    )

    assert recognizer.STREAM_URL == "https://audio-mp3.ibiblio.org/wxyc.mp3"
    assert recognizer.API_SECRET == "not-a-real-secret"
    assert recognizer.INTERVAL == 30


@pytest.mark.parametrize("name", ["parse", "capture", "post", "identify_once", "main"])
def test_exports_the_functions_later_refactors_must_keep(fresh_recognizer, name):
    recognizer = fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")

    assert callable(getattr(recognizer, name))
