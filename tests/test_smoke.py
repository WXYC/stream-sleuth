"""Smoke test: the recognizer module imports and reads its configuration.

The fastest signal that the module still imports and still exposes what
``tests/characterization/`` and WXDU's launchd job rely on.
"""

import pytest


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
