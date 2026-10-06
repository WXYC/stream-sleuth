"""Pins ``recognizer.parse()``: Shazam response body in, plain dict or ``None`` out."""

from __future__ import annotations

from typing import Any

import pytest

from tests.characterization import shazam_responses as r


@pytest.fixture
def parse(fresh_recognizer):
    return fresh_recognizer(WXDU_SHAZAM_SECRET="x").parse


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        pytest.param(
            r.JUANA_MOLINA,
            {"artist": "Juana Molina", "song": "la paradoja", "album": "DOGA", "label": "Sonamos"},
            id="full-match",
        ),
        pytest.param(
            r.match("Chuquimamani-Condori", "Call Your Name"),
            {"artist": "Chuquimamani-Condori", "song": "Call Your Name", "album": "", "label": ""},
            id="no-album-or-label-rows",
        ),
        pytest.param(
            {"track": {"title": "Back, Baby", "subtitle": "Jessica Pratt"}},
            {"artist": "Jessica Pratt", "song": "Back, Baby", "album": "", "label": ""},
            id="no-sections-key",
        ),
        pytest.param(
            {"track": {"title": None, "subtitle": None, "sections": None}},
            {"artist": "", "song": "", "album": "", "label": ""},
            id="null-fields-become-empty-strings",
        ),
        pytest.param(
            {
                "track": {
                    "title": "la paradoja",
                    "subtitle": "Juana Molina",
                    "sections": [
                        {"metadata": [{"title": "  ALBUM ", "text": "DOGA"}]},
                        {"metadata": [{"title": "Album", "text": "Halo"}, {"title": "label"}]},
                        {"metadata": None},
                    ],
                }
            },
            {"artist": "Juana Molina", "song": "la paradoja", "album": "DOGA", "label": ""},
            id="keys-stripped-and-lowercased-first-value-wins",
        ),
    ],
)
def test_matched_response_becomes_a_plain_dict(parse, response: dict[str, Any], expected):
    result = parse(response)

    assert type(result) is dict
    assert result == expected
    assert list(result) == ["artist", "song", "album", "label"]


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(r.NO_MATCH, id="no-track-key"),
        pytest.param({"track": {}}, id="empty-track"),
        pytest.param({"track": None}, id="null-track"),
        pytest.param(None, id="none"),
        pytest.param([], id="not-a-dict"),
    ],
)
def test_unmatched_response_is_none(parse, response):
    assert parse(response) is None
