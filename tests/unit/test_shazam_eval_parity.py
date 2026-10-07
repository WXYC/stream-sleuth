"""The study scores exactly the fields the deployed recognizer posts.

``outcome_from`` must take its four wire fields from ``stream_sleuth.recognizers.shazam.parse``,
the extraction WXDU runs live, so each response here goes through both and the fields are
compared. The first-value-wins and key-normalization cases are the ones where a second copy
of the extraction would drift.
"""

from __future__ import annotations

from typing import Any

import pytest

from evaluation.shazam_eval import outcome_from
from stream_sleuth.recognizers.shazam import parse
from tests.characterization import shazam_responses as r

FIELDS = ("artist", "song", "album", "label")

RESPONSES = [
    pytest.param(r.JUANA_MOLINA, id="full-match"),
    pytest.param(r.JESSICA_PRATT, id="second-full-match"),
    pytest.param(r.match("Chuquimamani-Condori", "Call Your Name"), id="no-album-or-label-rows"),
    pytest.param(
        r.match("Hermanos Gutiérrez", "Tres Hombres", label="Easy Eye Sound"), id="label-only"
    ),
    pytest.param(r.NO_MATCH, id="no-track-key"),
    pytest.param({"track": {}}, id="empty-track"),
    pytest.param({"track": None}, id="null-track"),
    pytest.param(
        {"track": {"title": "Back, Baby", "subtitle": "Jessica Pratt"}}, id="no-sections-key"
    ),
    pytest.param({"track": {"title": None, "subtitle": None, "sections": None}}, id="null-fields"),
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
        id="keys-stripped-and-lowercased-first-value-wins",
    ),
    pytest.param(
        {
            "track": {
                "title": "la paradoja",
                "subtitle": "Juana Molina",
                "sections": [
                    {"metadata": [{"title": "Album", "text": "DOGA"}]},
                    {"metadata": [{"title": "Album", "text": "Halo"}]},
                    {
                        "metadata": [
                            {"title": "Label", "text": ""},
                            {"title": "Label", "text": "Sonamos"},
                        ]
                    },
                ],
            }
        },
        id="duplicate-album-rows-and-an-empty-label-then-a-real-one",
    ),
]


@pytest.mark.parametrize("body", RESPONSES)
def test_the_four_wire_fields_are_what_the_runtime_parse_returns(body: dict[str, Any]) -> None:
    got = outcome_from(200, body)
    expected = parse(body)
    if expected is None:
        assert got.kind == "no_match"
        assert tuple(getattr(got, f) for f in FIELDS) == ("", "", "", "")
    else:
        assert got.kind == "matched"
        assert {f: getattr(got, f) for f in FIELDS} == expected


def test_the_fields_come_from_the_one_runtime_extraction() -> None:
    import evaluation.shazam_eval as shazam_eval

    assert shazam_eval.parse is parse
