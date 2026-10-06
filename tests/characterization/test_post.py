"""Pins ``recognizer.post()``: the exact request WXDU's ingest API receives."""

from __future__ import annotations

import urllib.error

import pytest

from tests.characterization.conftest import SECRET


@pytest.fixture
def recognizer(fresh_recognizer, ingest_server):
    return fresh_recognizer(WXDU_SHAZAM_API=ingest_server.url, WXDU_SHAZAM_SECRET=SECRET)


def test_posts_the_track_as_json_with_the_shared_secret(recognizer, ingest_server):
    track = {"artist": "Juana Molina", "song": "la paradoja", "album": "DOGA", "label": "Sonamos"}

    status = recognizer.post(track)

    assert status == 201
    [request] = ingest_server.requests
    assert request.path == "/api/shazam"
    assert request.json() == track
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["X-Ingest-Secret"] == SECRET


def test_serializes_whatever_dict_it_is_handed_without_filtering(recognizer, ingest_server):
    track = {"artist": "Hermanos Gutiérrez", "song": "El Bueno y el Malo", "extra": 1}

    recognizer.post(track)

    [request] = ingest_server.requests
    assert request.json() == track
    # json.dumps' default ensure_ascii: non-ASCII goes over the wire escaped.
    assert b"Guti\\u00e9rrez" in request.body


def test_an_error_status_raises(recognizer, ingest_server):
    ingest_server.statuses = [500]

    with pytest.raises(urllib.error.HTTPError):
        recognizer.post({"artist": "a", "song": "b", "album": "", "label": ""})
