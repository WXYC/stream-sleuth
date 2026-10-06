"""The ``stream_sleuth`` package: the shim's contract, the protocols, and the identification types.

``recognizer.py`` is a thin shim over the package, so its five functions must be
the package's own objects, not copies. Each protocol has exactly one conforming
class, which delegates to the function moved out of ``recognizer.py``.
"""

from __future__ import annotations

import importlib

import pytest
import shazamio


@pytest.fixture
def package(fresh_recognizer):
    """Import ``recognizer`` (and with it the package) afresh, then look up a submodule."""
    shim = fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")
    return shim, importlib.import_module


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("parse", "stream_sleuth.recognizers.shazam"),
        ("capture", "stream_sleuth.sources"),
        ("post", "stream_sleuth.outputs"),
        ("identify_once", "stream_sleuth.loop"),
        ("main", "stream_sleuth.loop"),
    ],
)
def test_the_shim_re_exports_the_packages_functions(package, name, module):
    shim, load = package

    assert getattr(shim, name) is getattr(load(module), name)


@pytest.mark.parametrize(
    ("protocol", "implementation"),
    [
        ("stream_sleuth.sources:Source", "stream_sleuth.sources:IcecastSource"),
        (
            "stream_sleuth.recognizers.base:Recognizer",
            "stream_sleuth.recognizers.shazam:ShazamRecognizer",
        ),
        ("stream_sleuth.outputs:Output", "stream_sleuth.outputs:HttpPostOutput"),
    ],
)
def test_each_protocol_has_a_conforming_class(package, protocol, implementation):
    _, load = package

    def resolve(ref: str) -> type:
        module, name = ref.split(":")
        return getattr(load(module), name)

    assert isinstance(resolve(implementation)(), resolve(protocol))


def test_icecast_source_captures_with_the_moved_function(package, monkeypatch):
    _, load = package
    sources = load("stream_sleuth.sources")
    calls = []
    monkeypatch.setattr(sources, "capture", lambda path, seconds: calls.append((path, seconds)))

    sources.IcecastSource().capture("clip.wav", 6)

    assert calls == [("clip.wav", 6)]


def test_shazam_recognizer_parses_shazams_response(package, monkeypatch):
    _, load = package

    async def recognize(self, data, *args, **kwargs):
        return {"track": {"subtitle": "Jessica Pratt", "title": "Back, Baby"}}

    monkeypatch.setattr(shazamio.Shazam, "recognize", recognize)

    result = load("stream_sleuth.recognizers.shazam").ShazamRecognizer().recognize("clip.wav")

    assert result == {"artist": "Jessica Pratt", "song": "Back, Baby", "album": "", "label": ""}


def test_http_post_output_emits_with_the_moved_function(package, monkeypatch):
    _, load = package
    outputs = load("stream_sleuth.outputs")
    track = {"artist": "Juana Molina", "song": "la paradoja", "album": "DOGA", "label": "Sonamos"}
    monkeypatch.setattr(outputs, "post", lambda t: 201 if t is track else 500)

    assert outputs.HttpPostOutput().emit(track) == 201


def test_identification_has_exactly_the_four_wire_keys(package):
    _, load = package
    base = load("stream_sleuth.recognizers.base")

    assert base.Identification.__required_keys__ == {"artist", "song", "album", "label"}
    assert base.Identification.__optional_keys__ == frozenset()
    assert base.EvalIdentification.__required_keys__ == {"artist", "song", "album", "label"}
    assert base.EvalIdentification.__optional_keys__ == {
        "at",
        "source",
        "confidence",
        "query_offset_s",
        "ref_start_s",
        "ref_key",
    }
