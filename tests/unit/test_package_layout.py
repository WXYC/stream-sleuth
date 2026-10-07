"""The ``stream_sleuth`` package: the shim's contract, the protocols, and the identification types.

``recognizer.py`` is a thin shim over the package, so its five functions must be
the package's own objects, not copies. Each protocol has exactly one conforming
class, which delegates to the function moved out of ``recognizer.py``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, get_args, get_type_hints

import pytest
import shazamio

if TYPE_CHECKING:
    from stream_sleuth.outputs import Output
    from stream_sleuth.recognizers.base import Recognizer


@pytest.fixture
def shim(fresh_recognizer):
    """Import ``recognizer``, and with it the package, afresh."""
    return fresh_recognizer(WXDU_SHAZAM_SECRET="not-a-real-secret")


load = importlib.import_module


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
def test_the_shim_re_exports_the_packages_functions(shim, name, module):
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
def test_each_protocol_has_a_conforming_class(shim, protocol, implementation):
    def resolve(ref: str) -> type:
        module, name = ref.split(":")
        return getattr(load(module), name)

    proto, impl = resolve(protocol), resolve(implementation)
    # The class subclasses its protocol, so isinstance() alone would pass on the
    # protocol's inherited stub; each method must be the class's own.
    methods = {name for name, value in vars(proto).items() if callable(value) and name[0] != "_"}
    assert methods
    assert methods <= set(vars(impl))
    assert isinstance(impl(), proto)


def test_icecast_source_captures_with_the_moved_function(shim, monkeypatch):
    sources = load("stream_sleuth.sources")
    calls = []
    monkeypatch.setattr(sources, "capture", lambda path, seconds: calls.append((path, seconds)))

    sources.IcecastSource().capture("clip.wav", 6)

    assert calls == [("clip.wav", 6)]


def test_shazam_recognizer_parses_shazams_response(shim, monkeypatch):

    async def recognize(self, data, *args, **kwargs):
        return {"track": {"subtitle": "Jessica Pratt", "title": "Back, Baby"}}

    monkeypatch.setattr(shazamio.Shazam, "recognize", recognize)

    result = load("stream_sleuth.recognizers.shazam").ShazamRecognizer().recognize("clip.wav")

    assert result == {"artist": "Jessica Pratt", "song": "Back, Baby", "album": "", "label": ""}


def test_http_post_output_emits_with_the_moved_function(shim, monkeypatch):
    outputs = load("stream_sleuth.outputs")
    track = {"artist": "Juana Molina", "song": "la paradoja", "album": "DOGA", "label": "Sonamos"}
    monkeypatch.setattr(outputs, "post", lambda t: 201 if t is track else 500)

    assert outputs.HttpPostOutput().emit(track) == 201


def _recognize_and_emit(recognizer: Recognizer, output: Output, wav_path: str) -> object:
    """Join the two seams the way the loop will, with no cast.

    This is a type check as much as a test: mypy rejects it if ``Output.emit``
    cannot take the ``Identification`` that ``Recognizer.recognize`` returns.
    """
    identification = recognizer.recognize(wav_path)
    return output.emit(identification) if identification else None


def test_a_recognizers_identification_reaches_the_output(shim, monkeypatch):
    async def recognize(self, data, *args, **kwargs):
        return {"track": {"subtitle": "Hermanos Gutiérrez", "title": "El Bueno y el Malo"}}

    monkeypatch.setattr(shazamio.Shazam, "recognize", recognize)
    outputs = load("stream_sleuth.outputs")
    posted = []
    monkeypatch.setattr(outputs, "post", lambda t: posted.append(t) or 201)
    shazam = load("stream_sleuth.recognizers.shazam")

    status = _recognize_and_emit(shazam.ShazamRecognizer(), outputs.HttpPostOutput(), "clip.wav")

    assert status == 201
    assert posted == [
        {"artist": "Hermanos Gutiérrez", "song": "El Bueno y el Malo", "album": "", "label": ""}
    ]


def test_identification_has_exactly_the_four_wire_keys(shim):
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


def test_evaluation_source_vocabulary_is_closed(shim):
    base = load("stream_sleuth.recognizers.base")

    assert get_args(base.IdentificationSource) == ("shazam", "local")
    assert get_type_hints(base.EvalIdentification)["source"] == base.IdentificationSource
