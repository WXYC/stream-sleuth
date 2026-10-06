"""Fixtures shared by every test suite."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Iterator
from types import ModuleType

import pytest


@pytest.fixture
def fresh_recognizer(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., ModuleType]]:
    """Return a function that imports ``recognizer`` afresh from a clean environment.

    ``recognizer.py`` reads every ``WXDU_*`` variable into a module constant at
    import time, so a test that sets the environment needs a fresh import. The
    function first deletes every ``WXDU_*`` and ``STREAM_SLEUTH_*`` variable, so
    nothing exported in the developer's shell can change the outcome; then sets
    the keyword arguments as environment variables; then removes ``recognizer``
    and every ``stream_sleuth*`` module from ``sys.modules`` and imports
    ``recognizer``. ``importlib.reload`` is deliberately not used: it stops
    working once the constants move into a submodule that siblings bind with
    ``from .config import ...``. On teardown the modules this test imported are
    dropped too, so no later import sees this test's environment.
    """

    def _ours(name: str) -> bool:
        return name == "recognizer" or name.startswith("stream_sleuth")

    def _import(**env: str) -> ModuleType:
        for name in [n for n in os.environ if n.startswith(("WXDU_", "STREAM_SLEUTH_"))]:
            monkeypatch.delenv(name)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        for name in [n for n in sys.modules if _ours(n)]:
            monkeypatch.delitem(sys.modules, name)
        return importlib.import_module("recognizer")

    yield _import
    # monkeypatch restores only entries that existed before the test, afterwards.
    for name in [n for n in sys.modules if _ours(n)]:
        del sys.modules[name]
