"""Fixtures shared by every test suite."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from types import ModuleType

import pytest


@pytest.fixture
def fresh_recognizer(monkeypatch: pytest.MonkeyPatch) -> Callable[..., ModuleType]:
    """Return a function that imports ``recognizer`` afresh from a clean environment.

    ``recognizer.py`` reads every ``WXDU_*`` variable into a module constant at
    import time, so a test that sets the environment needs a fresh import. The
    function first deletes every ``WXDU_*`` and ``STREAM_SLEUTH_*`` variable, so
    nothing exported in the developer's shell can change the outcome; then sets
    the keyword arguments as environment variables; then removes ``recognizer``
    and every ``stream_sleuth*`` module from ``sys.modules`` and imports
    ``recognizer``. ``importlib.reload`` is deliberately not used: it stops
    working once the constants move into a submodule that siblings bind with
    ``from .config import ...``.
    """

    def _import(**env: str) -> ModuleType:
        for name in [n for n in os.environ if n.startswith(("WXDU_", "STREAM_SLEUTH_"))]:
            monkeypatch.delenv(name)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        for name in list(sys.modules):
            if name == "recognizer" or name.startswith("stream_sleuth"):
                monkeypatch.delitem(sys.modules, name)
        return importlib.import_module("recognizer")

    return _import
