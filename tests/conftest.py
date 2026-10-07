"""Fixtures shared by every test suite."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
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


# Every prefix a test module's own settings could collide with: AWS's, the recognizer's,
# the harness's, and Backend-Service's names that the pool settings fall back to.
_ISOLATED_PREFIXES = ("AWS_", "WXDU_", "STREAM_SLEUTH_", "DIGITAL_ARCHIVE_STORE_")


@dataclass(frozen=True)
class AwsEnv:
    """What ``aws_isolated_env`` set up, for module fixtures that add to it."""

    config: Path
    credentials: Path
    _monkeypatch: pytest.MonkeyPatch

    def allow_custom_endpoints(self, *endpoints: str) -> None:
        """Register non-AWS endpoints with moto.

        Without this, moto passes an unrecognized host through to a real request.
        """
        self._monkeypatch.setenv("MOTO_S3_CUSTOM_ENDPOINTS", ",".join(endpoints))

    def use_pool(self, endpoint: str, bucket: str) -> None:
        """Point the pool client at a moto ``endpoint`` and ``bucket``.

        Registers the endpoint with moto and sets ``STREAM_SLEUTH_POOL_ENDPOINT``,
        ``STREAM_SLEUTH_POOL_BUCKET``, ``STREAM_SLEUTH_POOL_KEY_ID`` and
        ``STREAM_SLEUTH_POOL_SECRET`` (the last two to ``testing``). A
        module that needs other settings, such as ``STREAM_SLEUTH_DATA_DIR``, sets
        them itself. This only sets environment variables, so it imports no
        ``boto3``, ``botocore`` or ``s3transfer``.
        """
        self.allow_custom_endpoints(endpoint)
        self._monkeypatch.setenv("STREAM_SLEUTH_POOL_ENDPOINT", endpoint)
        self._monkeypatch.setenv("STREAM_SLEUTH_POOL_BUCKET", bucket)
        self._monkeypatch.setenv("STREAM_SLEUTH_POOL_KEY_ID", "testing")
        self._monkeypatch.setenv("STREAM_SLEUTH_POOL_SECRET", "testing")


@pytest.fixture
def aws_isolated_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> AwsEnv:
    """Cut a test off from the developer's real AWS setup, for moto to answer.

    Deletes every ``AWS_*``, ``WXDU_*``, ``STREAM_SLEUTH_*`` and
    ``DIGITAL_ARCHIVE_STORE_*`` variable. Purging ``AWS_*`` alone removes
    ``AWS_PROFILE`` but leaves ``~/.aws/config`` and ``~/.aws/credentials`` in
    play, so a ``[default]`` section's region, ``s3`` settings or ``endpoint_url``
    would reach the test; this points ``AWS_CONFIG_FILE`` and
    ``AWS_SHARED_CREDENTIALS_FILE`` at empty files in a temp directory instead,
    and sets fake keys and ``AWS_DEFAULT_REGION``. The files live outside
    ``tmp_path`` so a test that lists its own ``tmp_path`` does not see them.

    A module fixture builds on this and adds only its own settings; a profile
    goes in the returned files, and non-AWS endpoints go through
    :meth:`AwsEnv.allow_custom_endpoints` (or :meth:`AwsEnv.use_pool` for the pool
    client). This sets environment variables and
    writes files only, so it imports no ``boto3``, ``botocore`` or ``s3transfer``
    and the import scan's exemptions are unchanged.
    """
    for name in [n for n in os.environ if n.startswith(_ISOLATED_PREFIXES)]:
        monkeypatch.delenv(name)
    aws_dir = tmp_path_factory.mktemp("aws")
    config, credentials = aws_dir / "config", aws_dir / "credentials"
    config.write_text("")
    credentials.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    return AwsEnv(config, credentials, monkeypatch)
