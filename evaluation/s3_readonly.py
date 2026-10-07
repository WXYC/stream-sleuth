"""The one factory for every S3 client in the evaluation harness.

The harness reads two object stores: a station's broadcast archive, which is
irreplaceable, and the store holding its reference pool. Nothing in the harness
may write to either, whatever credential it runs with, so every client is built
here and carries a botocore ``before-call.s3`` handler that allows exactly
``ListObjectsV2``, ``GetObject``, and ``HeadObject`` and raises
:class:`S3WriteRefused` for any other operation before a request is signed.

This is a guard against accidents, not a security boundary: a caller holding a
client could unregister the handler. Pair it with read-only credentials. The
handler sees botocore API operations only. Presigning makes no API call, and
the CRT transfer client (``awscrt``) sends without botocore's events, so the
tests in ``tests/`` forbid the presign names and require ``awscrt`` be absent.

Settings come from ``os.environ`` only (see the README's evaluation section):

- :func:`archive_client` uses the named AWS profile in
  ``STREAM_SLEUTH_ARCHIVE_AWS_PROFILE``, or the default credential chain if unset;
  :func:`archive_bucket` reads ``STREAM_SLEUTH_ARCHIVE_BUCKET``.
- :func:`pool_client` and :func:`pool_bucket` use ``STREAM_SLEUTH_POOL_ENDPOINT``,
  ``_BUCKET``, ``_KEY_ID``, and ``_SECRET``. Each one that is unset or empty falls
  back to the same suffix under ``DIGITAL_ARCHIVE_STORE_AZURACAST_``, the names
  Backend-Service uses for the same store.
"""

from __future__ import annotations

import os
from typing import Any

from botocore.exceptions import IncompleteReadError, ReadTimeoutError, ResponseStreamingError

ALLOWED_OPERATIONS = frozenset({"ListObjectsV2", "GetObject", "HeadObject"})

# What a ``GetObject`` body raises when its stream ends before ``ContentLength``:
# the connection closed early, reset, or stalled. Exported here because no other
# module may import botocore.
STREAM_ERRORS: tuple[type[Exception], ...] = (
    IncompleteReadError,
    ResponseStreamingError,
    ReadTimeoutError,
)

_POOL_PREFIX = "STREAM_SLEUTH_POOL_"
_POOL_FALLBACK_PREFIX = "DIGITAL_ARCHIVE_STORE_AZURACAST_"


class S3WriteRefused(RuntimeError):  # noqa: N818 - name fixed by the plan and issue #4
    """Raised instead of sending any S3 operation outside :data:`ALLOWED_OPERATIONS`."""


class MissingSettingError(RuntimeError):
    """Raised when a required setting is absent under both of its names."""


def _refuse_non_reads(model: Any, **_: Any) -> None:
    if model.name not in ALLOWED_OPERATIONS:
        raise S3WriteRefused(
            f"{model.name} refused: this client is read-only "
            f"(allowed: {', '.join(sorted(ALLOWED_OPERATIONS))})"
        )


def _guarded_client(session_kwargs: dict[str, Any], **client_kwargs: Any) -> Any:
    # Imported here so this module exposes no ``boto3`` attribute through which
    # other code could build an unguarded client without importing boto3 itself.
    import boto3

    client = boto3.Session(**session_kwargs).client("s3", **client_kwargs)
    client.meta.events.register("before-call.s3", _refuse_non_reads)
    return client


def archive_client() -> Any:
    """Return a read-only S3 client for the broadcast archive."""
    profile = os.environ.get("STREAM_SLEUTH_ARCHIVE_AWS_PROFILE") or None
    return _guarded_client({"profile_name": profile})


def archive_bucket() -> str:
    """Return the broadcast archive's bucket, from ``STREAM_SLEUTH_ARCHIVE_BUCKET``."""
    bucket = os.environ.get("STREAM_SLEUTH_ARCHIVE_BUCKET")
    if not bucket:
        raise MissingSettingError("set STREAM_SLEUTH_ARCHIVE_BUCKET")
    return bucket


def _pool_setting(suffix: str) -> str:
    for name in (_POOL_PREFIX + suffix, _POOL_FALLBACK_PREFIX + suffix):
        value = os.environ.get(name)
        if value:
            return value
    raise MissingSettingError(f"set {_POOL_PREFIX}{suffix} (or {_POOL_FALLBACK_PREFIX}{suffix})")


def pool_bucket() -> str:
    """Return the reference pool's bucket name, resolved like :func:`pool_client`."""
    return _pool_setting("BUCKET")


def pool_client() -> Any:
    """Return a read-only S3 client for the reference pool's S3-compatible store."""
    return _guarded_client(
        {},
        endpoint_url=_pool_setting("ENDPOINT"),
        aws_access_key_id=_pool_setting("KEY_ID"),
        aws_secret_access_key=_pool_setting("SECRET"),
        region_name="us-east-1",
    )
