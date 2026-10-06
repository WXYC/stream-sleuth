"""The one factory for every S3 client in the evaluation harness.

The harness reads two object stores: a station's broadcast archive, which is
irreplaceable, and the store holding its reference pool. Nothing in the harness
may write to either, whatever credential it runs with, so every client is built
here and carries a botocore ``before-call.s3`` handler that allows exactly
``ListObjectsV2``, ``GetObject``, and ``HeadObject`` and raises
:class:`S3WriteRefused` for any other operation before a request is signed.

The handler sees API operations only. Presigning (``generate_presigned_url``,
``generate_presigned_post``) makes no API call and so is not guarded here; the
textual scan in ``tests/import_scan.py`` forbids those names instead.

Settings come from ``os.environ`` only (see the README's evaluation section):

- :func:`archive_client` uses the named AWS profile in
  ``STREAM_SLEUTH_ARCHIVE_AWS_PROFILE``, or the default credential chain if unset.
- :func:`pool_client` and :func:`pool_bucket` use ``STREAM_SLEUTH_POOL_ENDPOINT``,
  ``_BUCKET``, ``_KEY_ID``, and ``_SECRET``. Each one that is unset or empty falls
  back to the same suffix under ``DIGITAL_ARCHIVE_STORE_AZURACAST_``, the names
  Backend-Service uses for the same store.
"""

from __future__ import annotations

import os
from typing import Any

import boto3

ALLOWED_OPERATIONS = frozenset({"ListObjectsV2", "GetObject", "HeadObject"})

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


def _guarded(client: Any) -> Any:
    client.meta.events.register("before-call.s3", _refuse_non_reads)
    return client


def archive_client() -> Any:
    """Return a read-only S3 client for the broadcast archive."""
    profile = os.environ.get("STREAM_SLEUTH_ARCHIVE_AWS_PROFILE") or None
    return _guarded(boto3.Session(profile_name=profile).client("s3"))


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
    client = boto3.session.Session().client(
        "s3",
        endpoint_url=_pool_setting("ENDPOINT"),
        aws_access_key_id=_pool_setting("KEY_ID"),
        aws_secret_access_key=_pool_setting("SECRET"),
        region_name="us-east-1",
    )
    return _guarded(client)
