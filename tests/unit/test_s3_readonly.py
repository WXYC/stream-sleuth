"""The read-only S3 factory refuses every operation except the three reads.

These tests use moto, not ``botocore.stub.Stubber``. botocore stops at the first
``before-call`` handler that returns a response, and it runs ``before-call.*.*``
handlers before ``before-call.s3`` ones. ``Stubber`` answers at ``before-call.*.*``,
so under a ``Stubber`` the guard never runs and these tests would pass without it.
moto answers at ``before-send``, after every ``before-call`` handler, so the guard
fires first and removing it makes the write tests fail.

This module is the one named exemption from the import scan in
``tests/import_scan.py``: it has to build unguarded clients, to create buckets
and to prove that the guard, not moto, is what refuses a write.
"""

from __future__ import annotations

import os

import boto3
import pytest
from moto import mock_aws

from evaluation import s3_readonly
from evaluation.s3_readonly import S3WriteRefused

BUCKET = "synthetic-bucket"
KEY = "rotation/Heavy/juana-molina/doga/01-la-paradoja.mp3"
POOL_ENDPOINT = "https://pool.example.test"


@pytest.fixture(autouse=True)
def isolated_aws(monkeypatch, tmp_path):
    """No test may see real AWS configuration, real credentials, or real settings."""
    for name in list(os.environ):
        if name.startswith(("AWS_", "STREAM_SLEUTH_", "DIGITAL_ARCHIVE_STORE_")):
            monkeypatch.delenv(name)
    config = tmp_path / "config"
    config.write_text("[profile synthetic-archive]\nregion = us-east-1\n")
    credentials = tmp_path / "credentials"
    credentials.write_text(
        "[synthetic-archive]\naws_access_key_id = testing\naws_secret_access_key = testing\n"
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    # Without this, moto passes an unrecognized host through to a real request.
    monkeypatch.setenv("MOTO_S3_CUSTOM_ENDPOINTS", POOL_ENDPOINT)


def _archive(monkeypatch):
    monkeypatch.setenv("STREAM_SLEUTH_ARCHIVE_AWS_PROFILE", "synthetic-archive")
    return s3_readonly.archive_client(), {}


def _pool(monkeypatch):
    monkeypatch.setenv("STREAM_SLEUTH_POOL_ENDPOINT", POOL_ENDPOINT)
    monkeypatch.setenv("STREAM_SLEUTH_POOL_BUCKET", BUCKET)
    monkeypatch.setenv("STREAM_SLEUTH_POOL_KEY_ID", "testing")
    monkeypatch.setenv("STREAM_SLEUTH_POOL_SECRET", "testing")
    return s3_readonly.pool_client(), {"endpoint_url": POOL_ENDPOINT}


CONSTRUCTORS = pytest.mark.parametrize("build", [_archive, _pool], ids=["archive", "pool"])


def _seed(**client_kwargs):
    """Create the bucket and one object with an unguarded client, as moto's store."""
    raw = boto3.client("s3", region_name="us-east-1", **client_kwargs)
    raw.create_bucket(Bucket=BUCKET)
    raw.put_object(Bucket=BUCKET, Key=KEY, Body=b"synthetic audio")
    return raw


@CONSTRUCTORS
@mock_aws
def test_reads_succeed(monkeypatch, build):
    client, kwargs = build(monkeypatch)
    _seed(**kwargs)

    listed = client.list_objects_v2(Bucket=BUCKET)
    assert [o["Key"] for o in listed["Contents"]] == [KEY]
    assert client.head_object(Bucket=BUCKET, Key=KEY)["ContentLength"] == 15
    assert client.get_object(Bucket=BUCKET, Key=KEY)["Body"].read() == b"synthetic audio"


WRITES = {
    "PutObject": lambda c: c.put_object(Bucket=BUCKET, Key="new.mp3", Body=b"x"),
    "DeleteObject": lambda c: c.delete_object(Bucket=BUCKET, Key=KEY),
    "CopyObject": lambda c: c.copy_object(
        Bucket=BUCKET, Key="copy.mp3", CopySource={"Bucket": BUCKET, "Key": KEY}
    ),
    "CreateMultipartUpload": lambda c: c.create_multipart_upload(Bucket=BUCKET, Key="big.mp3"),
}


@CONSTRUCTORS
@pytest.mark.parametrize("operation", sorted(WRITES))
@mock_aws
def test_writes_are_refused_and_never_reach_the_store(monkeypatch, build, operation):
    client, kwargs = build(monkeypatch)
    raw = _seed(**kwargs)

    with pytest.raises(S3WriteRefused, match=operation):
        WRITES[operation](client)

    # The store is exactly as seeded: the refused call never reached moto.
    assert [o["Key"] for o in raw.list_objects_v2(Bucket=BUCKET)["Contents"]] == [KEY]


@CONSTRUCTORS
@mock_aws
def test_an_unguarded_client_can_write_so_the_guard_is_what_refuses(monkeypatch, build):
    # The tamper check: with the guard's handler removed, moto accepts the write
    # that test_writes_are_refused_and_never_reach_the_store expects to be refused.
    _, kwargs = build(monkeypatch)
    raw = _seed(**kwargs)

    raw.put_object(Bucket=BUCKET, Key="new.mp3", Body=b"x")

    assert raw.head_object(Bucket=BUCKET, Key="new.mp3")["ContentLength"] == 1


def test_pool_settings_fall_back_to_the_backend_service_names(monkeypatch):
    monkeypatch.setenv("DIGITAL_ARCHIVE_STORE_AZURACAST_ENDPOINT", POOL_ENDPOINT)
    monkeypatch.setenv("DIGITAL_ARCHIVE_STORE_AZURACAST_BUCKET", "fallback-bucket")
    monkeypatch.setenv("DIGITAL_ARCHIVE_STORE_AZURACAST_KEY_ID", "testing")
    monkeypatch.setenv("DIGITAL_ARCHIVE_STORE_AZURACAST_SECRET", "testing")
    monkeypatch.setenv("STREAM_SLEUTH_POOL_BUCKET", "")  # empty counts as unset

    assert s3_readonly.pool_bucket() == "fallback-bucket"
    assert s3_readonly.pool_client().meta.endpoint_url == POOL_ENDPOINT


def test_pool_settings_prefer_the_stream_sleuth_names(monkeypatch):
    monkeypatch.setenv("STREAM_SLEUTH_POOL_BUCKET", "primary-bucket")
    monkeypatch.setenv("DIGITAL_ARCHIVE_STORE_AZURACAST_BUCKET", "fallback-bucket")

    assert s3_readonly.pool_bucket() == "primary-bucket"


def test_a_missing_pool_setting_names_both_variables(monkeypatch):
    with pytest.raises(s3_readonly.MissingSettingError) as excinfo:
        s3_readonly.pool_client()

    message = str(excinfo.value)
    assert "STREAM_SLEUTH_POOL_ENDPOINT" in message
    assert "DIGITAL_ARCHIVE_STORE_AZURACAST_ENDPOINT" in message
