"""The read-only S3 factory refuses every operation except the three reads.

These tests use moto, not ``botocore.stub.Stubber``. botocore stops at the first
``before-call`` handler that returns a response, and it runs ``before-call.*.*``
handlers before ``before-call.s3`` ones. ``Stubber`` answers at ``before-call.*.*``,
so under a ``Stubber`` the guard never runs and these tests would pass without it.
moto answers at ``before-send``, after every ``before-call`` handler, so the guard
fires first and removing it makes the write tests fail.

This module is the one named exemption from the import scan in
``tests/import_scan.py``: it has to build unguarded clients to seed moto's store.
"""

from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from evaluation import s3_readonly
from evaluation.s3_readonly import S3WriteRefused

BUCKET = "synthetic-bucket"
KEY = "rotation/Heavy/juana-molina/doga/01-la-paradoja.mp3"
POOL_ENDPOINT = "https://pool.example.test"


@pytest.fixture(autouse=True)
def isolated_aws(aws_isolated_env):
    """The shared isolation, plus a ``synthetic-archive`` profile and the pool endpoint."""
    aws_isolated_env.config.write_text("[profile synthetic-archive]\nregion = us-east-1\n")
    aws_isolated_env.credentials.write_text(
        "[synthetic-archive]\naws_access_key_id = testing\naws_secret_access_key = testing\n"
    )
    aws_isolated_env.allow_custom_endpoints(POOL_ENDPOINT)


def _archive(monkeypatch, aws_env):
    monkeypatch.setenv("STREAM_SLEUTH_ARCHIVE_AWS_PROFILE", "synthetic-archive")
    return s3_readonly.archive_client()


def _pool(monkeypatch, aws_env):
    aws_env.use_pool(POOL_ENDPOINT, BUCKET)
    return s3_readonly.pool_client()


CONSTRUCTORS = pytest.mark.parametrize("build", [_archive, _pool], ids=["archive", "pool"])


def seed_objects(endpoint_url: str | None, bucket: str, objects: dict[str, bytes]) -> None:
    """Create ``bucket`` in moto's store holding ``objects``, for other test modules.

    Seeding moto takes write calls, which the scans allow only in this module, so
    every test module that needs a populated bucket seeds it through here.
    """
    raw = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_url)
    raw.create_bucket(Bucket=bucket)
    for key, body in objects.items():
        raw.put_object(Bucket=bucket, Key=key, Body=body)


def _seed(client):
    """Create the bucket and one object with an unguarded client on ``client``'s endpoint."""
    raw = boto3.client("s3", region_name="us-east-1", endpoint_url=client.meta.endpoint_url)
    raw.create_bucket(Bucket=BUCKET)
    raw.put_object(Bucket=BUCKET, Key=KEY, Body=b"synthetic audio")
    return raw


@CONSTRUCTORS
@mock_aws
def test_reads_succeed(monkeypatch, aws_isolated_env, tmp_path, build):
    client = build(monkeypatch, aws_isolated_env)
    _seed(client)

    listed = client.list_objects_v2(Bucket=BUCKET)
    assert [o["Key"] for o in listed["Contents"]] == [KEY]
    assert client.head_object(Bucket=BUCKET, Key=KEY)["ContentLength"] == 15
    assert client.get_object(Bucket=BUCKET, Key=KEY)["Body"].read() == b"synthetic audio"
    # The harness's own read paths: s3transfer's managed download and the paginator.
    client.download_file(BUCKET, KEY, str(tmp_path / "out.mp3"))
    assert (tmp_path / "out.mp3").read_bytes() == b"synthetic audio"
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET)
    assert [o["Key"] for page in pages for o in page["Contents"]] == [KEY]


COPY_SOURCE = {"Bucket": BUCKET, "Key": KEY}

# (operation the guard must name, call). The managed transfers run through
# s3transfer, which reaches the same client operations from worker threads.
WRITES = {
    "put_object": ("PutObject", lambda c: c.put_object(Bucket=BUCKET, Key="new.mp3", Body=b"x")),
    "delete_object": ("DeleteObject", lambda c: c.delete_object(Bucket=BUCKET, Key=KEY)),
    "delete_objects": (
        "DeleteObjects",
        lambda c: c.delete_objects(Bucket=BUCKET, Delete={"Objects": [{"Key": KEY}]}),
    ),
    "copy_object": (
        "CopyObject",
        lambda c: c.copy_object(Bucket=BUCKET, Key="copy.mp3", CopySource=COPY_SOURCE),
    ),
    "create_multipart_upload": (
        "CreateMultipartUpload",
        lambda c: c.create_multipart_upload(Bucket=BUCKET, Key="big.mp3"),
    ),
    "upload_part": (
        "UploadPart",
        lambda c: c.upload_part(
            Bucket=BUCKET, Key="big.mp3", PartNumber=1, UploadId="synthetic", Body=b"x"
        ),
    ),
    "managed upload_fileobj": (
        "PutObject",
        lambda c: c.upload_fileobj(io.BytesIO(b"x"), BUCKET, "new.mp3"),
    ),
    "managed copy": ("CopyObject", lambda c: c.copy(COPY_SOURCE, BUCKET, "copy.mp3")),
}


@CONSTRUCTORS
@pytest.mark.parametrize("operation, call", WRITES.values(), ids=list(WRITES))
@mock_aws
def test_writes_are_refused_and_never_reach_the_store(
    monkeypatch, aws_isolated_env, build, operation, call
):
    client = build(monkeypatch, aws_isolated_env)
    raw = _seed(client)

    with pytest.raises(S3WriteRefused, match=operation):
        call(client)

    # The store is exactly as seeded: the refused call never reached moto.
    assert [o["Key"] for o in raw.list_objects_v2(Bucket=BUCKET)["Contents"]] == [KEY]


@CONSTRUCTORS
@mock_aws
def test_without_its_handler_the_factory_client_writes_so_the_guard_is_what_refuses(
    monkeypatch, aws_isolated_env, build
):
    # The tamper check: take a client from the factory and remove only the guard's
    # handler, by the event name it is registered under. moto then accepts the write
    # that test_writes_are_refused_and_never_reach_the_store expects to be refused,
    # so it is the guard, not moto or the client's setup, that refuses.
    client = build(monkeypatch, aws_isolated_env)
    _seed(client)
    client.meta.events.unregister("before-call.s3", s3_readonly._refuse_non_reads)

    client.put_object(Bucket=BUCKET, Key="new.mp3", Body=b"x")

    assert client.head_object(Bucket=BUCKET, Key="new.mp3")["ContentLength"] == 1


def test_the_crt_transfer_client_is_not_installed():
    # With awscrt installed, boto3 may hand upload_file/copy to the CRT transfer
    # manager, which signs and sends without botocore's events, so the guard
    # would never see them. Keep boto3[crt] out of every extra.
    assert importlib.util.find_spec("awscrt") is None


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


def test_isolated_aws_builds_on_the_shared_aws_isolation(request, tmp_path_factory):
    assert "aws_isolated_env" in request.fixturenames
    config = Path(os.environ["AWS_CONFIG_FILE"])
    assert config.is_relative_to(tmp_path_factory.getbasetemp())
