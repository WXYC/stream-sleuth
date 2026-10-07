"""The shared AWS isolation fixture.

``aws_isolated_env`` (``tests/conftest.py``) cuts a test off from the developer's
real AWS setup. The module fixtures that talk to moto build on it, and each of those
modules has a test that its fixture does.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evaluation import s3_readonly

LEAKY_ENV = {
    "AWS_PROFILE": "someones-real-profile",
    "AWS_ENDPOINT_URL": "https://aws.example.test",
    "WXDU_SHAZAM_API": "https://shazam.example.test",
    "STREAM_SLEUTH_POOL_BUCKET": "someones-real-bucket",
    "DIGITAL_ARCHIVE_STORE_AZURACAST_KEY_ID": "someones-real-key",
}

LEAKY_AWS_CONFIG = "[default]\nregion = eu-west-1\nendpoint_url = https://aws.example.test\n"


@pytest.fixture
def leaky_home(monkeypatch, tmp_path_factory):
    """A home directory whose ``~/.aws/config`` would redirect every client, plus leaky variables."""
    home = tmp_path_factory.mktemp("home")
    (home / ".aws").mkdir()
    (home / ".aws" / "config").write_text(LEAKY_AWS_CONFIG)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("AWS_CONFIG_FILE", raising=False)
    monkeypatch.delenv("AWS_SHARED_CREDENTIALS_FILE", raising=False)
    for name, value in LEAKY_ENV.items():
        monkeypatch.setenv(name, value)
    return home


def _assert_cut_off_from_the_developers_aws(tmp_path_factory, home: Path) -> None:
    base = tmp_path_factory.getbasetemp()
    for name in ("AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE"):
        path = Path(os.environ[name])
        assert path.is_relative_to(base), f"{name} is outside the test's temp directory"
        assert not path.is_relative_to(home)
    assert not any(name in os.environ for name in LEAKY_ENV)


def test_the_fixture_points_aws_at_temp_files_and_purges_the_prefixes(
    leaky_home, aws_isolated_env, tmp_path_factory
):
    _assert_cut_off_from_the_developers_aws(tmp_path_factory, leaky_home)
    assert Path(os.environ["AWS_CONFIG_FILE"]) == aws_isolated_env.config
    assert Path(os.environ["AWS_SHARED_CREDENTIALS_FILE"]) == aws_isolated_env.credentials
    assert aws_isolated_env.config.read_text() == ""
    assert aws_isolated_env.credentials.read_text() == ""
    assert os.environ["AWS_DEFAULT_REGION"] == "us-east-1"
    assert os.environ["AWS_ACCESS_KEY_ID"] == os.environ["AWS_SECRET_ACCESS_KEY"] == "testing"


def test_the_default_credential_chain_ignores_the_developers_aws_config(
    leaky_home, aws_isolated_env
):
    client = s3_readonly.archive_client()

    assert client.meta.region_name == "us-east-1"
    assert client.meta.endpoint_url == "https://s3.amazonaws.com"


def test_custom_endpoints_are_registered_with_moto(aws_isolated_env):
    aws_isolated_env.allow_custom_endpoints("https://a.example.test", "https://b.example.test")

    assert os.environ["MOTO_S3_CUSTOM_ENDPOINTS"] == "https://a.example.test,https://b.example.test"


def test_use_pool_registers_the_endpoint_and_sets_the_four_pool_variables(aws_isolated_env):
    aws_isolated_env.use_pool("https://pool.example.test", "synthetic-pool")

    assert os.environ["MOTO_S3_CUSTOM_ENDPOINTS"] == "https://pool.example.test"
    pool_settings = {
        name: os.environ[f"STREAM_SLEUTH_POOL_{name}"]
        for name in ("ENDPOINT", "BUCKET", "KEY_ID", "SECRET")
    }
    assert pool_settings == {
        "ENDPOINT": "https://pool.example.test",
        "BUCKET": "synthetic-pool",
        "KEY_ID": "testing",
        "SECRET": "testing",
    }
