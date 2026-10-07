"""``stream_sleuth.paths``: the checkout, the data directory, and the refusal to write into either."""

from __future__ import annotations

from pathlib import Path

import pytest

from stream_sleuth import paths
from stream_sleuth.paths import CHECKOUT, DataPathError, data_dir, inside_checkout

DEFAULT = Path.home() / ".local" / "share" / "stream-sleuth"


@pytest.fixture
def outside_link(tmp_path):
    """A symlink outside the checkout that resolves to a directory inside it."""
    link = tmp_path / "link"
    link.symlink_to(CHECKOUT / "tests")
    return link


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param(CHECKOUT, True, id="checkout"),
        pytest.param(CHECKOUT / "README.md", True, id="file"),
        pytest.param(CHECKOUT / "tests" / "unit", True, id="nested"),
        pytest.param(CHECKOUT / "tests" / ".." / "data", True, id="dotdot"),
        pytest.param(CHECKOUT.parent / f"{CHECKOUT.name}-other", False, id="prefix-sibling"),
        pytest.param(CHECKOUT.parent, False, id="parent"),
        pytest.param(DEFAULT, False, id="default-data-dir"),
    ],
)
def test_inside_checkout(path, expected):
    assert inside_checkout(path) is expected


def test_a_symlink_resolving_into_the_checkout_is_inside(outside_link):
    assert inside_checkout(outside_link)
    assert inside_checkout(outside_link / "unit")


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(CHECKOUT, id="checkout"),
        pytest.param(CHECKOUT / "data" / "x.db", id="nested"),
        pytest.param(Path("relative/dir"), id="relative"),
        pytest.param(Path("~/x"), id="home-relative"),
    ],
)
def test_require_outside_checkout_refuses(path):
    with pytest.raises(DataPathError, match="checkout|absolute"):
        paths.require_outside_checkout(path)


def test_require_outside_checkout_refuses_a_symlink_into_the_checkout(outside_link):
    with pytest.raises(DataPathError, match="inside the checkout"):
        paths.require_outside_checkout(outside_link)


def test_require_outside_checkout_returns_the_resolved_path(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "real")
    assert (
        paths.require_outside_checkout(tmp_path / "alias" / "x")
        == (tmp_path / "real" / "x").resolve()
    )


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_data_dir_defaults_when_unset_or_empty(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("STREAM_SLEUTH_DATA_DIR", raising=False)
    else:
        monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", value)
    assert data_dir() == DEFAULT.resolve()


def test_data_dir_reads_the_setting_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(tmp_path / "a"))
    assert data_dir() == (tmp_path / "a").resolve()
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(tmp_path / "b"))
    assert data_dir() == (tmp_path / "b").resolve()


def test_data_dir_expands_a_home_relative_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", "~/research")
    assert data_dir() == (tmp_path / "research").resolve()


@pytest.mark.parametrize("value", ["relative/dir", ".", str(CHECKOUT), str(CHECKOUT / "data")])
def test_data_dir_refuses_a_relative_or_in_checkout_setting(monkeypatch, value):
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", value)
    with pytest.raises(DataPathError):
        data_dir()


def test_data_dir_does_not_create_the_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("STREAM_SLEUTH_DATA_DIR", str(tmp_path / "new"))
    data_dir()
    assert not (tmp_path / "new").exists()
