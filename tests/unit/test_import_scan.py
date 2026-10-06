"""The import scan catches every way of reaching boto3 outside the factory.

Each scanner is first shown to flag a synthetic snippet (the forced failure the
org's test-patterns doc asks for) and only then run over the real tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.import_scan import scan_for_s3_imports, scan_for_s3_write_names

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tree(tmp_path: Path, relative: str, source: str) -> Path:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return tmp_path


@pytest.mark.parametrize(
    "source",
    [
        "import boto3\n",
        "import boto3 as b\n",
        "from boto3 import client\n",
        "from botocore.session import get_session\n",
        "import s3transfer.manager\n",
    ],
)
@pytest.mark.parametrize(
    "relative",
    [
        "recognizer.py",
        "stream_sleuth/sources.py",
        "evaluation/pool.py",
        "tests/unit/test_x.py",
        "scripts/backfill.py",  # a directory that does not exist yet is still scanned
    ],
)
def test_import_scan_flags_each_import_form_anywhere_first_party(tmp_path, source, relative):
    root = _tree(tmp_path, relative, source)

    assert scan_for_s3_imports(root) == [f"{relative}:1"]


@pytest.mark.parametrize(
    "relative",
    [".venv/lib/site-packages/boto3/session.py", "build/lib/evaluation/pool.py"],
)
def test_import_scan_skips_installed_and_build_trees(tmp_path, relative):
    root = _tree(tmp_path, relative, "import boto3\n")

    assert scan_for_s3_imports(root) == []


def test_import_scan_allows_the_factory_and_its_named_test_module(tmp_path):
    _tree(tmp_path, "evaluation/s3_readonly.py", "import boto3\n")
    root = _tree(tmp_path, "tests/unit/test_s3_readonly.py", "import boto3\n")

    assert scan_for_s3_imports(root) == []


@pytest.mark.parametrize(
    "source",
    [
        "client.put_object(Bucket='b', Key='k', Body=b'')\n",
        "client.upload_file('f', 'b', 'k')\n",
        "client.upload_fileobj(f, 'b', 'k')\n",
        "client.copy_object(**kw)\n",
        "client.delete_object(Bucket='b', Key='k')\n",
        "client.delete_objects(Bucket='b', Delete={})\n",
        "client.create_multipart_upload(Bucket='b', Key='k')\n",
        "client.upload_part(**kw)\n",
        "client.abort_multipart_upload(**kw)\n",
        "client.put_object_acl(**kw)\n",
        "client.put_bucket_policy(**kw)\n",
        "client.delete_bucket(Bucket='b')\n",
        "client.restore_object(**kw)\n",
        "bucket.Object('k').copy_from(CopySource=src)\n",
        "client.meta.events.unregister('before-call.s3', handler)\n",
        "s3_readonly._refuse_non_reads\n",
        "bucket.Object('k').delete()\n",
        "client.generate_presigned_url('put_object')\n",
        "client.generate_presigned_post('b', 'k')\n",
        "importlib.import_module('boto3')\n",
        '__import__("botocore")\n',
    ],
)
def test_write_name_scan_flags_each_name(tmp_path, source):
    root = _tree(tmp_path, "evaluation/archive.py", source)

    assert scan_for_s3_write_names(root) == ["evaluation/archive.py:1"]


def test_real_tree_has_no_s3_client_outside_the_factory():
    assert scan_for_s3_imports(REPO_ROOT) == []


def test_real_tree_has_no_s3_write_or_presign_names():
    assert scan_for_s3_write_names(REPO_ROOT) == []
