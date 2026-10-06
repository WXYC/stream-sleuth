"""Structural and textual checks that no first-party code builds its own S3 client.

Every S3 client in this repo must come from ``evaluation.s3_readonly``, whose
guard refuses any operation but the three reads. These scanners enforce that
over all first-party code: ``recognizer.py``, ``stream_sleuth/``, ``evaluation/``,
and ``tests/``.

- :func:`scan_for_s3_imports` parses each file with ``ast`` and reports every
  ``import`` or ``from ... import`` of ``boto3``, ``botocore``, or ``s3transfer``.
- :func:`scan_for_s3_write_names` is the textual backstop for what an import walk
  misses: S3 write and presign method names, and dynamic imports of those packages.

Both return ``["<relative path>:<line>", ...]``; an empty list means clean.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

S3_PACKAGES = frozenset({"boto3", "botocore", "s3transfer"})

# The factory itself, and the one test module that must build unguarded clients
# to seed moto and to prove the guard is what refuses a write.
IMPORT_EXEMPT = frozenset({"evaluation/s3_readonly.py", "tests/unit/test_s3_readonly.py"})

# Files that legitimately spell the forbidden names: the factory's docstring,
# its tests, and this scanner and its synthetic snippets.
WRITE_NAME_EXEMPT = IMPORT_EXEMPT | {"tests/import_scan.py", "tests/unit/test_import_scan.py"}

_WRITE_NAMES = re.compile(
    r"put_object|upload_file|upload_fileobj|\.copy_object\(|delete_object|delete_objects"
    r"|create_multipart_upload|\.delete\(|generate_presigned_url|generate_presigned_post"
    r"|(?:import_module|__import__)\(\s*['\"](?:boto3|botocore|s3transfer)"
)


def _first_party_files(root: Path) -> Iterator[tuple[str, Path]]:
    candidates = [root / "recognizer.py"]
    for directory in ("stream_sleuth", "evaluation", "tests"):
        candidates.extend(sorted((root / directory).rglob("*.py")))
    for path in candidates:
        if path.is_file():
            yield path.relative_to(root).as_posix(), path


def _imported_packages(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if node.module and node.level == 0:
        return [node.module.split(".")[0]]
    return []


def scan_for_s3_imports(root: Path) -> list[str]:
    """Report every import of an S3 package outside :data:`IMPORT_EXEMPT`."""
    found = []
    for relative, path in _first_party_files(root):
        if relative in IMPORT_EXEMPT:
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=relative)):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if S3_PACKAGES.intersection(_imported_packages(node)):
                found.append(f"{relative}:{node.lineno}")
    return sorted(found)


def scan_for_s3_write_names(root: Path) -> list[str]:
    """Report every line naming an S3 write, a presign, or a dynamic S3 import."""
    found = []
    for relative, path in _first_party_files(root):
        if relative in WRITE_NAME_EXEMPT:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if _WRITE_NAMES.search(line):
                found.append(f"{relative}:{number}")
    return sorted(found)
