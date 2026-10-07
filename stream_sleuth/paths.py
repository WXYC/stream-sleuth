"""The checkout, which research data (result stores, indexes) never enters, and the data directory.

Reads no setting at import and imports nothing from :mod:`stream_sleuth.config`.
"""

import os
from pathlib import Path

# The directory holding this package: the repo checkout.
CHECKOUT = Path(__file__).resolve().parents[1]

# The one place the default lives.
_DEFAULT_DATA_DIR = "~/.local/share/stream-sleuth"


class DataPathError(ValueError):
    """A path research data would be written to is relative or inside the checkout."""


def inside_checkout(path: Path) -> bool:
    """True when ``path``, symlinks and ``..`` resolved, is the checkout or under it.

    Compares file identity, not spelling, for every existing ancestor, so a mis-cased
    path on a case-insensitive volume or one reached through a firmlink still counts.
    """
    resolved = path.resolve()
    if resolved == CHECKOUT or CHECKOUT in resolved.parents:
        return True
    return any(p.exists() and os.path.samefile(p, CHECKOUT) for p in (resolved, *resolved.parents))


def require_outside_checkout(path: Path) -> Path:
    """Return ``path`` resolved, or raise :class:`DataPathError` if it is relative or inside the checkout.

    Call it before any request, ``mkdir``, or write, so a refused path leaves no trace.
    """
    if not path.is_absolute():
        raise DataPathError(f"{path} is not an absolute path")
    if inside_checkout(path):
        raise DataPathError(f"{path} is inside the checkout {CHECKOUT}")
    return path.resolve()


def data_dir() -> Path:
    """``$STREAM_SLEUTH_DATA_DIR`` (unset or empty means ``~/.local/share/stream-sleuth``), resolved.

    Read at call time, never created. Raises :class:`DataPathError` if the result is
    relative after ``~`` expansion or inside the checkout.
    """
    setting = os.environ.get("STREAM_SLEUTH_DATA_DIR") or _DEFAULT_DATA_DIR
    return require_outside_checkout(Path(setting).expanduser())
