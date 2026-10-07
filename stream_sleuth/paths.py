"""The checkout, which research data (result stores, indexes) never enters."""

from pathlib import Path

# The directory holding this package: the repo checkout.
CHECKOUT = Path(__file__).resolve().parents[1]


def inside_checkout(path: Path) -> bool:
    """True when ``path``, symlinks and ``..`` resolved, is the checkout or under it."""
    resolved = path.resolve()
    return resolved == CHECKOUT or CHECKOUT in resolved.parents
