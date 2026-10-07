"""Read a numeric setting from the environment, or refuse the run with one line naming it.

Station-neutral: it imports nothing from the study or the runtime package.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", int, float)


def env_number(name: str, default: str, kind: Callable[[str], T], minimum: T, what: str) -> T:
    """``kind(os.environ[name])`` (or of ``default``) when it is finite and at least ``minimum``.

    Anything else, whether it does not parse (``500/day``), is NaN or infinite, or is below
    ``minimum``, raises ``SystemExit`` with the one line ``<name>=<value!r>: must be <what>``,
    which the interpreter prints without a traceback and exits 1 on.
    """
    raw = os.environ.get(name, default)
    try:
        value = kind(raw)
        if not (math.isfinite(value) and value >= minimum):
            raise ValueError(raw)
    except (ValueError, OverflowError):
        raise SystemExit(f"{name}={raw!r}: must be {what}") from None
    return value
