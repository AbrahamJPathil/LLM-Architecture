"""
Text utilities for output sanitization.
"""
from __future__ import annotations

import os
from typing import Any


def maybe_strip_emoji(text: Any) -> str:
    """Return text with emojis/non-ASCII removed when NO-EMOJI mode is enabled.

    If WARMSTART_NO_EMOJI == "1", we strip non-ASCII characters to avoid
    Windows console encoding issues. Otherwise, return text unchanged.
    Accepts any object and converts to string.
    """
    s = str(text)
    if os.environ.get("WARMSTART_NO_EMOJI", "0") == "1":
        try:
            return s.encode("ascii", "ignore").decode("ascii")
        except Exception:
            return s
    return s


def safe_print(*args: Any, **kwargs: Any) -> None:
    """Print that respects NO-EMOJI mode by sanitizing text first."""
    sanitized = [maybe_strip_emoji(a) for a in args]
    print(*sanitized, **kwargs)
