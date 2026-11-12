"""Parse a model's continuation into one number, and the outcome-independent validity rule
(plan D6). Rules are fixed here and bound to the tag."""

from __future__ import annotations

import math
import re

import numpy as np

MINUS = "-−–"  # hyphen-minus, minus sign, en dash
_NUM = re.compile(rf"(?<![\w.])([{MINUS}]?)(\d{{1,3}}(?:,\d{{3}})+|\d+)(\.\d+)?")
_SPLIT_MINUS = re.compile(rf"^[{MINUS}]\s(\d)")
MAX_SD = 10.0


def first_line(text: str) -> str:
    """Strip leading whitespace, then cut at the first newline (plan D3)."""
    return text.lstrip().split("\n", 1)[0]


def parse_number(text: str, fmt: str = "comma", target_label: str | None = None) -> float | None:
    """First number in the answer's first line.

    - An echoed period label at the start ("41. 23", "41: 23", "Period 41: 23") is skipped in the
      indexed format.
    - A minus sign separated by one space is accepted only as the first character ("- 23").
    - Thousands separators ("1,234") are read; "23, 45" reads 23.
    """
    line = first_line(text)
    if fmt == "indexed" and target_label:
        line = re.sub(rf"^(?:period\s+)?{re.escape(target_label)}\s*[:.)]\s*", "", line, flags=re.I)
    m = _SPLIT_MINUS.match(line)
    if m:
        line = "-" + line[2:]
    hit = _NUM.search(line)
    if not hit:
        return None
    sign, digits, frac = hit.groups()
    value = float(digits.replace(",", "") + (frac or ""))
    return -value if sign else value


def is_valid(value: float | None, hist: np.ndarray, max_sd: float = MAX_SD) -> bool:
    """Valid if parsed and within max_sd standard deviations of the shown history from the last
    shown value (prompt information only; plan D6)."""
    if value is None or not math.isfinite(value):
        return False
    sd = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.0
    return abs(value - float(hist[-1])) <= max_sd * max(sd, 1.0)
