import numpy as np
import pytest

from ewh.prompts.parse import first_line, is_valid, parse_number

CASES = [
    ("23", 23.0),
    (" 23", 23.0),
    ("\n23", 23.0),  # leading newline
    ("The next value is 23.", 23.0),  # words before
    ("I expect about -14 next.", -14.0),
    ("23 or 25", 23.0),  # two numbers: first
    ("23%", 23.0),  # trailing unit
    ("- 23", -23.0),  # split minus as first character
    ("−23", -23.0),  # unicode minus
    ("1,234", 1234.0),
    ("23, 45, 67", 23.0),  # continues the list
    ("2.5 percent", 2.5),
    ("-0.4", -0.4),
    ("", None),
    ("I cannot predict that.", None),
    ("23\n45", 23.0),
    ("\n\n\n", None),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("41. 23", 23.0), ("41: 23", 23.0), ("Period 41: -7", -7.0), (" 23", 23.0), ("41", 41.0)],
)
def test_index_echo_in_indexed_format(text, expected):
    assert parse_number(text, fmt="indexed", target_label="41") == expected


def test_minus_inside_words_is_not_a_sign():
    assert parse_number("value - 23") == 23.0  # a spaced minus mid-sentence is not a sign


def test_first_line():
    assert first_line("  \n 12, 13\nmore") == "12, 13"


def test_validity_rule():
    hist = np.array([10.0, 12.0, 8.0, 11.0, 9.0])  # sd ~ 1.58
    assert is_valid(11.0, hist)
    assert not is_valid(20000.0, hist)  # wild number
    assert not is_valid(None, hist)
    assert is_valid(9.0 + 10 * 1.0, np.array([9.0, 9.0, 9.0]))  # constant history: floor sd = 1
    assert not is_valid(9.0 + 10.5, np.array([9.0, 9.0, 9.0]))
