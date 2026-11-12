"""Render prompts from prompts/templates.yaml (plan D7).

A rendered prompt is (input_text, prefill): the instruction text, and the answer stem placed
after the model's response header. Backends wrap input_text in their own format (ChronoGPT's
Alpaca wrapper, or a chat template) and append the prefill.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

TEMPLATES_PATH = Path(__file__).resolve().parents[2] / "prompts" / "templates.yaml"
PARAPHRASES = tuple(f"P{i}" for i in range(1, 13))
FORMATS = ("comma", "indexed")


@cache
def spec() -> dict:
    return yaml.safe_load(TEMPLATES_PATH.read_text())


def templates_sha256() -> str:
    return hashlib.sha256(TEMPLATES_PATH.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Template:
    paraphrase: str
    fmt: str

    @property
    def name(self) -> str:
        return f"{self.paraphrase}-{self.fmt}"


ALL_TEMPLATES = tuple(Template(p, f) for p in PARAPHRASES for f in FORMATS)


def format_value(v: float, decimals: int) -> str:
    """Fixed decimals; never prints a negative zero."""
    s = f"{v:.{decimals}f}"
    return f"{0:.{decimals}f}" if float(s) == 0 else s


def render(
    template: Template,
    values: list[float],
    h: int,
    decimals: int = 0,
    labels: list[str] | None = None,
    target_label: str | None = None,
    named: dict | None = None,
) -> tuple[str, str]:
    """Render one prompt.

    values: the shown history, oldest first. h: steps from the last shown value to the target.
    labels / target_label: period labels for the indexed format (default 1..T and T+h; the
    named framing passes quarters). named: {"series", "first", "last", "target_q"} or None.
    """
    sp = spec()
    par = sp["paraphrases"][template.paraphrase]
    T = len(values)
    labels = labels or [str(i) for i in range(1, T + 1)]
    target_label = target_label or str(T + h)
    vals = [format_value(v, decimals) for v in values]

    if template.fmt == "comma":
        body = ", ".join(vals)
    else:
        body = "\n".join(f"{lab}: {v}" for lab, v in zip(labels, vals, strict=True))

    if named is None:
        td = sp["target_desc"]["anonymous"]
        if template.fmt == "comma":
            desc = td["comma"][1] if h == 1 else td["comma"]["other"].format(h=h)
        else:
            desc = td["indexed"].format(target_label=target_label)
        intro = par["intro_anonymous"]
    else:
        desc = sp["target_desc"]["named"][template.fmt].format(target_q=named["target_q"])
        intro = par["intro_named"].format(
            series=named["series"], first=named["first"], last=named["last"]
        )
    question = par["question"].format(target_desc=desc)
    input_text = f"{intro}\n{body}\n{question}"
    prefill = sp["formats"][template.fmt]["prefill"].format(target_label=target_label)
    return input_text, prefill


def chronogpt_wrap(input_text: str, prefill: str) -> str:
    c = spec()["chronogpt"]
    head = c["wrapper"].format(system_prompt=c["system_prompt"], input_text=input_text)
    return head + prefill


def prompt_id(input_text: str, prefill: str) -> str:
    return hashlib.sha256(f"{input_text}\x00{prefill}".encode()).hexdigest()[:24]
