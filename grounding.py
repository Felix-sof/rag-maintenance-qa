"""
Numeric grounding check: every number in an answer must come from a passage it cites.

The prompt tells the model to use only the context, and rag.py records which
[n] the answer cites - but neither proves that a cited passage says what the
sentence claims. In a maintenance answer the costly version of that gap is a
number: a threshold, a limit or a waiting time that reads plausibly and
carries a citation, but that no passage contains. Numbers are also the one
kind of claim that can be checked exactly, with no second LLM call.

A number is supported if it appears in a cited passage (including its
header: source, section, priority, revision date) or in the question. The
comparison is by value, not by spelling:

    8,6  == 8.6                              decimal comma or point
    13.000 == 13,000 == 13 000 == 13000      thousands grouping
    "beş arıza modu" supports "5"            number words up to ten

The check leans towards silence. A token that could be read two ways
("1.250" is 1250 in Turkish and 1.25 in English) counts if either reading
matches, so a wrong value can occasionally slip through - but a flagged
number is one the cited passages really don't contain. Not checked:
citation markers, list numbering at the start of a line, and digits inside
identifiers ("AI4I", "H2O"). A count the model made itself ("şu 4 neden")
is flagged, because no passage states it.
"""

import re
from decimal import Decimal

from retriever import turkish_lower

# a space-grouped number ("13 000"), else digits with "." / "," separators; never the tail of a word
NUMBER_PATTERN = re.compile(r"(?<![\w.,])(?:\d{1,3}(?:[   ]\d{3})+(?!\d)|\d+(?:[.,]\d+)*)")
CITATION_PATTERN = re.compile(r"\[[\d,\s]+\]")
LIST_MARKER_PATTERN = re.compile(r"^\s*(?:[-*]\s+)?\**\d+[.)]\**(?=\s)", re.MULTILINE)

NUMBER_WORDS = {
    "bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _values(token: str) -> tuple[set[Decimal], set[Decimal]]:
    """What a number token can mean: (its readings as one number, its digit groups one by one).

    The second set is only filled when the groups may be separate numbers: a
    space-grouped token ("2 300" can be 2300, or 2 followed by 300) or one
    with no reading as a single number (a date "08.10.2026", a version "1.2.3").
    """
    groups = re.split(r"[.,\s]", token)
    separators = re.findall(r"[.,\s]", token)
    if not separators:
        return {Decimal(token)}, set()

    whole = set()
    grouped = len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]) and len(set(separators)) == 1
    if grouped:
        whole.add(Decimal("".join(groups)))  # 13.000 -> 13000
    if not separators[-1].isspace() and separators[-1] not in separators[:-1]:
        decimal = Decimal("".join(groups[:-1]) + "." + groups[-1])  # 8,6 -> 8.6 and 1.234,5 -> 1234.5
        # "10.000" is ten thousand, not a long way of writing 10: read as 10 it would be
        # "supported" by any 10 in the passage, such as the month of its revision date
        if not (grouped and decimal == decimal.to_integral_value()):
            whole.add(decimal)
    pieces = {Decimal(g) for g in groups} if not whole or separators[0].isspace() else set()
    return whole, pieces


def _known_values(texts: list[str]) -> set[Decimal]:
    known = set()
    for text in texts:
        for token in NUMBER_PATTERN.findall(text):
            whole, pieces = _values(token)
            known |= whole | pieces
        known.update(Decimal(NUMBER_WORDS[w]) for w in re.findall(r"\w+", turkish_lower(text)) if w in NUMBER_WORDS)
    return known


def unsupported_numbers(answer: str, passages: list[str], question: str = "") -> list[str]:
    """Numbers the answer states that are in none of `passages` nor in the question.

    Returned as written in the answer, in order, without repeats.
    """
    known = _known_values([*passages, question])
    stated = LIST_MARKER_PATTERN.sub(" ", CITATION_PATTERN.sub(" ", answer))
    missing = []
    for token in NUMBER_PATTERN.findall(stated):
        whole, pieces = _values(token)
        if not (whole & known or (pieces and pieces <= known)) and token not in missing:
            missing.append(token)
    return missing
