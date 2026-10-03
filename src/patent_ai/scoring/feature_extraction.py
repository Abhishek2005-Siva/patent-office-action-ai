"""Linguistic + structural feature extraction from a single rejection's text.

These are the signals the guide's scoring formulas are built on: hedging vs.
confident language, reference count, explicit motivation-to-combine (for
103), and rejection length. Kept as simple counts/flags on purpose -- the
guide's own advice is "start with 3-4 features, add complexity only if
accuracy drops."
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from patent_ai.extraction.office_action_parser import RejectionRecord

HEDGING_WORDS = [
    "may", "seems", "seem", "likely", "appears", "appear", "arguably",
    "possibly", "might", "could be", "not entirely clear", "believes",
    "submits that", "may not be",
]

CONFIDENT_WORDS = [
    "clearly", "explicitly", "teaches", "discloses each and every",
    "obviously", "unambiguously", "expressly",
]

MOTIVATION_PHRASES = [
    "motivation to combine", "would have been obvious to combine",
    "explicit motivation", "predictable result", "reasonable expectation of success",
]

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")


def _count_phrases(text: str, phrases: list[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(p) for p in phrases)


def _sentence_count(text: str) -> int:
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return len(sentences)


@dataclass(frozen=True)
class RejectionFeatures:
    statute: str
    num_references: int
    num_claims_rejected: int
    hedging_score: int
    confidence_score: int
    has_explicit_motivation: bool
    text_length: int
    sentence_count: int
    all_elements_present_language: bool
    """True if the text asserts the reference discloses "each and every" /
    "every element" of the claim (relevant to 102 strength)."""


ALL_ELEMENTS_PATTERN = re.compile(
    r"each and every|every element|all (?:of the )?(?:claimed )?elements", re.IGNORECASE
)


def extract_rejection_features(rejection: RejectionRecord) -> RejectionFeatures:
    text = rejection.text
    return RejectionFeatures(
        statute=rejection.statute,
        num_references=len(rejection.cited_references),
        num_claims_rejected=len(rejection.claims_rejected),
        hedging_score=_count_phrases(text, HEDGING_WORDS),
        confidence_score=_count_phrases(text, CONFIDENT_WORDS),
        has_explicit_motivation=_count_phrases(text, MOTIVATION_PHRASES) > 0,
        text_length=len(text),
        sentence_count=_sentence_count(text),
        all_elements_present_language=bool(ALL_ELEMENTS_PATTERN.search(text)),
    )
