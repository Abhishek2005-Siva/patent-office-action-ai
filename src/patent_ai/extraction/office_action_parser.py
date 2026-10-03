"""Rule-based structural parser for office action text.

Extracts, per 37 CFR / MPEP conventions: which claims are rejected, under
which statutory basis (101/102/103/112), and which references are cited —
without calling an LLM. This is the "Method 1" (template/rule) layer that
the hybrid scorer and response generator build on top of.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

STATUTE_PATTERNS = {
    "101": re.compile(r"35\s*U\.?S\.?C\.?\s*(?:Â§|§)?\s*101\b"),
    "102": re.compile(r"35\s*U\.?S\.?C\.?\s*(?:Â§|§)?\s*102\b"),
    "103": re.compile(r"35\s*U\.?S\.?C\.?\s*(?:Â§|§)?\s*103\b"),
    "112": re.compile(r"35\s*U\.?S\.?C\.?\s*(?:Â§|§)?\s*112\b"),
}

# A rejection paragraph typically opens with "Claim(s) <nums> [is/are]
# rejected under 35 U.S.C. <statute> ...". We split the office action into
# blocks starting at each such opening.
REJECTION_OPENER = re.compile(
    r"Claims?\s+([0-9,\sand\-]+?)\s+(?:is|are)\s+rejected\s+under\s+35\s*U\.?S\.?C\.?\s*(?:Â§|§)?\s*(101|102|103|112)",
    re.IGNORECASE,
)

CLAIM_NUM_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)")
CLAIM_NUM_SINGLE = re.compile(r"\d+")

# US patent number citation, several common formats:
#   US 5,123,456   US5,123,456   U.S. Patent No. 5,123,456   US 2015/0123456 A1
# The "US"/"U.S. Patent No." prefix is optional when the number is
# parenthesized -- real office actions routinely introduce the first
# reference with "US" ("Smith (US 5,123,456)") and then cite a later
# reference by name + bare number only ("Jones (6,234,567)"). Verified
# against the real USPTO OCE office-action dataset on Kaggle/BigQuery
# (patents-public-data.uspto_oce_office_actions): the original US-required
# regex matched only 0.4% of real cited-reference identifiers.
#
# Two distinct number shapes -- granted patents are comma-grouped in 3s
# (5,123,456), but pre-grant publication numbers are a 4-digit year + a
# 7-digit sequence (2015/0123456), NOT grouped in 3s. The original regex's
# own docstring claimed to support "US 2015/0123456 A1" but its shape
# (\d{1,2}[,/]\d{3}[,/]\d{3}) could never actually match a 4-digit year --
# real publication-number citations like "US 2009/0139986" silently matched
# nothing. Found via the same real-data check above.
_US_PATENT_NUMBER_SHAPE = r"\d{1,2},\d{3},\d{3}(?:\s*[A-Z]\d)?"
_US_PUBLICATION_NUMBER_SHAPE = r"\d{4}/\d{7}(?:\s*[A-Z]\d)?"
_PATENT_NUMBER_SHAPE = rf"(?:{_US_PATENT_NUMBER_SHAPE}|{_US_PUBLICATION_NUMBER_SHAPE})"
PATENT_CITATION = re.compile(
    rf"""
    (?:U\.?S\.?\s*(?:Patent\s*(?:No\.?)?)?\s*({_PATENT_NUMBER_SHAPE}))
    |
    \(\s*({_PATENT_NUMBER_SHAPE})\s*\)
    """,
    re.VERBOSE,
)

# Foreign patent/publication citations, e.g. "JP 2004-159476", "WO 2012136906".
# Requires a plausible number length right after the country code so common
# US state abbreviations (CA, DE, ...) in ordinary text aren't matched.
FOREIGN_PATENT_CITATION = re.compile(
    r"\b(JP|WO|EP|GB|DE|FR|CN|KR|AU|CA|RU)[\s-]+(\d[\d/-]{5,14})\b"
)

OFFICE_ACTION_TYPE = re.compile(r"\b(non-final|final)\s+rejection\b", re.IGNORECASE)

APPLICATION_NUMBER = re.compile(r"Application\s*(?:No\.?|Number)\s*[:\-]?\s*(\d{2}/?\d{3},?\d{3}|\d{7,8})", re.IGNORECASE)


@dataclass
class RejectionRecord:
    statute: str  # "101" | "102" | "103" | "112"
    claims_rejected: list[int]
    cited_references: list[str]
    text: str


@dataclass
class OfficeActionExtraction:
    application_number: str | None
    office_action_type: str | None  # "non-final" | "final" | None
    rejections: list[RejectionRecord] = field(default_factory=list)

    @property
    def rejection_types(self) -> list[str]:
        seen: list[str] = []
        for r in self.rejections:
            if r.statute not in seen:
                seen.append(r.statute)
        return seen

    @property
    def all_rejected_claims(self) -> list[int]:
        claims: set[int] = set()
        for r in self.rejections:
            claims.update(r.claims_rejected)
        return sorted(claims)

    @property
    def all_cited_references(self) -> list[str]:
        refs: list[str] = []
        for r in self.rejections:
            for c in r.cited_references:
                if c not in refs:
                    refs.append(c)
        return refs


def _parse_claim_numbers(claim_text: str) -> list[int]:
    """'1-3 and 5' -> [1, 2, 3, 5]; '7' -> [7]."""
    claim_text = claim_text.replace("and", ",")
    numbers: set[int] = set()
    # ranges first, then remove them so lone numbers aren't double counted
    remainder = claim_text
    for m in CLAIM_NUM_RANGE.finditer(claim_text):
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo <= hi and hi - lo < 1000:
            numbers.update(range(lo, hi + 1))
        remainder = remainder.replace(m.group(0), " ")
    for m in CLAIM_NUM_SINGLE.finditer(remainder):
        numbers.add(int(m.group(0)))
    return sorted(numbers)


def _extract_citations(text: str) -> list[str]:
    refs: list[str] = []
    for m in PATENT_CITATION.finditer(text):
        num = m.group(1) or m.group(2)
        normalized = "US " + num
        if normalized not in refs:
            refs.append(normalized)
    for m in FOREIGN_PATENT_CITATION.finditer(text):
        normalized = f"{m.group(1)} {m.group(2)}"
        if normalized not in refs:
            refs.append(normalized)
    return refs


def parse_office_action(text: str) -> OfficeActionExtraction:
    """Parse raw office action text into structured rejection records.

    Splits the document at each rejection-opener sentence, so the text
    associated with a rejection runs until the next opener (or end of doc).
    """
    openers = list(REJECTION_OPENER.finditer(text))
    rejections: list[RejectionRecord] = []

    for i, m in enumerate(openers):
        start = m.start()
        end = openers[i + 1].start() if i + 1 < len(openers) else len(text)
        block = text[start:end]

        claims_raw, statute = m.group(1), m.group(2)
        rejections.append(
            RejectionRecord(
                statute=statute,
                claims_rejected=_parse_claim_numbers(claims_raw),
                cited_references=_extract_citations(block),
                text=block.strip(),
            )
        )

    oa_type_match = OFFICE_ACTION_TYPE.search(text)
    app_number_match = APPLICATION_NUMBER.search(text)

    return OfficeActionExtraction(
        application_number=app_number_match.group(1) if app_number_match else None,
        office_action_type=oa_type_match.group(1).lower() if oa_type_match else None,
        rejections=rejections,
    )
