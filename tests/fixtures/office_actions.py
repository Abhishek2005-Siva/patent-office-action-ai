"""Synthetic-but-realistic office action text fixtures.

These are NOT real USPTO documents (no scriptable, unauthenticated source of
bulk real office-action text exists — Patent Center / data.uspto.gov sit
behind an AWS WAF JS challenge). Each fixture is hand-written to mirror
genuine MPEP/37 CFR examiner phrasing and statute boilerplate so the
rule-based parser and scorer are exercised against realistic structure and
language, with known-correct expected extractions for testing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OAFixture:
    name: str
    text: str
    expected_statutes: list[str]
    expected_claims: list[int]
    expected_min_references: int
    qualitative_strength: str  # "weak" | "moderate" | "strong" -- for scorer sanity checks


WEAK_103_NO_MOTIVATION = OAFixture(
    name="weak_103_no_motivation",
    qualitative_strength="weak",
    expected_statutes=["103"],
    expected_claims=[1, 2, 3],
    expected_min_references=2,
    text="""
NON-FINAL REJECTION

Application No.: 15/123,456

Claims 1-3 are rejected under 35 U.S.C. 103 as being unpatentable over
Smith (US 5,111,222) in view of Jones (US 6,222,333).

Smith discloses a water bottle comprising a body and a cap. Jones discloses
a separate container with a hinged lid. It may be possible to combine the
hinge of Jones with the cap of Smith. The resulting combination appears to
render claim 1 obvious. Applicant's magnetic locking mechanism recited in
claim 1 is not explicitly discussed by either reference, but the examiner
believes it would have been an obvious design choice.
""".strip(),
)


STRONG_103_WITH_MOTIVATION = OAFixture(
    name="strong_103_with_motivation",
    qualitative_strength="strong",
    expected_statutes=["103"],
    expected_claims=[1, 2, 3, 4, 5],
    expected_min_references=2,
    text="""
FINAL REJECTION

Application No.: 15/987,654

Claims 1-5 are rejected under 35 U.S.C. 103 as being unpatentable over
Nguyen (US 7,444,555) in view of Park (US 8,555,666).

Nguyen explicitly teaches a magnetic locking mechanism for a container
closure, and Park explicitly teaches applying such a mechanism to a
portable drinking vessel to prevent accidental opening during transport.
Park provides an explicit motivation to combine, clearly teaching that
combining a magnetic latch with a portable bottle closure reduces spillage.
One of ordinary skill in the art would have had a reasonable expectation of
success in combining these known elements according to their established
functions, and the combination merely yields the predictable result of a
sealed, spill-resistant container. Claims 1-5 are clearly and explicitly
taught by the combined references.
""".strip(),
)


MODERATE_102_MISSING_ELEMENT = OAFixture(
    name="moderate_102_missing_element",
    qualitative_strength="moderate",
    expected_statutes=["102"],
    expected_claims=[1],
    expected_min_references=1,
    text="""
NON-FINAL REJECTION

Application No.: 14/555,000

Claim 1 is rejected under 35 U.S.C. 102(a)(1) as being anticipated by
Lee (US 9,000,111).

Lee discloses a hiking shoe comprising a sole, an upper, and a lacing
system. Lee appears to also disclose a heel counter, though the reference
does not explicitly show the counter formed as a single molded piece as
recited in claim 1. The examiner submits that this distinction may not be
patentably significant.
""".strip(),
)


WEAK_112_ELEMENT_IN_SPEC = OAFixture(
    name="weak_112_written_description",
    qualitative_strength="weak",
    expected_statutes=["112"],
    expected_claims=[4],
    expected_min_references=0,
    text="""
NON-FINAL REJECTION

Application No.: 16/222,333

Claim 4 is rejected under 35 U.S.C. 112(a) as failing to comply with the
written description requirement. Claim 4 recites "a self-adjusting tension
band," and it seems this term may not be adequately supported by the
original disclosure. The examiner notes the specification appears to
describe a tension band at paragraph [0031], but it is not entirely clear
whether the self-adjusting feature is explicitly disclosed.
""".strip(),
)


STRONG_102_ALL_ELEMENTS = OAFixture(
    name="strong_102_all_elements",
    qualitative_strength="strong",
    expected_statutes=["102"],
    expected_claims=[1],
    expected_min_references=1,
    text="""
FINAL REJECTION

Application No.: 14/600,700

Claim 1 is rejected under 35 U.S.C. 102(a)(1) as being anticipated by
Osei (US 8,111,999).

Osei explicitly discloses each and every element of claim 1, arranged in
the exact configuration recited, including the sole, upper, lacing system,
and a heel counter formed as a single molded piece as claimed. Osei
unambiguously teaches this precise arrangement.
""".strip(),
)


MULTI_REJECTION_MIXED = OAFixture(
    name="multi_rejection_mixed",
    qualitative_strength="moderate",
    expected_statutes=["102", "103"],
    expected_claims=[1, 2, 6, 7, 8],
    expected_min_references=3,
    text="""
NON-FINAL REJECTION

Application No.: 15/700,001

Claims 1-2 are rejected under 35 U.S.C. 102(a)(1) as being anticipated by
Chen (US 6,777,888).

Chen discloses each and every element of claim 1, including the recited
sensor array and control circuit, arranged in the exact configuration
claimed.

Claims 6-8 are rejected under 35 U.S.C. 103 as being unpatentable over
Chen (US 6,777,888) in view of Patel (US 7,888,999) and further in view of
Osei (US 8,999,000).

Patel teaches a wireless transmitter module, and Osei teaches a battery
management circuit. It would have been obvious to combine these known
components with Chen's sensor array because doing so would predictably
enable remote monitoring, a commonly desired feature.
""".strip(),
)


STRONG_101_ABSTRACT_IDEA = OAFixture(
    name="strong_101_abstract_idea",
    qualitative_strength="strong",
    expected_statutes=["101"],
    expected_claims=[1, 9, 15],
    expected_min_references=0,
    text="""
FINAL REJECTION

Application No.: 15/300,200

Claims 1, 9, and 15 are rejected under 35 U.S.C. 101 because the claimed
invention is directed to a judicial exception (an abstract idea) without
significantly more. The claims recite the abstract idea of collecting,
analyzing, and displaying information, which clearly falls within the
"certain methods of organizing human activity" grouping. The additional
elements of a generic processor and generic memory, considered individually
and in combination, do not integrate the judicial exception into a
practical application and do not amount to significantly more than the
judicial exception itself, as they explicitly recite only generic computer
components performing generic computer functions.
""".strip(),
)


ALL_FIXTURES: list[OAFixture] = [
    WEAK_103_NO_MOTIVATION,
    STRONG_103_WITH_MOTIVATION,
    MODERATE_102_MISSING_ELEMENT,
    STRONG_102_ALL_ELEMENTS,
    WEAK_112_ELEMENT_IN_SPEC,
    MULTI_REJECTION_MIXED,
    STRONG_101_ABSTRACT_IDEA,
]
