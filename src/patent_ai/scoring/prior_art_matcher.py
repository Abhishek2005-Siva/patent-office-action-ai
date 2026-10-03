"""Claim-to-reference element overlap scoring.

Implements the guide's `score_prior_art_relevance` concretely: split the
applicant's claim into its clauses ("claim elements"), then measure how much
of each element's vocabulary is covered by the candidate reference text using
TF-IDF cosine similarity (a real, well-understood lexical-overlap measure --
no embedding model download required to keep this fully offline-runnable).

This is deliberately a lexical heuristic, not a legal anticipation/obviousness
determination -- it estimates how much an attorney should expect a reference
to actually "read on" the claim, to prioritize which citations are worth
addressing in a response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Claims are drafted as one sentence with clauses separated by semicolons or
# "wherein" / "further comprising" transitions -- these are the natural
# "element" boundaries patent attorneys reason about.
import re

_CLAUSE_SPLIT = re.compile(
    r";|(?:,?\s+wherein\s+)|(?:,?\s+further\s+comprising\s+)|(?:\.\s+)", re.IGNORECASE
)

MATCH_THRESHOLD = 0.15


def extract_claim_elements(claim_text: str) -> list[str]:
    """Split a claim into its constituent elements/clauses."""
    text = claim_text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(text) if p and p.strip()]
    # drop tiny fragments (e.g. leftover numbering) that carry no signal
    return [p for p in parts if len(p.split()) >= 3] or [text]


@dataclass(frozen=True)
class PriorArtOverlap:
    elements: list[str]
    element_similarities: list[float]
    matching_element_ratio: float  # fraction of elements with sim >= MATCH_THRESHOLD
    overlap_score: int  # 0-100


def score_prior_art_relevance(applicant_claim: str, reference_text: str) -> PriorArtOverlap:
    elements = extract_claim_elements(applicant_claim)
    if not elements or not reference_text.strip():
        return PriorArtOverlap(elements, [], 0.0, 0)

    corpus = elements + [reference_text]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(corpus)
    except ValueError:
        # empty vocabulary after stopword removal (e.g. all-numeric/junk text)
        return PriorArtOverlap(elements, [0.0] * len(elements), 0.0, 0)

    element_vectors = tfidf[:-1]
    reference_vector = tfidf[-1]
    sims = cosine_similarity(element_vectors, reference_vector).flatten()

    matched = sims >= MATCH_THRESHOLD
    ratio = float(matched.sum()) / len(sims)
    base_score = ratio * 60
    bonus = float(np.mean(sims)) * 40
    overlap = int(round(min(100, max(0, base_score + bonus))))

    return PriorArtOverlap(elements, [round(float(s), 4) for s in sims], round(ratio, 4), overlap)
