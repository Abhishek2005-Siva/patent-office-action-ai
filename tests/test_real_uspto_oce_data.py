"""Regression tests against the real USPTO OCE office-action dataset
(patents-public-data.uspto_oce_office_actions on Google BigQuery, mirrored on
Kaggle as bigquery/uspto-oce-office-actions). Unlike HUPD (application-level
claims + outcome), this dataset has real per-rejection structured fields:
statute (action_type), claim numbers rejected, and cited reference IDs --
exactly the shape our office-action parser targets, though as structured DB
fields rather than office-action prose.

Obtained via a headless Kaggle kernel (BigQuery is not reachable through the
Kaggle Datasets API for this dataset -- see docs/DATA_SOURCES.md and
scripts/run_kaggle_bigquery_query.sh). This is real, but not
committed to the repo (data/raw/ is gitignored); tests skip if absent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from patent_ai.extraction.office_action_parser import _extract_citations, _parse_claim_numbers

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "kaggle_uspto_oce"

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason="Kaggle USPTO OCE sample not downloaded (see scripts/run_kaggle_bigquery_query.sh)"
)


def _embed_in_citation_sentence(ref: str) -> str:
    return f"The reference ({ref}) discloses a similar configuration."


def test_real_claim_numbers_parse_with_zero_mismatches():
    df = pd.read_csv(DATA_ROOT / "claim_numbers_large_sample.csv", dtype=str)
    claim_lists = df["claim_numbers"].dropna().tolist()
    assert len(claim_lists) > 1000

    mismatches = []
    for raw in claim_lists:
        expected = sorted(int(x) for x in raw.split(","))
        got = _parse_claim_numbers(raw)
        if got != expected:
            mismatches.append((raw, expected, got))

    assert mismatches == [], f"{len(mismatches)} real claim_numbers strings parsed incorrectly"


def test_real_claim_numbers_contain_no_dash_ranges():
    """Documents an assumption relied on above: real OCE claim_numbers are
    always fully enumerated, never dash-range notation ('1-3'). If USPTO
    changes this, the assumption -- not just the test -- needs revisiting."""
    df = pd.read_csv(DATA_ROOT / "claim_numbers_large_sample.csv", dtype=str)
    has_dash = df["claim_numbers"].dropna().str.contains("-").sum()
    assert has_dash == 0


def test_citation_regex_catches_most_formatted_real_citations_in_prose():
    """Guards the fix: originally 0.4% of real citation IDs matched at all.
    'Formatted' here excludes bare undelimited digit strings (a DB-storage
    artifact -- see docs/DATA_SOURCES.md for why those are out of scope) and
    free-text-leaked fragments, focusing on the comma/slash/country-code
    shapes real office-action prose actually uses."""
    df = pd.read_csv(DATA_ROOT / "citation_id_large_sample.csv", dtype=str)
    ids = df["citation_pat_pgpub_id"].dropna().tolist()
    formatted = [c for c in ids if not re.fullmatch(r"\d+", c)]
    assert len(formatted) > 100

    matched = sum(1 for c in formatted if _extract_citations(_embed_in_citation_sentence(c)))
    match_rate = matched / len(formatted)
    # 49.7% measured at fix time; regression-guard floor set below that with
    # margin. This is a real, messy field (some entries are leaked prose
    # fragments) -- the target is "meaningfully better than 0.4%", not 100%.
    assert match_rate > 0.35, f"citation match rate on real formatted references regressed to {match_rate:.1%}"


def test_citation_regex_still_matches_synthetic_dashed_publication_numbers():
    # US publication-number shape (4-digit year + 7-digit sequence) was
    # silently broken before the fix -- direct, explicit check.
    assert _extract_citations("Lee (US 2009/0139986) discloses a widget.") == ["US 2009/0139986"]


def test_real_action_types_cover_all_four_statutes_our_scorer_handles():
    df = pd.read_csv(DATA_ROOT / "rejections_action_type_full.csv", dtype=str)
    real_statutes = set(df["action_type"]) & {"101", "102", "103", "112"}
    assert real_statutes == {"101", "102", "103", "112"}
