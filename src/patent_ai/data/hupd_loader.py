"""Loader for the Harvard USPTO Patent Dataset (HUPD) sample.

HUPD (Suzgun et al., NeurIPS 2022 Datasets & Benchmarks track, CC-BY-SA-4.0)
provides real USPTO patent applications with claims/abstract/description text
and real prosecution outcomes (decision field). It does not include the raw
office-action examiner text, so this module is the "real applicant data" half
of the pipeline: applicant claims + specification text + ground-truth outcome.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class PatentApplication:
    application_number: str
    title: str
    decision: str
    filing_date: str
    examiner_id: str
    main_cpc_label: str
    cpc_labels: tuple[str, ...]
    abstract: str
    claims: str
    background: str
    summary: str

    @property
    def is_allowed(self) -> bool:
        return self.decision in ("ACCEPTED", "CONT-ACCEPTED")

    @property
    def is_rejected(self) -> bool:
        return self.decision in ("REJECTED", "CONT-REJECTED")

    @property
    def first_claim(self) -> str:
        """The text of claim 1 (independent claim), used as the primary
        claim for prior-art overlap comparisons."""
        text = self.claims.strip()
        if not text:
            return ""
        # Claims are numbered "1. ... 2. ..."; split on a digit+period at
        # the start of a claim. Fall back to the whole block if that fails.
        import re

        parts = re.split(r"\n?\s*\d+\s*\.\s+", text)
        parts = [p.strip() for p in parts if p.strip()]
        return parts[0] if parts else text


def _record_from_json(data: dict) -> PatentApplication:
    return PatentApplication(
        application_number=data.get("application_number", ""),
        title=data.get("title", ""),
        decision=data.get("decision", ""),
        filing_date=data.get("filing_date", ""),
        examiner_id=str(data.get("examiner_id", "")),
        main_cpc_label=data.get("main_cpc_label", "") or "",
        cpc_labels=tuple(data.get("cpc_labels", []) or []),
        abstract=data.get("abstract", "") or "",
        claims=data.get("claims", "") or "",
        background=data.get("background", "") or "",
        summary=data.get("summary", "") or "",
    )


class HUPDDataset:
    """Reads HUPD per-application JSON files from an extracted sample directory."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"HUPD sample directory not found: {self.root}")
        self._files = sorted(self.root.glob("*.json"))
        if not self._files:
            raise FileNotFoundError(f"No .json application files found under {self.root}")

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self) -> Iterator[PatentApplication]:
        for path in self._files:
            yield self.load_one(path)

    @staticmethod
    def load_one(path: Path) -> PatentApplication:
        with open(path, "r", encoding="utf-8") as f:
            return _record_from_json(json.load(f))

    def sample(self, n: int, seed: int = 0) -> list[PatentApplication]:
        import random

        rng = random.Random(seed)
        chosen = rng.sample(self._files, min(n, len(self._files)))
        return [self.load_one(p) for p in chosen]

    def find_by_application_number(self, app_number: str) -> PatentApplication | None:
        path = self.root / f"{app_number}.json"
        if path.exists():
            return self.load_one(path)
        return None
