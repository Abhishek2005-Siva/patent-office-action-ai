# Patent Office Action AI

Two tools for patent prosecution, built from the planning docs in this repo:

1. **Rejection Strength Scorer** -- scores an office action's rejection(s)
   0-100 (how weak/strong), estimates a win probability, and gives a
   strategic recommendation.
2. **Response Generator** -- drafts a 37 CFR 1.111/1.121-style response
   letter, either from a deterministic template (no API key needed) or with
   Claude-generated argument prose (hybrid mode, requires
   `ANTHROPIC_API_KEY`).

Both are wired together behind a small Flask app.

## Status

All 5 pipeline stages are implemented and verified: **143 automated tests
pass**, including statistical validation against 2,500+ real patent
applications, regression tests against a real USPTO office-action dataset
(which surfaced and fixed two real citation-parsing bugs -- see below), and
a live HTTP smoke test of the running server. See `docs/DATA_SOURCES.md` for
exactly which parts are backed by real public
data vs. hand-authored realistic fixtures, and why.

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# one-time: download the real public dataset (~388MB, ~40s)
python scripts/download_hupd_sample.py

# run the full test suite (143 tests, ~7s)
pytest tests/ -v

# run the evaluation report (fixture ordering + real-data correlations)
python scripts/run_evaluation.py

# run the web app locally
python -m patent_ai.webapp.app
# -> open http://127.0.0.1:5000
```

To enable Claude-generated argument prose instead of the template-only
fallback, set `ANTHROPIC_API_KEY` before starting the app and check "Use
Claude to draft arguments" in the UI (or pass `"use_llm": true` to
`POST /analyze`).

## Streamlit app

`streamlit_app/app.py` is a hosted-friendly front end for the same pipeline. Pick one of the synthetic
sample office actions (or paste text, or upload a PDF), and it shows the 0-100 rejection strength,
the estimated chance of overcoming it, a per-statute breakdown, and a drafted response letter you
can download.

- **Template mode** needs no key and runs fully offline.
- **Model-written arguments:** pick NVIDIA (free key), OpenAI or Anthropic (Claude), paste your own
  key, and choose a model. The key is used only for your requests in that browser session. If the
  provider call fails, the app shows the error and falls back to the template letter.
- **Privacy:** in model mode the office action text is sent to the provider you chose, so don't paste
  confidential matter into a shared demo.

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

To host it: [share.streamlit.io](https://share.streamlit.io), choose this repository, the default
branch and `streamlit_app/app.py`. No secrets are needed.

## Architecture

```
src/patent_ai/
  data/hupd_loader.py            real HUPD applications: claims, spec text, decision outcome
  extraction/office_action_parser.py   rule-based: statute, claims rejected, citations, OA type
  scoring/
    feature_extraction.py        hedging/confidence/motivation signals from rejection text
    rejection_scorer.py          per-statute 0-100 strength scoring (102/103/112/101)
    prior_art_matcher.py         TF-IDF claim-element overlap vs. a reference/candidate text
    win_probability.py           strength -> win probability, with amendment/spec/examiner adjustments
  generation/
    llm_client.py                LLMClient protocol: AnthropicLLMClient, OpenAICompatibleLLMClient
                                  (OpenAI, NVIDIA free models, ...) / FakeLLMClient (tests)
    templates.py                 per-statute argument templates (fully offline)
    response_generator.py        TemplateResponseGenerator + LLMResponseGenerator (hybrid, with
                                  a guardrail that rejects any citation the LLM invents)
  evaluation/metrics.py          Spearman rank correlation, ROC-AUC (guide's Part 3 metrics)
  analysis.py                    analyze_office_action(): the one pipeline entry point both apps call
  webapp/app.py                  Flask app: /, /health, /analyze (text, JSON, or PDF upload)

streamlit_app/
  app.py                         Streamlit app (same pipeline), plus samples/ and its own requirements.txt

scripts/
  download_hupd_sample.py        reproducible HUPD dataset download
  run_kaggle_bigquery_query.sh   reproducible USPTO OCE pull via a headless Kaggle kernel
  kaggle_bigquery_kernel/        the kernel script + metadata that queries BigQuery
  run_evaluation.py              produces data/processed/evaluation_report.json

tests/
  fixtures/office_actions.py     7 realistic synthetic office actions (102/103/112/101, multi-rejection)
  pdf_helper.py                  builds a minimal real PDF for testing the upload path
  test_*.py                      143 tests total, run against fixtures, real HUPD data, and real USPTO OCE data
```

## What's real, what's synthetic, and why

Short version (full detail in `docs/DATA_SOURCES.md`): the real USPTO Office
Action Research Dataset -- the ideal training/eval data for a rejection
scorer -- sits behind an AWS WAF JavaScript challenge on
`data.uspto.gov`/Patent Center that a scripted client cannot pass; it is not
obtainable without a real browser session. The same underlying dataset
turned out to be reachable a different way, though: Kaggle's
`bigquery/uspto-oce-office-actions` gives any Kaggle *kernel* free,
pre-authenticated BigQuery access to it (not downloadable via the Kaggle
Datasets API directly), so it was pulled by pushing a headless script kernel
that queries it and exports CSVs -- see `docs/DATA_SOURCES.md` and
`scripts/run_kaggle_bigquery_query.sh`. This gave real, per-rejection
structured fields (statute, claim numbers, cited references) and, in the
process, surfaced and fixed two real bugs in the citation-extraction regex
(it matched only 0.4% of real cited-reference identifiers; see
`docs/DATA_SOURCES.md` for the details and the fix).

This project also uses the **Harvard USPTO Patent Dataset (HUPD)**, a real,
directly-downloadable public research dataset (26,808 real applications:
real claims, real examiner IDs, real CPC/IPC codes, real ACCEPTED/REJECTED
outcomes) for everything that operates on applicant claim text and real
outcomes (the prior-art matcher, the evaluation script's claim-breadth
analysis). Neither real dataset includes the examiner's actual office-action
*prose* (paragraphs of rejection reasoning), so the office-action parser and
rejection scorer are additionally tested against six hand-authored fixtures
that mirror genuine MPEP/37 CFR rejection phrasing -- clearly labeled as
synthetic, not real USPTO documents.

The evaluation report (`data/processed/evaluation_report.json`) is honest
about results either way: a real, weak-but-statistically-significant
correlation was found between claim breadth and real rejection outcomes
(shorter/broader claims reject more often, p<0.01 on ~800 real
applications); a same-CPC-class lexical "crowdedness" proxy was tested and
found to have **no** meaningful signal (reported as a negative result rather
than hidden).

## What this project deliberately does not do

Per the planning docs' own 6-week roadmap, phases 5-6 (payment integration,
accounts, beta customer acquisition, talking to attorneys) are
business decisions requiring real accounts, credentials, and human judgment
calls. What's here is a complete, tested implementation of both tools plus a
free Streamlit demo (bring your own API key). There is no payment, login or
storage of user documents; going from "demo" to "live SaaS product" (pricing,
attorney validation, Stripe, confidentiality handling) is still a deliberate
next step for you to decide on.

## Known limitations (be aware before trusting scores on real cases)

- The rejection scorer is a transparent, auditable **heuristic**, not a
  model trained on labeled office-action outcomes -- because no such
  labeled dataset was obtainable (see above). Treat scores as a structured
  starting point for attorney judgment, not a validated prediction.
- The prior-art matcher measures **lexical** (TF-IDF) overlap, not legal
  anticipation/obviousness -- it's a triage signal for "how much does this
  reference actually overlap the claim," not a substitute for reading the
  reference.
- `LLMResponseGenerator` needs a real key (Anthropic, or any OpenAI-compatible
  endpoint such as OpenAI or NVIDIA); without one, the app falls back to the
  template-only generator (by design -- see
  `test_analyze_endpoint_use_llm_without_api_key_falls_back_to_template`). Argument
  quality varies by model; the generator's guardrail rejects invented citations but
  cannot judge legal soundness.
- Not legal advice. Have a registered patent practitioner review anything before filing.
