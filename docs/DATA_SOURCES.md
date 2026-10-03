# Data sources: what was actually accessible

The original build guide names several data sources. Here is what actually
happened when this project tried to reach each one from a non-interactive,
scripted environment (no browser, no manual captcha-solving, no paid API
keys) -- so future work doesn't waste time re-treading the same ground.

## USPTO Office Action Research Dataset for Patents (blocked)

This is the guide's best real fit for the Rejection Strength Scorer: bulk
CSVs of real office actions with structured rejection basis codes (101/102/
103/112), examiner IDs, and cited references, 2008-2017.

- Landing page (`www.uspto.gov/.../office-action-research-dataset-patents`)
  loads fine and lists direct download links to `data.uspto.gov/ui/datasets/
  products/files/PTOFFACT/2017/*.csv.zip`.
- Every one of those links, and every USPTO Open Data Portal / Patent Center
  API path tried (`data.uspto.gov/api/...`, `patentcenter.uspto.gov/
  retrieval/...`), returns the same 20KB Angular SPA shell instead of data.
  The page's own `<script>` tags reveal why: `challenge.js` from
  `*.edge.sdk.awswaf.com` -- the whole platform sits behind an AWS WAF
  JavaScript challenge. A scripted HTTP client gets the SPA shell, not the
  API response, because it can't execute or solve the challenge.
- `bulkdata.uspto.gov` (the old static-file mirror some older guides
  reference) no longer resolves in DNS.
- `ped.uspto.gov` (Patent Examination Data System) likewise doesn't resolve.

**Bottom line:** obtaining this dataset requires a real browser session (to
pass the WAF challenge) -- it is not scriptable from here. If you want it,
the practical path is: open the landing page in a real browser, download the
`office_actions.csv.zip` / `rejections.csv.zip` / `citations.csv.zip` files
manually, and drop them in `data/raw/`. The `rejections.csv` schema (statute,
claims, cited patent/applicant references per rejection) would slot directly
into `extraction/office_action_parser.py`'s output shape if you do this.

## PatentsView API

Requires a registered API key (their v1 REST API stopped being fully
anonymous some time ago) and, more importantly, is oriented around granted
patent bibliographic/citation data, not office-action rejection text. Not
pursued.

## Google Patents scraping

Technically reachable, but (a) scraping is against Google's terms for bulk/
automated use, and (b) individual Google Patents pages do not embed full
office-action text even when a "Similar Documents" or legal-events section
is present. Not pursued.

## Harvard USPTO Patent Dataset (HUPD) -- used

`https://huggingface.co/datasets/HUPD/hupd` (Suzgun et al., NeurIPS 2022
Datasets & Benchmarks track, CC-BY-SA-4.0). Hosted as plain static files on
Hugging Face's CDN -- no WAF, no API key, no JS execution needed.

Downloaded: `data/sample-jan-2016.tar.gz` (≈388MB, `scripts/
download_hupd_sample.py`), extracting to 26,808 real, individual patent
application JSON records with:

- Real applicant-authored text: `claims`, `abstract`, `background`,
  `summary`, `full_description`.
- Real prosecution metadata: `examiner_id`/name, `main_cpc_label`/
  `cpc_labels`, `main_ipcr_label`, `filing_date`, `patent_issue_date`.
- Real ground-truth outcome: `decision` ∈ {ACCEPTED, REJECTED, PENDING,
  CONT-ACCEPTED, CONT-REJECTED, CONT-PENDING}.

**What HUPD does not have:** the office action itself -- the examiner's
actual rejection paragraphs, statute citations, and reference numbers. It is
application-level (claims + final outcome), not office-action-level
(per-rejection reasoning). This is a real, documented limitation, not an
oversight:

- The **prior-art matcher** (`scoring/prior_art_matcher.py`) and the
  **claim-breadth evaluation** (`scripts/run_evaluation.py`) use real HUPD
  claim text and real decision outcomes directly -- these results are
  genuine.
- The **office-action parser and rejection scorer**
  (`extraction/office_action_parser.py`, `scoring/rejection_scorer.py`)
  necessarily operate on hand-authored, realistic-but-synthetic office
  action text (`tests/fixtures/office_actions.py`), because no real
  office-action text corpus was scriptable-obtainable. The fixtures mirror
  genuine MPEP/37 CFR examiner phrasing so the parsing/scoring logic is
  exercised realistically, but they are not real USPTO documents, and the
  scorer has not been validated against real per-rejection outcomes.

## USPTO OCE Office Actions (via Kaggle's BigQuery integration) -- used

`https://www.kaggle.com/datasets/bigquery/uspto-oce-office-actions` -- this
is the same underlying real dataset as the blocked one above
(`patents-public-data.uspto_oce_office_actions` on Google BigQuery), but
reached a different way. Two things had to be worked out:

1. **It isn't a downloadable Kaggle dataset.** `kaggle datasets download` /
   `kaggle datasets files` both 404 on it -- `bigquery/`-namespaced Kaggle
   listings are BigQuery-only integrations, not flat files, and there's no
   public GCS mirror either (checked and ruled out).
2. **Kaggle Kernels get free, pre-authenticated BigQuery access to these
   specific integrations.** A kernel with `"dataset_sources":
   ["bigquery/uspto-oce-office-actions"]` in its `kernel-metadata.json` can
   run `google.cloud.bigquery.Client()` with zero credential setup -- Kaggle
   injects it. So the actual approach was: push a script kernel (`kaggle
   kernels push`) that runs the queries and writes CSVs to
   `/kaggle/working/`, poll `kaggle kernels status` until it completes, then
   `kaggle kernels output` to pull the CSVs back. Fully reproducible via
   `scripts/run_kaggle_bigquery_query.sh` (requires a Kaggle API token at
   `~/.kaggle/access_token`).

This gets real, per-rejection structured data -- not office-action prose,
but the fields our parser targets, extracted by USPTO's own Office of Chief
Economist: `rejections.action_type` (101/102/103/112 + non-rejection events
like `objected`/`cancelled`/`allowed`), `rejections.claim_numbers` (comma-
enumerated, e.g. `"6,7,8,11,12,13"`), and `citations.citation_pat_pgpub_id`
(cited reference identifiers). Downloaded to
`data/raw/kaggle_uspto_oce/*.csv` (gitignored, same as HUPD).

**Real bugs this surfaced in `extraction/office_action_parser.py`** (fixed,
regression-tested in `tests/test_office_action_parser.py` and
`tests/test_real_uspto_oce_data.py`):

- The citation regex required a literal `"US"`/`"U.S. Patent No."` prefix.
  Against 5,000 real distinct `citation_pat_pgpub_id` values, it matched
  **0.4%**. Real office-action style routinely cites a later reference by
  name + bare number only (`"Jones (6,234,567)"`, no repeated "US"), and
  cites foreign patents directly (`"JP 2004-159476"`, `"WO 2012136906"`).
  Fixed by allowing the "US" prefix to be omitted when the number is
  parenthesized (a real, common citation style; not relaxed everywhere, to
  avoid confusing an application number like `15/123,456` for a citation),
  and by adding explicit foreign-country-code matching (JP/WO/EP/GB/DE/FR/
  CN/KR/AU/CA/RU). Measured on the subset of real citations that are
  formatted at all (i.e., excluding the ~93% that are bare undelimited
  digits -- a database-storage artifact of this dataset, not how examiners
  write prose, and deliberately not chased): match rate went from 44.1% to
  **49.7%** after also fixing the bug below.
- Separately, the regex's own comment claimed to support US publication
  numbers (`"US 2015/0123456 A1"`), but its digit-shape
  (`\d{1,2}[,/]\d{3}[,/]\d{3}`) could never match one -- publication numbers
  are a 4-digit year + a 7-digit sequence, not three comma/slash-grouped
  groups. Every publication-number citation silently matched nothing, in
  every test fixture, until a real example (`"Lee (US 2009/0139986)"`)
  exposed it. Fixed with a distinct, correct shape for publication numbers.

**What's still not caught, on purpose:** bare, unparenthesized, undelimited
digit runs (`"Pan 5386490"`) and free-text fragments where USPTO's own
extraction leaked surrounding prose into the citation field (e.g. `"Gao CN
1804138A provided with IDS dated 9 May 2014 and citations drawn to the
translation via EPO website"`). Chasing the former risks matching arbitrary
numbers in ordinary prose (dates, dollar amounts, paragraph numbers); the
latter would need LLM-based extraction, not a regex, and mostly reflects
this specific dataset's extraction noise rather than genuine office-action
prose patterns.

## If you later obtain real office-action text

Nothing in the architecture assumes synthetic text -- `parse_office_action()`
and `score_office_action()` operate on any office-action string. If you
manually download real office actions (a handful from Patent Center's UI, or
the full USPTO Office Action Research Dataset via a real browser session),
point them at the same functions and, ideally, replace/extend
`tests/fixtures/office_actions.py` with real (redacted, if needed for
confidentiality) examples so the test suite validates against reality
instead of hand-authored approximations.
