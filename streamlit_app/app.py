"""
Patent Office Action AI — Streamlit app
Run:  streamlit run streamlit_app/app.py

Paste (or upload) a USPTO office action to get a 0-100 rejection-strength score, an estimated
win probability, a strategy recommendation, and a drafted 37 CFR 1.111 response letter.

Template mode needs no key and runs entirely offline. For model-written arguments, pick a
provider and paste your own key: it is used only for your requests in this browser session.
"""
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from patent_ai.analysis import analyze_office_action  # noqa: E402
from patent_ai.generation.llm_client import AnthropicLLMClient, OpenAICompatibleLLMClient  # noqa: E402

SAMPLES = ROOT / "streamlit_app" / "samples"
TEST_PDFS_URL = "https://github.com/Abhishek2005-Siva/patent-office-action-test-pdfs"
MAX_PDF_MB = 16
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
TEMPLATE = "Template only (no key needed)"

st.set_page_config(page_title="Patent Office Action AI", page_icon="⚖️", layout="wide")

# NVIDIA's hosted lineup changes often (models get retired without notice), so read the live list.
_NON_CHAT = re.compile(
    r"embed|rerank|safety|guard|reward|parse|vlm|vision|clip|retriev|riva|neva|vila|kosmos|"
    r"deplot|fuyu|video|cosmos|ising|starcoder", re.I)
_PREFERRED = [
    "mistralai/mistral-large-2-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/nemotron-nano-3-30b-a3b",
    "openai/gpt-oss-20b",
]


@st.cache_data(ttl=1800, show_spinner=False)
def nvidia_models() -> list[str]:
    """Chat models NVIDIA is serving right now, preferred ones first. Falls back to a short list."""
    try:
        with urllib.request.urlopen(f"{NVIDIA_BASE_URL}/models", timeout=8) as resp:
            ids = [m["id"] for m in json.load(resp)["data"]]
        chat = [i for i in ids if not _NON_CHAT.search(i)]
        first = [m for m in _PREFERRED if m in chat]
        return (first + [i for i in chat if i not in first]) or list(_PREFERRED)
    except Exception:  # noqa: BLE001 - offline or endpoint changed
        return list(_PREFERRED)


PROBE_TIMEOUT = 15       # seconds a model gets to answer the probe
PROBE_SECONDS = 60       # stop looking after this long
FAST_ENOUGH_SECONDS = 4  # stop at the first good model that answers this quickly
PROBE_PROMPT = ("In one sentence, explain why a claim is not anticipated under 35 U.S.C. 102 "
                "if the cited reference lacks one claimed element.")


def probe_model(api_key: str, model: str) -> tuple[str, float, str]:
    """Ask a small drafting question and time it. Returns (state, seconds, detail):
    "good" (a real sentence came back), "callable" (answered but empty, as reasoning models do
    with a small token budget) or "failed" (404 not available to this key, 503, timeout...)."""
    started = time.monotonic()
    try:
        client = OpenAICompatibleLLMClient(api_key=api_key, model=model, base_url=NVIDIA_BASE_URL,
                                           timeout=PROBE_TIMEOUT, max_retries=1)
        text = client.generate(system="You are a patent attorney's assistant.", prompt=PROBE_PROMPT,
                               max_tokens=120)
    except Exception as exc:  # noqa: BLE001
        return "failed", time.monotonic() - started, str(exc)
    seconds = time.monotonic() - started
    return ("good" if len(text.strip()) >= 25 else "callable"), seconds, text


def find_working_model(api_key: str) -> dict:
    """Try NVIDIA models with the visitor's own key and pick the fastest that drafts properly.

    NVIDIA's catalog lists models a given key cannot call (HTTP 404), and free endpoints are
    sometimes overloaded (503), so the only reliable test is a real request."""
    report = {"model": None, "quality": None, "seconds": None, "failures": []}
    started, good, callable_only = time.monotonic(), [], []
    for model in nvidia_models():
        if time.monotonic() - started > PROBE_SECONDS or len(good) >= 3:
            break
        state, seconds, detail = probe_model(api_key, model)
        if state == "failed":
            report["failures"].append(f"{model}: {detail[:140]}")
        elif state == "good":
            good.append((seconds, model))
            if seconds <= FAST_ENOUGH_SECONDS:
                break
        else:
            callable_only.append((seconds, model))
            report["failures"].append(f"{model}: answered with nothing usable ({seconds:.0f}s)")
    pool, quality = (good, "good") if good else (callable_only, "untested")
    if pool:
        report["seconds"], report["model"] = min(pool)
        report["quality"] = quality
    return report


PROVIDERS = {
    TEMPLATE: None,
    "NVIDIA (free)": {"hint": "nvapi-…  (free key at build.nvidia.com)"},
    "OpenAI": {"hint": "sk-…", "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]},
    "Anthropic (Claude)": {"hint": "sk-ant-…",
                           "models": ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-4-5-20251001"]},
}


def build_llm_client(provider: str, api_key: str, model: str):
    if provider == "NVIDIA (free)":
        return OpenAICompatibleLLMClient(api_key=api_key, model=model, base_url=NVIDIA_BASE_URL, timeout=90)
    if provider == "OpenAI":
        return OpenAICompatibleLLMClient(api_key=api_key, model=model, timeout=90)
    return AnthropicLLMClient(model=model, api_key=api_key)


def pdf_to_text(upload) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(upload.getvalue()))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def sidebar() -> tuple[str, str, str]:
    # A previous "Find a working model" run chose a model; apply it before the widgets are built.
    pending = st.session_state.pop("pending_model", None)
    if pending:
        st.session_state["model_sel"] = pending

    with st.sidebar:
        st.title("⚖️ Office Action AI")
        st.caption("Score how strong an examiner's rejection is, estimate your odds, and draft a response.")
        provider = st.selectbox("Argument drafting", list(PROVIDERS))
        api_key, model = "", ""
        if PROVIDERS[provider]:
            cfg = PROVIDERS[provider]
            api_key = st.text_input("API key", type="password", placeholder=cfg["hint"],
                                    help="Used only for your requests in this browser session. Never stored."
                                    ).strip()
            # The field only registers when you press Enter or click away, so show what the app sees.
            if api_key:
                st.caption("✅ Key received.")
            else:
                st.caption("⚠️ No key yet. Paste it, then press **Enter**.")
            models = nvidia_models() if provider == "NVIDIA (free)" else cfg["models"]
            model = st.selectbox("Model", models, key="model_sel")
            if provider == "NVIDIA (free)":
                if st.button("Find a working model", disabled=not api_key,
                             help="Some models in NVIDIA's catalog aren't available to every key (404). "
                                  "This tries them with your key and picks the fastest that works."):
                    with st.spinner("Trying models with your key (a few seconds)..."):
                        report = find_working_model(api_key)
                    st.session_state["probe_report"] = report
                    if report["model"]:
                        st.session_state["pending_model"] = report["model"]
                    st.rerun()
                report = st.session_state.get("probe_report")
                if report:
                    if report["model"]:
                        st.success(f"Using {report['model']}.")
                        if report["quality"] == "untested":
                            st.warning("It responded but returned little text, so letters may be slow "
                                       "or fall back to the template.")
                    else:
                        st.error("No model worked for this key. Check the key at build.nvidia.com.")
                    if report["failures"]:
                        with st.expander(f"Models that didn't work ({len(report['failures'])})"):
                            for line in report["failures"]:
                                st.text(line)
            st.caption("Your office action text is sent to this provider to draft arguments.")
        else:
            st.caption("The scoring and a template-based letter run fully offline.")
        st.divider()
        st.warning("**Not legal advice.** Scores come from a transparent heuristic, not a model "
                   "trained on labeled outcomes. Have a registered patent practitioner review "
                   "anything before filing.", icon="⚠️")
        st.markdown("[Source on GitHub](https://github.com/Abhishek2005-Siva/patent-office-action-ai)")
        st.markdown(f"[Example test PDFs]({TEST_PDFS_URL})")
    return provider, api_key, model


def load_sample() -> None:
    choice = st.session_state.get("sample_choice")
    if choice and choice != "—":
        st.session_state["oa_text"] = (SAMPLES / f"{choice}.txt").read_text(encoding="utf-8")


def inputs() -> dict:
    st.header("1. The office action")
    st.caption("Text you paste is processed in this session. Don't paste confidential, unpublished "
               "matter into a shared demo. The samples are synthetic.")
    names = ["—"] + sorted(p.stem for p in SAMPLES.glob("*.txt"))
    st.selectbox("Load a synthetic sample", names, key="sample_choice", on_change=load_sample,
                 format_func=lambda n: n.replace("_", " "))
    text_tab, pdf_tab = st.tabs(["Paste text", "Upload PDF"])
    with text_tab:
        st.text_area("Office action text", key="oa_text", height=260,
                     placeholder="Paste the examiner's rejection text here…")
    uploaded_text = ""
    with pdf_tab:
        st.markdown(f"No office action handy? [Download example test PDFs on GitHub →]({TEST_PDFS_URL})")
        pdf = st.file_uploader("Office action PDF", type=["pdf"])
        if pdf is not None:
            if pdf.size > MAX_PDF_MB * 1024 * 1024:
                st.error(f"PDF is larger than {MAX_PDF_MB} MB.")
            else:
                try:
                    uploaded_text = pdf_to_text(pdf)
                    st.success(f"Read {len(uploaded_text):,} characters from the PDF.")
                except Exception as exc:  # noqa: BLE001 - malformed PDF
                    st.error(f"Could not read that PDF: {exc}")

    c1, c2 = st.columns([1, 2])
    applicant = c1.text_input("Applicant name (for the letter)", value="Applicant")
    with st.expander("Your claims and specification (used when a model drafts the arguments)"):
        claims = st.text_area("Claims", height=140)
        spec = st.text_area("Specification excerpts", height=140)
    return {"text": uploaded_text or st.session_state.get("oa_text", ""),
            "applicant": applicant.strip() or "Applicant", "claims": claims, "spec": spec}


def signature(form: dict, provider: str, api_key: str, model: str) -> tuple:
    """Everything that changes what an analysis would produce, to tell when a shown result is stale."""
    return (form["text"], form["applicant"], form["claims"], form["spec"], provider, bool(api_key), model)


def run_analysis(form: dict, provider: str, api_key: str, model: str) -> None:
    notice, client = None, None
    if PROVIDERS[provider]:
        if not api_key:
            st.session_state["result"] = None
            st.error(f"You chose {provider} but no API key has been received. Paste the key in the sidebar "
                     "and press **Enter** (then click Analyze again), or switch to "
                     "'Template only'.")
            return
        try:
            client = build_llm_client(provider, api_key, model)
        except Exception as exc:  # noqa: BLE001
            notice = f"Could not set up {provider}: {exc}. Used the offline template letter."
    kwargs = dict(claims_text=form["claims"], spec_text=form["spec"], applicant_name=form["applicant"])
    try:
        result = analyze_office_action(form["text"], llm_client=client, **kwargs)
    except Exception as exc:  # noqa: BLE001 - provider error: keep the user's result
        hint = ""
        if provider == "NVIDIA (free)" and any(code in str(exc) for code in ("404", "410")):
            hint = " That model isn't available to your key: click **Find a working model** in the sidebar."
        notice = f"{provider} failed ({str(exc)[:200]}). Showing the offline template letter instead.{hint}"
        result = analyze_office_action(form["text"], llm_client=None, **kwargs)
    st.session_state["result"] = result
    st.session_state["notice"] = notice
    st.session_state["result_signature"] = signature(form, provider, api_key, model)


def show_result(result: dict, notice: str | None) -> None:
    st.header("2. Analysis")
    if notice:
        st.warning(notice)
    if not result["rejection_types"]:
        st.warning("No rejections were recognised in that text. Check that it is the body of an "
                   "office action (statutes such as 35 U.S.C. § 102 or § 103 and claim numbers).")

    a, b, c = st.columns(3)
    a.metric("Rejection strength", f"{result['overall_strength']} / 100",
             help="0 = very weak rejection, 100 = very strong")
    b.metric("Estimated chance of overcoming", f"{result['win_probability'] * 100:.0f}%")
    c.metric("Letter written by", "A language model" if result["used_llm"] else "Template")
    st.progress(min(max(result["overall_strength"], 0), 100) / 100)
    st.info(result["recommendation"])

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Application:** {result['application_number'] or 'not found'}")
        st.markdown(f"**Type:** {result['office_action_type'] or 'unknown'}")
        st.markdown("**Statutes:** " + (", ".join(f"§ {s}" for s in result["rejection_types"]) or "—"))
    with right:
        st.markdown("**Rejected claims:** " + (", ".join(map(str, result["rejected_claims"])) or "—"))
        st.markdown("**Cited references:** " + (", ".join(result["cited_references"]) or "—"))

    if result["per_rejection"]:
        st.subheader("Per-rejection breakdown")
        for rej in result["per_rejection"]:
            claims = ", ".join(map(str, rej["claims_rejected"]))
            with st.expander(f"§ {rej['statute']} · claims {claims} · strength {rej['strength']}/100"):
                for reason in rej["reasoning"]:
                    st.markdown(f"- {reason}")

    st.subheader("Draft response letter")
    st.text_area("Letter", value=result["response_letter"], height=520, label_visibility="collapsed")
    st.download_button("Download letter (.txt)", result["response_letter"],
                       file_name="office_action_response.txt", mime="text/plain")


def main() -> None:
    provider, api_key, model = sidebar()
    form = inputs()
    if st.button("Analyze", type="primary", disabled=not form["text"].strip()):
        with st.spinner("Analyzing…"):
            run_analysis(form, provider, api_key, model)
    if st.session_state.get("result"):
        stale = st.session_state.get("result_signature") != signature(form, provider, api_key, model)
        if stale:
            st.info("The text or settings have changed since this analysis. Click **Analyze** to refresh it.")
        show_result(st.session_state["result"], None if stale else st.session_state.get("notice"))


main()
