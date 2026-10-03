"""Flask web app tying the parser, scorer, and response generator together.

Local/dev use only -- no deployment, auth, or payment wiring here (those are
business/infrastructure decisions for the user to make explicitly, not
something to stand up unattended).
"""

from __future__ import annotations

import io
import os

from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader

from patent_ai.analysis import analyze_office_action
from patent_ai.generation.llm_client import AnthropicLLMClient, is_llm_available

MAX_CONTENT_LENGTH = 16 * 1024 * 1024


def _extract_text_from_pdf(file_storage) -> str:
    reader = PdfReader(io.BytesIO(file_storage.read()))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "llm_available": is_llm_available()})

    @app.get("/")
    def index():
        return render_template("index.html", llm_available=is_llm_available())

    @app.post("/analyze")
    def analyze():
        office_action_text = request.form.get("office_action_text", "")

        if "office_action_file" in request.files and request.files["office_action_file"].filename:
            file = request.files["office_action_file"]
            if not file.filename.lower().endswith(".pdf"):
                return jsonify({"error": "office_action_file must be a .pdf"}), 400
            try:
                office_action_text = _extract_text_from_pdf(file)
            except Exception as exc:  # malformed PDF, etc.
                return jsonify({"error": f"Could not read PDF: {exc}"}), 400
        elif request.is_json:
            body = request.get_json(silent=True) or {}
            office_action_text = body.get("office_action_text", office_action_text)

        if not office_action_text or not office_action_text.strip():
            return jsonify({"error": "office_action_text (or office_action_file) is required"}), 400

        claims_text = request.form.get("claims_text", "") or (
            request.get_json(silent=True) or {}
        ).get("claims_text", "")
        spec_text = request.form.get("spec_text", "") or (request.get_json(silent=True) or {}).get(
            "spec_text", ""
        )
        applicant_name = request.form.get("applicant_name") or (
            request.get_json(silent=True) or {}
        ).get("applicant_name", "Applicant")
        use_llm = request.form.get("use_llm") == "on" or (request.get_json(silent=True) or {}).get(
            "use_llm", False
        )

        llm_client = AnthropicLLMClient() if use_llm and is_llm_available() else None
        return jsonify(
            analyze_office_action(
                office_action_text,
                claims_text=claims_text,
                spec_text=spec_text,
                applicant_name=applicant_name,
                llm_client=llm_client,
            )
        )

    return app


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes", "on")
    create_app().run(debug=debug, port=5000)
