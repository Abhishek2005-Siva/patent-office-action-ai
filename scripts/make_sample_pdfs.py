"""Build example office-action PDFs for trying the app's PDF upload.

The text comes from the synthetic fixtures in streamlit_app/samples/*.txt (hand-written to mirror
real MPEP / 37 CFR rejection phrasing; they are NOT real USPTO documents). No dependencies: it
writes a plain multi-page PDF with real, extractable text.

    python scripts/make_sample_pdfs.py
"""
from __future__ import annotations

import io
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "streamlit_app" / "samples"

# name of the .txt sample -> output PDF name
PDFS = {
    "multi_rejection_mixed": "example_office_action_multi_rejection.pdf",
    "weak_103_no_motivation": "example_office_action_weak_103.pdf",
    "strong_103_with_motivation": "example_office_action_strong_103.pdf",
}

PAGE_W, PAGE_H = 612, 792
MARGIN, FONT_SIZE, LEADING, WRAP = 72, 11, 15, 84
LINES_PER_PAGE = (PAGE_H - 2 * MARGIN) // LEADING
FOOTER = "Synthetic example for testing patent-office-action-ai. Not a real USPTO document."


def _esc(line: str) -> str:
    safe = line.encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap(text: str) -> list[str]:
    out: list[str] = []
    for paragraph in text.strip().splitlines():
        out.extend(textwrap.wrap(paragraph, WRAP) or [""])
    return out


def build_pdf(title: str, body: str) -> bytes:
    lines = _wrap(body)
    pages = [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)] or [[]]

    objs: list[bytes] = [b""] * (4 + 2 * len(pages))  # catalog, pages, 2 fonts, then page/content pairs
    objs[0] = b"<</Type/Catalog/Pages 2 0 R>>"
    kids = " ".join(f"{5 + 2 * i} 0 R" for i in range(len(pages)))
    objs[1] = f"<</Type/Pages/Kids[{kids}]/Count {len(pages)}>>".encode()
    objs[2] = b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"
    objs[3] = b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica-Bold>>"

    for i, page_lines in enumerate(pages):
        parts = []
        if i == 0:
            parts.append(f"BT /F2 13 Tf {MARGIN} {PAGE_H - MARGIN + 28} Td ({_esc(title)}) Tj ET")
        parts.append(f"BT /F1 {FONT_SIZE} Tf {LEADING} TL {MARGIN} {PAGE_H - MARGIN} Td")
        for line in page_lines:
            parts.append(f"({_esc(line)}) Tj T*")
        parts.append("ET")
        parts.append(f"BT /F1 8 Tf {MARGIN} 40 Td ({_esc(FOOTER)}) Tj ET")
        content = "\n".join(parts).encode("latin-1")
        page_no, content_no = 5 + 2 * i, 6 + 2 * i
        objs[page_no - 1] = (f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 {PAGE_W} {PAGE_H}]"
                             f"/Resources<</Font<</F1 3 0 R/F2 4 0 R>>>>/Contents {content_no} 0 R>>").encode()
        objs[content_no - 1] = (f"<</Length {len(content)}>>\nstream\n".encode() + content + b"\nendstream")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body_bytes in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body_bytes + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<</Size {len(objs) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF".encode())
    return out.getvalue()


def main() -> None:
    for name, pdf_name in PDFS.items():
        body = (SAMPLES / f"{name}.txt").read_text(encoding="utf-8")
        target = SAMPLES / pdf_name
        target.write_bytes(build_pdf("UNITED STATES PATENT AND TRADEMARK OFFICE - OFFICE ACTION", body))
        print(f"wrote {target.relative_to(ROOT)}  ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
