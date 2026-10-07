"""serve/extract.py - text out of an attached document, for the web app's attach button.

The web app reads text files in the browser and sends pictures to the vision encoder; a PDF is neither (it is
binary), so it posts the file here and gets its text back, which is then attached like any other text file.  This
is deliberately dependency-light: `pypdf` (pure Python) only when a PDF actually arrives.  A scanned PDF has no
text layer; the caller gets an empty text and a note (OCR/vision is not this path).
"""
from __future__ import annotations

MAX_EXTRACT_BYTES = 25 * 1024 * 1024        # the endpoint refuses larger uploads


def extract_text(name: str, data: bytes) -> dict:
    """-> {text, pages, chars, note}.  Raises ValueError for an unsupported type, ImportError when pypdf is
    missing (the endpoint turns that into a 501 with the pip line)."""
    if data[:5] == b"%PDF-" or name.lower().endswith(".pdf"):
        return _pdf(data)
    raise ValueError(f"unsupported document type: {name or 'file'}")


def _pdf(data: bytes) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("reading a PDF needs the pypdf package (pip install pypdf)") from e
    import io
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:
        raise ValueError(f"the PDF could not be read ({e})") from None
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")               # an empty-password PDF (common) decrypts; a real one raises below
        except Exception:
            pass
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    text = "\n\n".join(parts).strip()
    note = None if len(text) >= 16 else "little or no text found (a scanned PDF needs OCR, which this path does not do)"
    return {"text": text, "pages": len(reader.pages), "chars": len(text), "note": note}
