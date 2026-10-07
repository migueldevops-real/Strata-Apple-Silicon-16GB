"""serve/test_extract.py - a PDF's text out of /v1/extract's helper (no browser, no GPU).

    python -m unittest serve.test_extract

The PDF is a minimal one-object Page with a text object, embedded below so the test needs no fixture file.  pypdf
is only needed for the PDF case; the unsupported-type check runs without it.
"""
from __future__ import annotations

import base64
import unittest

from serve.extract import extract_text

# %PDF-1.4, one page, one text object: "Hello PDF secret 7391"
MINI_PDF_B64 = (
    "JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2JqCjIgMCBvYmoKPDwg"
    "L1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2JqCjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2Ug"
    "L1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCAyMDAgMjAwXSAvUmVzb3VyY2VzIDw8IC9Gb250IDw8IC9GMSA0IDAg"
    "UiA+PiA+PiAvQ29udGVudHMgNSAwIFIgPj4KZW5kb2JqCjQgMCBvYmoKPDwgL1R5cGUgL0ZvbnQgL1N1YnR5cGUgL1R5"
    "cGUxIC9CYXNlRm9udCAvSGVsdmV0aWNhID4+CmVuZG9iago1IDAgb2JqCjw8IC9MZW5ndGggNTIgPj4Kc3RyZWFtCkJU"
    "IC9GMSAyNCBUZiAyMCAxMDAgVGQgKEhlbGxvIFBERiBzZWNyZXQgNzM5MSkgVGogRVQKZW5kc3RyZWFtCmVuZG9iagp4"
    "cmVmCjAgNgowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMDkgMDAwMDAgbiAKMDAwMDAwMDA1OCAwMDAwMCBuIAow"
    "MDAwMDAwMTE1IDAwMDAwIG4gCjAwMDAwMDAyNDEgMDAwMDAgbiAKMDAwMDAwMDMxMSAwMDAwMCBuIAp0cmFpbGVyCjw8"
    "IC9TaXplIDYgL1Jvb3QgMSAwIFIgPj4Kc3RhcnR4cmVmCjQxMwolJUVPRgo="
)


class Extract(unittest.TestCase):
    def test_unsupported_type_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_text("notes.txt", b"plain text is handled by the web app, not here")

    def test_pdf_text(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("pypdf is not installed")
        data = base64.b64decode(MINI_PDF_B64)
        out = extract_text("mini.pdf", data)
        self.assertIn("7391", out["text"])
        self.assertEqual(out["pages"], 1)
        self.assertIsNone(out["note"])

    def test_magic_bytes_without_extension(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("pypdf is not installed")
        out = extract_text("upload", base64.b64decode(MINI_PDF_B64))
        self.assertIn("secret", out["text"])


if __name__ == "__main__":
    unittest.main()
