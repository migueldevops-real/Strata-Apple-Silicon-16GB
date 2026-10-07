"""serve/backends/hf_tokenizer.py - a Hugging Face tokenizer behind the server's tokenizer interface.

The Service talks to a tokenizer through a small interface (built for the pack tokenizer in
tools/strata_tokenizer.py):

    encode(text, parse_special=False, plain=()) -> list[int]
    decode(ids, errors="replace") -> str
    token_bytes(i) -> bytes                       (the raw bytes of one token; Detokenizer's incremental path)
    control_tokens                                 (the CONTROL literals, for literal_tags())

A backend whose model carries its own HF tokenizer (mlx-lm's `load` returns a PreTrainedTokenizerFast) wraps it
here, so the tokenizer, not a re-implementation, decides every id.  `parse_special=False` and the `plain` spans
of #537 are encoded with `split_special_tokens=True`, so a control token written inside a user's message stays
ordinary text instead of opening or ending a turn.
"""
from __future__ import annotations


# GPT-2 byte <-> unicode (the byte-level BPE alphabet), copied from tools/strata_tokenizer.py so this module
# imports nothing heavy and matches that reference exactly.
def _bytes_to_unicode() -> dict[int, str]:
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


UNICODE_TO_BYTE = {v: k for k, v in _bytes_to_unicode().items()}


class HFTokenizer:
    """A PreTrainedTokenizerFast (or anything with the same `__call__` / `decode` / `convert_ids_to_tokens`) as
    the server's tokenizer."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        # the CONTROL literals the template writes (<|im_start|>, <|im_end|>, <|endoftext|>, ...): a copy of one of
        # these inside a message must stay text, so the Service escapes them (literal_tags).
        self.control_tokens = list(getattr(tokenizer, "all_special_tokens", ()) or ())

    # ------------------------------------------------------------------ encode
    def _encode_span(self, text: str, parse_special: bool) -> list[int]:
        if not text:
            return []
        if parse_special:
            out = self.tok(text, add_special_tokens=False)
        else:
            try:
                out = self.tok(text, add_special_tokens=False, split_special_tokens=True)
            except TypeError:               # an older tokenizer without split_special_tokens: best effort
                out = self.tok(text, add_special_tokens=False)
        return list(out["input_ids"])

    def encode(self, text: str, parse_special: bool = False, plain=()) -> list[int]:
        spans = sorted((int(a), int(b)) for a, b in plain if b > a) if plain else []
        if not spans:
            return self._encode_span(text, parse_special)
        out: list[int] = []
        pos = 0
        for a, b in spans:
            a = max(a, pos)
            b = min(b, len(text))
            if a >= b:
                continue
            if a > pos:
                out += self._encode_span(text[pos:a], parse_special)
            out += self._encode_span(text[a:b], False)   # a literal tag: ordinary text
            pos = b
        if pos < len(text):
            out += self._encode_span(text[pos:], parse_special)
        return out

    # ------------------------------------------------------------------ decode
    def decode(self, ids, errors: str = "replace") -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def token_bytes(self, i: int) -> bytes:
        """The raw bytes of one token, so Detokenizer's incremental UTF-8 decoder emits a split character only
        once it is complete.  A byte-level BPE token maps back through the GPT-2 byte alphabet; a special/added
        token is its literal UTF-8 bytes (ASCII, so the byte alphabet also round-trips it)."""
        s = self.tok.convert_ids_to_tokens(int(i))
        if s is None:
            raise IndexError(f"token id {i} is outside the vocabulary")
        if isinstance(s, bytes):
            return s
        try:
            return bytes(UNICODE_TO_BYTE[c] for c in s)
        except KeyError:
            return s.encode("utf-8")

    def __getattr__(self, name):
        # anything else the server may reach for (bos_token, eos_token, chat_template, ...) is the HF tokenizer's
        return getattr(self.tok, name)
