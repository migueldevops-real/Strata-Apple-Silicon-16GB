"""serve/test_mlx_backend.py - the MLX backend's logic that needs no GPU and no model.

    python -m unittest serve.test_mlx_backend

The engine's `generate` (mlx_lm + mlx) is exercised only when mlx-lm is importable and a model is given through
STRATA_MLX_TEST_MODEL; the pure parts (prefix reuse math, stop ids, the template, the HF tokenizer adapter) run
everywhere.
"""
from __future__ import annotations

import unittest

from serve.backends import BackendBundle, backend_names, load_backend
from serve.backends.hf_tokenizer import HFTokenizer
from serve.backends.mlx import MlxTemplate, _common_prefix, _stop_ids, _make_processors


class _FakeHF:
    """A tiny byte-per-char tokenizer with two ChatML specials, enough for the adapter and stop-id logic."""

    all_special_tokens = ["<|im_start|>", "<|im_end|>"]

    def __init__(self):
        self.forwarded = []

    def __call__(self, text, add_special_tokens=False, split_special_tokens=False):
        self.forwarded.append((text, split_special_tokens))
        ids = []
        i = 0
        specials = {"<|im_start|>": 200, "<|im_end|>": 201} if not split_special_tokens else {}
        while i < len(text):
            for s, sid in specials.items():
                if text.startswith(s, i):
                    ids.append(sid)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]) % 256 + 3)
                i += 1
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        out = []
        for i in ids:
            out.append("<|im_start|>" if i == 200 else "<|im_end|>" if i == 201 else chr((i - 3) % 256))
        return "".join(out)

    def convert_ids_to_tokens(self, i):
        return "<|im_start|>" if i == 200 else "<|im_end|>" if i == 201 else chr((i - 3) % 256)


class CommonPrefix(unittest.TestCase):
    def test_prefix(self):
        self.assertEqual(_common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(_common_prefix([], [1]), 0)
        self.assertEqual(_common_prefix([1, 2], [1, 2, 3]), 2)


class StopIds(unittest.TestCase):
    def test_only_singletons(self):
        # <|im_end|> is a known stop and one token; <|im_start|> is not a stop
        self.assertEqual(_stop_ids(_FakeHF()), {201})


class TokenizerAdapter(unittest.TestCase):
    def test_round_trip_bytes(self):
        t = HFTokenizer(_FakeHF())
        self.assertEqual(t.encode("ab"), [100, 101])          # ord('a')=97 -> 97%256+3=100
        self.assertEqual(t.token_bytes(100), b"a")
        self.assertEqual(t.decode(t.encode("ab")), "ab")

    def test_plain_span_does_not_parse_specials(self):
        fake = _FakeHF()
        t = HFTokenizer(fake)
        # without plain, the special matches (id 201); with a plain span over it, it is ordinary text
        self.assertIn(201, t.encode("<|im_end|>", parse_special=True))
        ids = t.encode("x<|im_end|>y", parse_special=True, plain=[(1, 10)])
        self.assertNotIn(201, ids)
        self.assertTrue(any(split for _, split in fake.forwarded))   # split_special_tokens was used


class Template(unittest.TestCase):
    SOURCE = ("{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"
              "{% if add_generation_prompt %}assistant:{% endif %}")

    def test_render_and_no_reasoning(self):
        tpl = MlxTemplate(_FakeHF(), self.SOURCE)
        out = tpl.render([{"role": "user", "content": "hi"}])
        self.assertIn("user:hi", out)
        self.assertIn("assistant:", out)
        self.assertFalse(tpl.starts_in_reasoning(out))
        self.assertTrue(tpl.starts_in_reasoning("assistant:\n<think>"))

    def test_tools_passed_to_template(self):
        # the model's template describes the tools; the backend must forward them (not drop them)
        source = "tools={{ tools|length }};{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}"
        out = MlxTemplate(_FakeHF(), source).render([{"role": "user", "content": "hi"}],
                                                    tools=[{"name": "get_weather"}])
        self.assertIn("tools=1", out)


class JsonToolCall(unittest.TestCase):
    """Qwen2.5 writes `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`; the server must parse it."""

    def test_parse_json_call(self):
        from serve.frontend import parse_tool_call
        call = parse_tool_call('{"name": "get_weather", "arguments": {"city": "Oslo"}}')
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.arguments, {"city": "Oslo"})

    def test_parser_emits_a_json_tool_call(self):
        from serve.frontend import OutputParser
        p = OutputParser(thinking=False, tools=[{"name": "get_weather"}])
        evs = []
        evs += p.feed('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>')
        evs += p.finish("stop")
        calls = [e for e in evs if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call.name, "get_weather")
        self.assertEqual(calls[0].call.arguments, {"city": "Oslo"})


class Processors(unittest.TestCase):
    def test_none_when_default(self):
        self.assertEqual(_make_processors({}, None), [])
        self.assertEqual(_make_processors({"temperature": 0.7}, None), [])


class Registry(unittest.TestCase):
    def test_mlx_registered(self):
        self.assertIn("mlx", backend_names())

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            load_backend("nope", {})

    def test_bundle_shape(self):
        b = BackendBundle(engine=object(), tokenizer=object(), template=object(), stop_ids=set())
        self.assertEqual(b.stop_ids, set())


if __name__ == "__main__":
    unittest.main()
