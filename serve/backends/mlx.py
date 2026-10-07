"""serve/backends/mlx.py - the Apple Silicon backend: mlx-lm (MLX / Metal).

This runs a quantized Hugging Face model in-process (no `strata --serve` subprocess) and implements the server's
token-id Engine protocol.  It is not a translation of the CUDA expert cache: MLX supplies the Metal kernels,
unified memory and the quantization; mlx-lm supplies the model, the KV cache and generation.

What it supports: text chat, streaming (the server does it), sampling, stop tokens and cancellation.  What it
does not (yet): tools (a GGUF/HF template's JSON tool calls do not match the server's XML parser), images,
speculative decoding and native batching - the server degrades each one (`hasattr`), as for any in-process engine.

Long context is the point of the MLX backend: the KV cache is quantized (`kv_bits`) and the prompt cache is kept
across requests, so a follow-up prefills only what is new.  `prefill_step_size` bounds peak memory while a long
prompt is read.

    {"model": "mlx-community/Qwen2.5-7B-Instruct-1M-4bit", "context": 262144,
     "kv_bits": 4, "kv_group_size": 64, "quantized_kv_start": 0, "prefill_step_size": 1024}

`kv_bits: null` keeps the KV in fp16.  `rope_scaling` is optional and only needed when the model's own config
does not already carry the long-context rope.
"""
from __future__ import annotations

import json
import os
import platform
import time

from serve.backends import BackendBundle
from serve.backends.hf_tokenizer import HFTokenizer

BACKEND_NAME = "mlx"
# the tokens a chat template may end a turn with; only singletons become stop ids, so a missing special never
# turns ordinary letters into a stop.
KNOWN_STOP_STRINGS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>", "<|eom_id|>")


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class MlxEngine:
    """The token-id engine.  `generate(ids, max_new, sampling, cancel)` yields the new ids one at a time."""

    def __init__(self, model, *, max_context: int, kv_bits=None, kv_group_size: int = 64,
                 quantized_kv_start: int = 0, prefill_step_size: int = 1024, version: str = ""):
        self.model = model
        self.max_context = int(max_context)
        self.kv_bits = int(kv_bits) if kv_bits else None
        self.kv_group_size = int(kv_group_size)
        self.quantized_kv_start = int(quantized_kv_start)
        self.prefill_step_size = int(prefill_step_size)
        self.info = {"backend": BACKEND_NAME, "version": version or "mlx",
                     "architecture": platform.machine(), "expert_streaming": False,
                     "kv_bits": self.kv_bits or 16}
        self.last: dict = {}
        self.progress = None                     # (read, total) while a prompt is read, for the Monitor
        self._cache = None
        self._cache_tokens: list[int] = []

    # ------------------------------------------------------------------ generation
    def _ensure_cache(self):
        from mlx_lm.models.cache import make_prompt_cache
        if self._cache is None:
            self._cache = make_prompt_cache(self.model)
            self._cache_tokens = []
        else:
            # the cache must hold exactly the tokens we think it does, or a prefix comparison is unsafe
            try:
                off = int(self._cache[0].offset)
            except (AttributeError, IndexError):
                off = len(self._cache_tokens)
            if off != len(self._cache_tokens):
                self._cache = make_prompt_cache(self.model)
                self._cache_tokens = []

    def reset(self) -> None:
        self._cache = None
        self._cache_tokens = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if embeddings:
            raise ValueError("images are not supported by the MLX backend")
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import trim_prompt_cache

        ids = [int(t) for t in ids]
        max_new = int(max_new)
        self._ensure_cache()

        # Prefix reuse: keep the cache's common prefix with this prompt and prefill only the rest.  At 256K this
        # is what makes a follow-up turn start in seconds instead of re-reading the whole conversation.
        cp = min(_common_prefix(self._cache_tokens, ids), len(ids))
        if cp == len(ids) and cp > 0:
            cp -= 1                              # generate_step needs at least one prompt token
        drop = len(self._cache_tokens) - cp
        if drop > 0:
            trim_prompt_cache(self._cache, drop)
        del self._cache_tokens[cp:]
        suffix = ids[cp:]
        # the cache now holds `ids`: track the whole prompt (not only what this call generates) so the next
        # request's common-prefix comparison is against the real cache contents.
        self._cache_tokens.extend(suffix)

        base, total = cp, len(ids)

        def on_progress(done, _tot):
            self.progress = (base + done, total)

        sampler = _make_sampler(sampling)
        processors = _make_processors(sampling, mx)
        seed = sampling.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool) and seed > 0:
            mx.random.seed(seed)

        gen = generate_step(
            mx.array(suffix), self.model, max_tokens=max_new, sampler=sampler,
            logits_processors=processors or None, prompt_cache=self._cache,
            prefill_step_size=self.prefill_step_size, kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size, quantized_kv_start=self.quantized_kv_start,
            prompt_progress_callback=on_progress)

        started, first, count = time.monotonic(), None, 0
        self.progress = (base, total)
        try:
            for token, _logprobs in gen:
                if cancel.is_set():
                    break
                if first is None:
                    first = time.monotonic()
                count += 1
                self._cache_tokens.append(int(token))
                yield int(token)
                if count >= max_new:
                    break
        finally:
            gen.close()
            self.progress = None
            ended = time.monotonic()
            self.last = {"generated": count, "prompt_tokens": len(ids), "reused": cp,
                         "prompt_ms": ((first or ended) - started) * 1000,
                         "decode_ms": (ended - (first or ended)) * 1000}

    def close(self):
        self.reset()
        self.model = None


def _make_sampler(sampling: dict):
    from mlx_lm.sample_utils import make_sampler
    temp = sampling.get("temperature")
    temp = float(temp) if isinstance(temp, (int, float)) and not isinstance(temp, bool) else 0.0
    top_p = sampling.get("top_p")
    top_p = float(top_p) if isinstance(top_p, (int, float)) and not isinstance(top_p, bool) else 1.0
    min_p = sampling.get("min_p")
    min_p = float(min_p) if isinstance(min_p, (int, float)) and not isinstance(min_p, bool) else 0.0
    top_k = sampling.get("top_k")
    top_k = int(top_k) if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0 else 0
    return make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)


class _Penalties:
    """repetition / frequency / presence penalties over the last `last_n` tokens, as the server spells them."""

    def __init__(self, mx, repetition=1.0, frequency=0.0, presence=0.0, last_n=64):
        self.mx, self.repetition, self.frequency, self.presence = mx, repetition, frequency, presence
        self.last_n = last_n

    def __call__(self, tokens, logits):
        mx = self.mx
        toks = tokens[-self.last_n:].tolist() if self.last_n else tokens.tolist()
        counts: dict[int, int] = {}
        for t in toks:
            counts[int(t)] = counts.get(int(t), 0) + 1
        if not counts:
            return logits
        idx = mx.array(list(counts.keys()), dtype=mx.int32)
        cnt = mx.array([counts[k] for k in counts], dtype=logits.dtype)
        sel = logits[..., idx]
        if self.frequency:
            sel = sel - self.frequency * cnt
        if self.presence:
            sel = sel - self.presence
        if self.repetition != 1.0:
            sel = mx.where(sel > 0, sel / self.repetition, sel * self.repetition)
        return logits.at[..., idx].add(sel - logits[..., idx])


def _make_processors(sampling: dict, mx):
    rep = sampling.get("repetition_penalty")
    rep = float(rep) if isinstance(rep, (int, float)) and not isinstance(rep, bool) else 1.0
    freq = sampling.get("frequency_penalty")
    freq = float(freq) if isinstance(freq, (int, float)) and not isinstance(freq, bool) else 0.0
    pres = sampling.get("presence_penalty")
    pres = float(pres) if isinstance(pres, (int, float)) and not isinstance(pres, bool) else 0.0
    last_n = sampling.get("penalty_last_n")
    last_n = int(last_n) if isinstance(last_n, int) and not isinstance(last_n, bool) and last_n > 0 else 64
    if rep == 1.0 and freq == 0.0 and pres == 0.0:
        return []
    return [_Penalties(mx, repetition=rep, frequency=freq, presence=pres, last_n=last_n)]


class MlxTemplate:
    """The model's own chat template (from its tokenizer config), rendered by the server's ChatTemplate."""

    def __init__(self, tokenizer, source: str):
        from serve.frontend import ChatTemplate
        self.source = source                      # the server's /props reports the template it renders
        self.template = ChatTemplate(source=source)
        self.kwargs = {}
        for key in ("bos_token", "eos_token"):
            value = getattr(tokenizer, key, None)
            if isinstance(value, str) and value:
                self.kwargs[key] = value

    @staticmethod
    def starts_in_reasoning(prompt: str) -> bool:
        """The server's parser starts in `reasoning` only when the prompt ends inside an open `<think>`.  Qwen2.5
        never writes one, so this is False and answers are content, not reasoning."""
        return prompt.rfind("<think>") > prompt.rfind("</think>")

    def render(self, messages, tools=None, add_generation_prompt: bool = True, **kwargs) -> str:
        # The model's own template describes the tools (Qwen2.5 writes `<tool_call>{"name": ..., "arguments": {...}}
        # </tool_call>`); the server parses that JSON form (frontend.parse_tool_call) and returns or runs the calls.
        return self.template.render(messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **self.kwargs, **kwargs)


def _unwrap_tokenizer(tokenizer):
    """mlx-lm may return a TokenizerWrapper around the HF tokenizer; the adapter needs the HF one."""
    for attr in ("_tokenizer", "tokenizer"):
        inner = getattr(tokenizer, attr, None)
        if inner is not None and hasattr(inner, "convert_ids_to_tokens"):
            return inner
    return tokenizer


def _chat_template_source(tokenizer) -> str | None:
    source = getattr(tokenizer, "chat_template", None)
    if not source and hasattr(tokenizer, "get_chat_template"):
        try:
            source = tokenizer.get_chat_template()
        except Exception:
            source = None
    return source or None


def _stop_ids(hf_tokenizer) -> set[int]:
    stops: set[int] = set()
    eos = getattr(hf_tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        stops.add(eos)
    elif isinstance(eos, (list, tuple)):
        stops.update(int(x) for x in eos)
    for text in KNOWN_STOP_STRINGS:
        try:
            ids = hf_tokenizer(text, add_special_tokens=False)["input_ids"]
        except Exception:
            continue
        if len(ids) == 1:                       # only singletons: a missing special must not become a stop
            stops.add(int(ids[0]))
    return stops


def _apply_rope_scaling(model, scaling) -> None:
    """Best effort: rebuild each attention layer's rope with the given scaling (e.g. YaRN).  Models that already
    carry the right rope in their config do not need this; a failure is a warning, not a start failure."""
    try:
        from mlx_lm.models.rope_utils import initialize_rope
        args = model.args
        args.rope_scaling = scaling
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "rope"):
                continue
            head_dim = getattr(attn, "head_dim", None) or getattr(args, "head_dim", None) \
                or args.hidden_size // args.num_attention_heads
            attn.rope = initialize_rope(head_dim, base=args.rope_theta, traditional=False,
                                        scaling_config=scaling, max_position_embeddings=args.max_position_embeddings)
    except Exception as e:                       # no model internals match: keep the model's own rope
        print(f"[strata] could not apply rope_scaling {scaling!r}: {e}", flush=True)


def _status(phase: str) -> None:
    """Update the startup status file the menu bar icon reads (set by serve/server.py's main).  Best effort."""
    path = os.environ.get("STRATA_STATUS_FILE")
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"phase": phase, "pid": os.getpid(), "at": time.time()}, f)
    except OSError:
        pass


def _warmup(engine, tok) -> None:
    """One short generation at startup, so MLX compiles its Metal kernels and allocates the KV/workspace now,
    instead of on the first chat (whose first token is otherwise much slower).  A failure is a warning, never a
    start failure, and the engine's cache is reset afterwards so the first real request is unaffected."""
    import threading
    import time
    started = time.monotonic()
    _status("warming")
    try:
        ids = tok.encode("Warm up the model before the first request.", parse_special=False)
        for _ in engine.generate(ids, 1, {"temperature": 0.0}, threading.Event()):
            pass
        engine.reset()
        print(f"[strata] warmup: kernels ready in {time.monotonic() - started:.1f}s", flush=True)
    except Exception as e:                       # noqa: BLE001 - a warmup must never stop the server
        print(f"[strata] warmup skipped: {e}", flush=True)
        try:
            engine.reset()
        except Exception:                        # noqa: BLE001
            pass


def create_backend(config: dict, *, metal: bool = True) -> BackendBundle:
    if metal and (platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64")):
        raise ValueError("the MLX backend requires Apple Silicon (native arm64 macOS)")
    model_id = config.get("model")
    if not model_id:
        raise ValueError("config.model is required (an mlx-community HF repo id or a local MLX directory)")
    context = int(config.get("context", 262144))
    if context < 128:
        raise ValueError("context must be at least 128")

    try:
        import mlx_lm
    except ImportError as e:
        raise ValueError("mlx-lm is not installed: pip install mlx-lm") from e
    try:
        model, tokenizer = mlx_lm.load(str(model_id))
    except Exception as e:
        raise ValueError(f"could not load {model_id!r} with mlx-lm: {e}") from e

    if config.get("rope_scaling"):
        _apply_rope_scaling(model, config["rope_scaling"])

    hf_tok = _unwrap_tokenizer(tokenizer)
    source = _chat_template_source(hf_tok if hf_tok is not None else tokenizer)
    if not source:
        raise ValueError("the model carries no chat_template; use an instruct/chat model")

    engine = MlxEngine(model, max_context=context, kv_bits=config.get("kv_bits"),
                       kv_group_size=config.get("kv_group_size", 64),
                       quantized_kv_start=config.get("quantized_kv_start", 0),
                       prefill_step_size=config.get("prefill_step_size", 1024),
                       version=getattr(mlx_lm, "__version__", ""))
    tok = HFTokenizer(hf_tok)
    template = MlxTemplate(hf_tok, source)
    stops = _stop_ids(hf_tok)
    if config.get("warmup", False):
        _warmup(engine, tok)
    return BackendBundle(engine, tok, template, stops)
