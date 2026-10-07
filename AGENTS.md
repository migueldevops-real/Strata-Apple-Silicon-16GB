# AGENTS.md

This is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) that adds an **Apple Silicon (macOS) backend
through MLX**.  It runs a small or medium quantized GGUF / Hugging Face model in-process (`--engine mlx`) behind the
same Python server, web app and OpenAI / Anthropic / Responses API as upstream.

It does **not** run the original 125B Qwen3.8-Flash-Next model (it does not fit unified memory), and the upstream
C++/CUDA/HIP engine under `src/` and `include/` is kept as-is but is not used on macOS.

## What this fork adds

- `serve/backends/` - a backend registry (`register_backend`, `BackendBundle`) and the MLX backend (`mlx.py`,
  `hf_tokenizer.py`).  A backend is one file behind the registry; `--engine mlx` selects it.
- `serve/server.py`, `serve/frontend.py` - `--engine` plugin support, JSON tool calls (Qwen2.5's
  `<tool_call>{"name": ...}</tool_call>` form), and `ChatTemplate(source=...)`.
- `serve/extract.py` + `POST /v1/extract` - attach a PDF as text in the web app (`pypdf`).
- `serve/telemetry.py` - Apple Silicon GPU/memory readings for the Monitor (the chip name and MLX's memory; load,
  temperature and power stay blank, they need `sudo`).
- macOS: `setup-macos.sh`, `run-macos.sh`, the menu bar icon (`strata-menubar.command`, `install-menubar.sh`),
  example run configs (`strata-qwen25-7b-1m.json`, `strata-qwen25-7b-1m-256k.json`) and
  [docs/MACOS_MLX.md](docs/MACOS_MLX.md).
- Tests: `serve/test_mlx_backend.py` (registry, tokenizer adapter, tool-call parsing, template) and
  `serve/test_extract.py` (PDF text).

Still missing in the MLX backend: images/vision, speculative decoding, native batching/parallel requests, and the
upstream expert cache / MTP / n-gram / disk sessions.  Text chat, streaming, sampling, stop tokens, cancellation,
PDF attachments, tool calling and cross-turn prefix reuse work.

## Running it on a Mac

`./setup-macos.sh` (first time; installs MLX and `rumps`) then `./run-macos.sh`; the web app opens on
`http://127.0.0.1:8080`.  Model, context, KV bits and prefill step live in the run config - see
[docs/MACOS_MLX.md](docs/MACOS_MLX.md) for every field and the measured limits (on a 16 GB Mac, KV 8-bit is the
quality-usable ceiling at about 128K; 256K fits only at KV 4-bit, which we measured to degrade answers).  Never
expose the server beyond `127.0.0.1` without `--api-key`.

## Tests

```bash
.venv/bin/python -m unittest serve.test_mlx_backend serve.test_extract   # the new backend (needs mlx-lm / pypdf)
python -m unittest serve.test_server                                     # the whole server suite (mock engine, no model)
```

## Upstream (unchanged)

- Engine internals, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and the
  [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Windows/Linux install of the native 125B engine: [docs/AI_SETUP.md](docs/AI_SETUP.md), `setup.py` started by
  `START-HERE.bat` / `setup.sh`.  That path needs an NVIDIA or AMD GPU and 32 GB+ of RAM; it is not what this fork
  is used for on a Mac.
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py`.

## Style

Plain words; measured numbers with what they were measured on; no claims without a measurement.
