# Strata on Apple Silicon (MLX backend)

The server (`serve/`) can run a small/medium quantized model **in-process** on Apple Silicon through
[mlx-lm](https://github.com/ml-explore/mlx-lm), with no CUDA/HIP engine and no `strata --serve` subprocess.  It keeps
the whole Strata surface: the OpenAI chat/completions API, the Anthropic messages API, the Responses API, the web
app, MCP and the monitor.  The engine boundary stays token ids, so the backend is one file
(`serve/backends/mlx.py`) behind a registry (`serve/backends/__init__.py`).

This does **not** port the CUDA expert cache or the Qwen3.8-Flash-Next model: MLX runs a normal dense model that
fits unified memory.  It is a separate backend, off unless `--engine mlx` is chosen.

## What this fork changes

Everything below is added around upstream `serve/`; the native engine (`src/`, `include/`) is untouched.

- **Backend registry** (`serve/backends/`): a `BackendBundle` of engine/tokenizer/template/stop_ids and
  `register_backend`, so `--engine mlx` (or a future backend) is one file.
- **MLX backend** (`serve/backends/mlx.py`): in-process `mlx-lm`; quantized KV (`kv_bits`/`kv_group_size`/
  `quantized_kv_start`); `prefill_step_size` to bound prefill memory; **cross-turn prefix reuse**; sampling
  (temperature/top_p/top_k/min_p, penalties, seed) and cancellation.
- **Tokenizer adapter** (`serve/backends/hf_tokenizer.py`): the model's own Hugging Face tokenizer behind the
  server's tokenizer interface (including the `plain` spans and `token_bytes` the server uses).
- **`--engine` plugin support** and `ChatTemplate(source=...)` in `serve/server.py` / `serve/frontend.py`.
- **JSON tool calling**: the parser accepts Qwen2.5's `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`;
  OpenAI `tool_calls`, Anthropic `tool_use` and MCP work, including the `role: "tool"` follow-up.
- **PDF attachments**: `serve/extract.py` + `POST /v1/extract` (via `pypdf`), attached as text from the web app.
  Documents **accumulate across a chat** and a short note in the prompt makes "this document"/"the document" mean the
  most recently attached one; a scanned PDF (no text layer) reports that instead.
- **Monitor**: Apple Silicon readings in `serve/telemetry.py` (the chip name and MLX memory against the wired limit;
  the CPU row shows the real chip).  The recent-requests table's `reused` column reports the prefix reuse.
- **Menu bar icon** (`tools/strata_menubar.py`, `strata-menubar.command`, `install-menubar.sh`) and the macOS
  `setup-macos.sh` / `run-macos.sh` scripts with example configs.
- Docs (`docs/MACOS_MLX.md`, `ATTRIBUTION.md`) and tests (`serve/test_mlx_backend.py`, `serve/test_extract.py`).

## Install and run

Requires Apple Silicon and macOS.  `setup-macos.sh` makes a `.venv`, installs the requirements and `mlx-lm`, and
starts the server; `run-macos.sh` starts it again from an existing `.venv`.

```bash
./setup-macos.sh                          # first time (downloads the model on first start, ~4.3 GB)
./run-macos.sh                            # later
CONFIG=strata-qwen25-7b-1m-256k.json ./run-macos.sh   # the 256K attempt (see below)
```

The web app is at `http://127.0.0.1:8080`; the OpenAI base URL is `http://127.0.0.1:8080/v1`.

## The run config

A JSON file passed with `--config`.  Keys:

| key | meaning |
| --- | --- |
| `model` | an `mlx-community/...` Hugging Face id or a local MLX directory (must carry a `chat_template`) |
| `context` | the context window in tokens (`max_context`) |
| `kv_bits` | KV cache bits: `null` = fp16, `8` or `4` = quantized |
| `kv_group_size` | KV quantization group size (default 64) |
| `quantized_kv_start` | quantize the KV only from this step on (default 4096; keeps the first tokens exact) |
| `prefill_step_size` | prompt tokens processed per step: smaller = lower peak memory |
| `model_name`, `aliases`, `host`, ... | as for the other engines |

## What works, what does not

Works: text chat, streaming (SSE), sampling (temperature/top_p/top_k/min_p and penalties), stop tokens,
cancellation, **prefix reuse across turns** (the KV of the previous turn is kept, so a follow-up prefills only
what is new), **PDF attachments** (the web app posts a PDF to `/v1/extract`; the server returns its text with
`pypdf` and it is attached like any text file - a scanned PDF has no text layer and says so), and **tool calling**.
The model's own template describes the tools and Qwen2.5 writes them as
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`; the server parses that JSON form and returns the calls
the way each API expects - OpenAI `tool_calls`, Anthropic `tool_use` - including the follow-up: a `role: "tool"`
result goes back in and the model answers.  MCP tools run through the server's MCP path as before.  The server
already degrades what a backend lacks: `unload`/`restart`/VRAM resize/disk sessions are reported unsupported,
`/slots` returns 501.

Not yet: **images** and speculative decoding.

## Measured (MacBook Pro M2 Pro, 16 GB, macOS, mlx-lm 0.32.0, Qwen2.5-7B-Instruct-1M-4bit)

Measured while the machine still had other programs open and was already swapping (~14 GB), so the timings are
pessimistic; the memory ceiling is what matters.

- Short chat and multi-turn: correct answers ("1..5", "6..8", "144"); prefix reuse makes follow-up prefill ~146 ms
  instead of re-reading the conversation.
- Needle-in-a-haystack, ~8.8K tokens: **`kv_bits: null` and `kv_bits: 8` retrieve the needle**; **`kv_bits: 4`
  does not** (it returns garbage).  A ~2.2K needle is retrieved at 4-bit too.
- Peak MLX memory while reading ~26K tokens: ~8.6 GiB.

## The 256K question

Qwen2.5-7B-Instruct-1M supports ~1M tokens, but **1M and even 256K do not fit 16 GB at usable quality**:

- KV per token is 0.0547 MiB (fp16), ~0.029 (8-bit), ~0.0154 (4-bit).  At **256K**: 13.7 / 7.3 / 3.8 GiB.
- **4-bit** 256K fits the memory but the measurement above says the quality is not usable.
- **8-bit** 256K does not fit (7.3 GiB KV + 4.3 GiB weights + macOS > 16 GB).

So on 16 GB the quality-usable ceiling is about **128K with `kv_bits: 8`** (the shipped
`strata-qwen25-7b-1m.json`), and that already needs a relatively free machine.  `strata-qwen25-7b-1m-256k.json`
(256K, 4-bit) is kept for experiments and is expected to degrade.

A second limit is speed: mlx-lm implements **no Dual Chunk Attention** (Qwen2.5-1M's efficient long-context
attention), so a long prompt is read with ordinary quadratic attention.  The first read of a very long prompt is
slow (it is paid once per conversation; prefix reuse keeps later turns cheap).

## Menu bar icon

`install-menubar.sh` installs an icon in the macOS menu bar and starts it now and at every login (a LaunchAgent,
`~/Library/LaunchAgents/com.strata.menubar.plist`):

```bash
./install-menubar.sh              # install + start (also at login)
./install-menubar.sh --uninstall  # remove it
```

It also runs `strata-menubar.command` (double-click) for a one-off start without installing anything.  The icon
(`tools/strata_menubar.py`, needs `rumps`, installed by `setup-macos.sh`) shows whether the model is loaded and lets
you open the web app and start/stop/restart the server without a terminal:

- the status line reads `Strata — <model> · <context>` when it is up, `Strata — stopped` when it is not;
- **Open the web app**, **Start**, **Stop**, **Restart**, **Quit**.

It only talks HTTP (`/health`) and starts the server as a child process; it never loads the model in the icon
itself.  Quitting it from its own menu stops it until the next login (the agent does not keep it alive).

## The Monitor (dashboard)

The web app's **Monitor** tab reads `/metrics`.  On Apple Silicon the GPU rows carry the MLX runtime's own numbers
instead of dashes:

- **GPU / VRAM**: the Apple GPU's name (`Apple M2 Pro (Apple GPU)`) and MLX memory - the model's weights, the KV
  cache and working buffers - against the GPU's recommended working-set size (the wired limit), e.g. `4.0 / 11.8 GB`.
  The CPU row also shows the real chip name rather than `arm`.
- **Load, temperature and power** stay `–`: macOS exposes them only through `powermetrics`, which needs `sudo`, and
  a guess would be worse than an honest dash.
- CPU %, RAM, disk, context fill, tok/s and the recent-requests table work as for any backend.

## Diagnosing

`STRATA_ACCESS_LOG=1` makes the server print every request line and its status as it arrives (which path reached
it); it is what settles "is this 404 the server's or the browser's?".  The server's own log is `strata-<model>.log`
(written by the run scripts) and the menu bar's is `/tmp/strata-menubar.log`.

## Credits

The backend-registry shape (a `BackendBundle` of engine/tokenizer/template/stop_ids, `register_backend`) follows
[jinzy0623/Strata-macOS](https://github.com/jinzy0623/Strata-macOS), MIT.  See `ATTRIBUTION.md`.
