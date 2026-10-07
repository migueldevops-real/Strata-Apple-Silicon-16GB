# Attribution

Strata is MIT (see `LICENSE`).  A few parts carry their own notices.

## Upstream (the initial project)

This repository is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata), the original Strata project
(MIT): the C++/CUDA/HIP engine (`src/`, `include/`), the Python server, the OpenAI/Anthropic-compatible API and the
web app (`serve/`), and the Windows/Linux installer.  All of that is theirs.  The Apple Silicon (MLX) backend
described below is the addition made in this fork.

## Apple Silicon (MLX) backend

The backend-registry shape in `serve/backends/` - a `BackendBundle` of `engine`, `tokenizer`, `template` and
`stop_ids`, a `register_backend(name, factory)` table, and the `starts_in_reasoning` hook a model's own chat
template can expose - follows the design of [jinzy0623/Strata-macOS](https://github.com/jinzy0623/Strata-macOS),
which is MIT.

The MLX backend itself (`serve/backends/mlx.py`, `serve/backends/hf_tokenizer.py`) is written for this repository
against `mlx-lm` (MIT, Apple) and the model's own Hugging Face tokenizer and chat template.  No code from
`Strata-macOS` is copied verbatim; the borrowed part is the plugin boundary.

The models are not part of this repository; each model's own license applies to its files (e.g.
`mlx-community/Qwen2.5-7B-Instruct-1M-4bit` is Apache-2.0 from Qwen).
