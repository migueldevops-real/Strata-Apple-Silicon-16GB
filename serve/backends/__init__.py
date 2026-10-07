"""serve/backends - inference backends behind the token-id Engine protocol.

The resident CUDA/HIP engine stays `--engine strata` (a subprocess).  A backend plugin here runs the model
in-process (no `strata --serve` process) and hands the Service a `BackendBundle`: the engine, its tokenizer,
its chat template and the stop token ids.  The Service only ever sees token ids, so a backend is a local file.

    from serve.backends import register_backend, BackendBundle

    def _open(config):
        return BackendBundle(engine=..., tokenizer=..., template=..., stop_ids={...})
    register_backend("my-backend", _open)

`backend_names()` is imported by serve/server.py to build `--engine`'s choices, so importing this package must
never import a heavy dependency (mlx, llama_cpp): each factory imports its own backend lazily.

The registry concept and the Apple/Metal backend this mirrors come from jinzy0623/Strata-macOS (MIT); see
ATTRIBUTION.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class BackendBundle:
    """What a backend hands the Service: the engine, its tokenizer, its chat template and its stop ids."""
    engine: object
    tokenizer: object
    template: object
    stop_ids: set[int]


_FACTORIES: dict[str, Callable[[dict], BackendBundle]] = {}


def register_backend(name: str, factory: Callable[[dict], BackendBundle]) -> None:
    if name in _FACTORIES:
        raise ValueError(f"backend already registered: {name}")
    _FACTORIES[name] = factory


def backend_names() -> tuple[str, ...]:
    return tuple(_FACTORIES)


def load_backend(name: str, config: dict) -> BackendBundle:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(f"unknown backend: {name}") from None
    return factory(config)


def _mlx(config: dict) -> BackendBundle:
    from .mlx import create_backend
    return create_backend(config)


register_backend("mlx", _mlx)
