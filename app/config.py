"""Hardware and model physics for the simulated server; derivations in the README."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = Path("config.json")


class Config(BaseModel):
    # protected_namespaces: model_name is our field, not a pydantic accessor.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    gpu: str

    # ~16 GFLOPs/token for an 8B model, H100 bf16 at ~40% MFU.
    prefill_tokens_per_sec: float = Field(gt=0)
    # ~16GB of weights from HBM at 3.35 TB/s is ~4.8ms; ~2x that with kernel overhead.
    decode_base_step_ms: float = Field(gt=0)
    # Per-sequence KV read on top of the shared weight read.
    decode_per_seq_step_ms: float = Field(ge=0)

    max_num_seqs: int = Field(gt=0)  # vLLM's default decode-batch cap
    # (80GB - 16GB weights - ~4GB overhead) / 128KB per token under GQA.
    kv_cache_tokens: int = Field(gt=0)
    max_queue_depth: int = Field(gt=0)

    chars_per_token: float = Field(gt=0)  # English prose, Llama-3 tokenizer
    dataset_path: str


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"config file not found: {path}") from exc
    return Config.model_validate(json.loads(raw))
