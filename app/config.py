"""Hardware and model physics for the simulated server: realistic, conservative
values for this hardware rather than derived ones. Disagree with one and re-run."""

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

    # Sustained prefill rate for an 8B model on one H100.
    prefill_tokens_per_sec: float = Field(gt=0)
    # One decode step with a single sequence resident.
    decode_base_step_ms: float = Field(gt=0)
    # What each further resident sequence adds to that step.
    decode_per_seq_step_ms: float = Field(ge=0)

    max_num_seqs: int = Field(gt=0)  # vLLM's default decode-batch cap
    # Tokens the pool holds once weights and working memory are off the top, at
    # vLLM's gpu_memory_utilization=0.90 default rather than the whole card.
    kv_cache_tokens: int = Field(gt=0)
    max_queue_depth: int = Field(gt=0)

    chars_per_token: float = Field(gt=0)  # characters per token, Qwen3's tokenizer
    dataset_path: str


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"config file not found: {path}") from exc
    return Config.model_validate(json.loads(raw))
