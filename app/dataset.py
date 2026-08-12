"""Prompts and reference completions, held in memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.tokenizer import count_tokens


@dataclass(frozen=True)
class DatasetRecord:
    id: int
    prompt: str
    reference_completion: str
    prompt_tokens: int
    completion_tokens: int


class Dataset:
    def __init__(self, records: dict[int, DatasetRecord]) -> None:
        self._records = records

    def get(self, record_id: int) -> DatasetRecord | None:
        return self._records.get(record_id)

    def ids(self) -> list[int]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)


def load_dataset(path: str | Path, chars_per_token: float) -> Dataset:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"dataset not found: {path} (generate it with scripts/build_dataset.py)"
        ) from exc

    records: dict[int, DatasetRecord] = {}
    for item in json.loads(raw):
        # Token counts are a property of the configured tokenizer, not of the
        # file, so they are derived once here rather than stored in the JSON.
        record = DatasetRecord(
            id=item["id"],
            prompt=item["prompt"],
            reference_completion=item["reference_completion"],
            prompt_tokens=count_tokens(item["prompt"], chars_per_token),
            completion_tokens=count_tokens(item["reference_completion"], chars_per_token),
        )
        records[record.id] = record

    if not records:
        raise ValueError(f"dataset is empty: {path}")
    return Dataset(records)
