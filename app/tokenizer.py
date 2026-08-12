"""Character-ratio approximation, not a real tokenizer.

It undercounts wherever the BPE vocabulary fragments more than the ratio
suggests — code and markup, non-Latin scripts, rare words, long numbers,
URLs — and overcounts long words that are a single token.
"""

from __future__ import annotations

import math


def count_tokens(text: str, chars_per_token: float) -> int:
    return max(1, math.ceil(len(text) / chars_per_token))
