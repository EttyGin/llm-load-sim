"""A queueing model of a GPU, not a GPU: a fixed KV-cache pool, a cap on
concurrent sequences, and a decode step that slows as occupancy rises."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

from app.config import Config
from app.dataset import DatasetRecord

# One sleep per token would add asyncio's ~0.6ms of scheduling overhead to every
# 9.9ms step, inflating decode by 6%; 8 tokens per sleep amortises it to ~1.6%.
DECODE_CHUNK_TOKENS = 8


class QueueFullError(Exception):
    pass


@dataclass(frozen=True)
class Timings:
    queue_wait: float
    prefill_time: float
    ttft: float
    decode_time: float
    e2e: float


@dataclass(frozen=True)
class GenerationResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    timings: Timings


class Engine:
    def __init__(self, config: Config) -> None:
        self._config = config
        # A Condition, not a Semaphore: KV reservations vary in size per request.
        self._cond = asyncio.Condition()
        self._waiters: deque[object] = deque()

        self.running = 0
        self.kv_used = 0
        self.peak_running = 0
        self.peak_kv_used = 0

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    # No locks: single-threaded event loop, no await between check and mutate.
    def _has_room(self, kv_tokens: int) -> bool:
        return (
            self.running < self._config.max_num_seqs
            and self.kv_used + kv_tokens <= self._config.kv_cache_tokens
        )

    @asynccontextmanager
    async def _admit(self, kv_tokens: int) -> AsyncGenerator[None]:
        cfg = self._config
        if kv_tokens > cfg.kv_cache_tokens:
            raise ValueError(f"request needs {kv_tokens} KV tokens, pool holds {cfg.kv_cache_tokens}")
        if len(self._waiters) >= cfg.max_queue_depth:
            raise QueueFullError(f"queue is full ({cfg.max_queue_depth} waiting)")

        ticket = object()
        self._waiters.append(ticket)
        admitted = False
        try:
            async with self._cond:
                # Head-only admission: no starvation of large requests, and HOL blocking stays visible.
                await self._cond.wait_for(
                    lambda: self._waiters[0] is ticket and self._has_room(kv_tokens)
                )
                self._waiters.popleft()
                admitted = True
                self.running += 1
                self.kv_used += kv_tokens
                self.peak_running = max(self.peak_running, self.running)
                self.peak_kv_used = max(self.peak_kv_used, self.kv_used)
            yield
        finally:
            async with self._cond:
                if admitted:
                    self.running -= 1
                    self.kv_used -= kv_tokens
                else:
                    self._waiters.remove(ticket)
                # notify_all, not notify: one release may admit several small requests.
                self._cond.notify_all()

    async def generate(self, record: DatasetRecord) -> GenerationResult:
        cfg = self._config
        arrived = time.perf_counter()

        async with self._admit(record.prompt_tokens + record.completion_tokens):
            admitted = time.perf_counter()

            # Prefill: compute-bound.
            await asyncio.sleep(record.prompt_tokens / cfg.prefill_tokens_per_sec)
            first_token = time.perf_counter()

            # Decode: memory-bandwidth-bound, and the step cost is re-read from
            # current occupancy so a request slows as others join the batch.
            remaining = record.completion_tokens - 1
            while remaining > 0:
                tokens = min(DECODE_CHUNK_TOKENS, remaining)
                step_ms = cfg.decode_base_step_ms + cfg.decode_per_seq_step_ms * self.running
                await asyncio.sleep(tokens * step_ms / 1000)
                remaining -= tokens
            finished = time.perf_counter()

        return GenerationResult(
            content=record.reference_completion,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            timings=Timings(
                queue_wait=admitted - arrived,
                prefill_time=first_token - admitted,
                ttft=first_token - arrived,
                decode_time=finished - first_token,
                e2e=finished - arrived,
            ),
        )
