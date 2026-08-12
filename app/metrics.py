"""Prometheus metrics. Names mirror vLLM's own, so a scrape from the simulator
lines up field-for-field with a scrape from a real deployment."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from app.config import Config
from app.engine import Engine, GenerationResult

REQUESTS = Counter("sim:request_total", "Requests by terminal status.", ["status"])
PROMPT_TOKENS = Counter("sim:prompt_tokens_total", "Prompt tokens prefilled.")
GENERATION_TOKENS = Counter("sim:generation_tokens_total", "Tokens decoded.")

# Buckets are set from observed runs: the client defaults top out at 10s, and
# most of this workload's e2e latency lands past that.
TTFT = Histogram(
    "sim:time_to_first_token_seconds", "Arrival to first token.", ["size"],
    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10),
)
QUEUE_TIME = Histogram(
    "sim:request_queue_time_seconds", "Time waiting for admission.",
    buckets=(.001, .01, .1, .5, 1, 2.5, 5, 10, 20, 45),
)
E2E = Histogram(
    "sim:e2e_request_latency_seconds", "Arrival to last token.", ["size"],
    buckets=(.5, 1, 2.5, 5, 10, 20, 30, 45, 60, 120),
)
# TPOT's range is bounded by the model, not by observation: step time runs from
# 9.8ms idle to 9.8 + 0.124 × 256 = 41.5ms at the sequence cap. Buckets are spaced
# across that span — roughly one per 40 resident sequences — with a single bucket
# above the ceiling. Measured at 300 concurrent, ~1% lands there: asyncio overshoot,
# not the model, so treat a large overflow count as the signal, not any at all.
TPOT = Histogram(
    "sim:time_per_output_token_seconds", "Decode time per output token.",
    buckets=(.01, .015, .02, .025, .03, .035, .04, .045, .08),
)

RUNNING = Gauge("sim:num_requests_running", "Sequences resident in the decode batch.")
WAITING = Gauge("sim:num_requests_waiting", "Requests queued for admission.")
KV_USAGE = Gauge("sim:kv_cache_usage_perc", "Fraction of the KV pool reserved, 0..1.")


def bind_engine(engine: Engine, config: Config) -> None:
    # Read at scrape time, not mirrored on every state change: a second copy of
    # the engine's counters is a second source of truth, and it would drift.
    RUNNING.set_function(lambda: engine.running)
    WAITING.set_function(lambda: engine.waiting)
    KV_USAGE.set_function(lambda: engine.kv_used / config.kv_cache_tokens)


def _size(prompt_tokens: int) -> str:
    if prompt_tokens < 500:
        return "short"
    if prompt_tokens < 3000:
        return "medium"
    return "long"


def record_success(result: GenerationResult) -> None:
    # Only the latency histograms carry the size label: it is there to separate
    # prefill-dominated TTFT from decode-dominated e2e, and nothing else.
    size = _size(result.prompt_tokens)
    t = result.timings
    REQUESTS.labels(status="success").inc()
    PROMPT_TOKENS.inc(result.prompt_tokens)
    GENERATION_TOKENS.inc(result.completion_tokens)
    TTFT.labels(size=size).observe(t.ttft)
    E2E.labels(size=size).observe(t.e2e)
    QUEUE_TIME.observe(t.queue_wait)
    if result.completion_tokens > 1:
        TPOT.observe(t.decode_time / (result.completion_tokens - 1))


def record_rejected() -> None:
    REQUESTS.labels(status="rejected").inc()


def record_not_found() -> None:
    REQUESTS.labels(status="not_found").inc()
