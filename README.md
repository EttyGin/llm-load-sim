# vllm-sim

## What this is

A simulator of a vLLM server running Qwen3-8B on an H100. There is no
GPU: the engine is a queueing model of one, and time passes through
`asyncio.sleep`, so requests contend for real.

## Running it

```bash
uv sync
uv run uvicorn app.main:app
```

```bash
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt_id": 449}'
```

```json
{"prompt_id": 449,
 "usage": {"prompt_tokens": 68, "completion_tokens": 89, "total_tokens": 157},
 "timing_ms": {"queue_wait": 0.03, "prefill": 5.8, "ttft": 5.83,
               "decode": 884.92, "e2e": 890.75}}
```

`/metrics` is Prometheus text, `/health` a probe, `/v1/models` OpenAI-compatible
discovery; `scripts/demo.py` tours the behaviour, driving the engine directly.

## Timing model

```
TTFT    = queue_wait + prefill
e2e     = TTFT + decode
prefill = prompt_tokens / prefill_tokens_per_sec
step_ms = decode_base_step_ms + decode_per_seq_step_ms × running
```

Prefill is compute-bound, linear in prompt tokens. 
Decode is limited by memory, not compute: each step reads the model's weights once, and that single read serves every sequence in the batch — so the base cost is shared, not paid per request.

| constant | value | derivation |
| --- | --- | --- |
| `decode_base_step_ms` | 9.8 | 15.3 GiB weights / 3.35 TB/s ≈ 4.9ms theoretical, ~2× that in practice |
| `decode_per_seq_step_ms` | 0.124 | per-sequence KV read on top of the shared weight read |
| `prefill_tokens_per_sec` | 15000 | ~16 GFLOPs/token at 8.2B, H100 bf16, ~40% MFU |
| `kv_cache_tokens` | 396000 | (79.6 × 0.90 − 15.3 − 2 GiB) / 144KiB per token; 144KiB = 2 × 36 layers × 8 KV heads × 128 head_dim × 2 bytes |
| `chars_per_token` | 3.8 | Qwen3's 152K vocab beats the older families behind the 4-char rule |
| `max_num_seqs` | 256 | vLLM's default cap on the decode batch |
| `max_queue_depth` | 512 | waiters admitted before shedding load with a 503 |

The pool follows vLLM's `gpu_memory_utilization=0.90` default rather than
assuming all 80 GB is free; the 2 GiB is activations and CUDA graphs.

## Design choices

The engine is a plain async library, HTTP and metrics thin adapters, so `demo.py`
load-tests it with no HTTP client's queueing over its own.

Admission uses a `Condition`, not a `Semaphore`: KV reservations vary in size, and
a semaphore counts interchangeable units, so a waiter re-tests a predicate rather
than decrementing a count. Strictly FIFO and head-only — if the head does not fit,
nothing enters, even when a smaller request behind it would. That mirrors vLLM's
scheduler and keeps head-of-line blocking visible rather than hidden behind
backfill. No locks: single-threaded loop, no `await` between check and mutate.

Decode advances 8 tokens per sleep, not one. `asyncio.sleep` overshoots by ~0.6ms,
6% of a 9.9ms step: a 190-step record predicted at 1.886s measures 1.999s
per-token and 1.915s chunked. Step time is still re-read from occupancy per chunk,
so the property worth simulating survives.

The dataset is synthetic and seeded, so I chose the length distribution rather than
inheriting one: 750 records spanning prefill-bound and decode-bound traffic.

| tokens | min | p50 | p90 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| prompt | 67 | 255 | 3932 | 5802 | 6048 |
| completion | 40 | 214 | 871 | 1841 | 1966 |

## What the numbers show

All `scripts/demo.py` output. Prefill and decode are independent axes:

| | prompt | completion | ttft_ms | decode_ms | e2e_ms |
| --- | --- | --- | --- | --- | --- |
| short prompt, short output | 68 | 89 | 5.8 | 886.3 | 892.1 |
| short prompt, long output | 175 | 1966 | 12.5 | 20707.8 | 20720.2 |
| long prompt, short output | 6048 | 83 | 404.4 | 825.0 | 1229.4 |

The ordering inverts in the last two rows: the 6048-token prompt costs 32×
the TTFT and a seventeenth of the e2e — the measured reason both are reported.

A 324-token record decodes at 10.07 ms/token alone against 9.92 predicted, and
34.7–43.2 with 200 resident against 34.7: the batch model is right, the wall-clock
tail is not chunked away.

| concurrent | peak running | peak KV | queue_wait med/p95 | TTFT med/p95 | e2e med/p95 |
| --- | --- | --- | --- | --- | --- |
| 50 | 50/256 | 78804 (20%) | 0.00 / 0.00 | 0.02 / 0.32 | 3.33 / 12.24 |
| 150 | 150/256 | 256320 (65%) | 0.00 / 0.00 | 0.02 / 0.33 | 5.49 / 19.55 |
| 400 | 253/256 | 395988 (100%) | 0.00 / 8.34 | 0.24 / 8.41 | 11.00 / 25.77 |

Seconds. The first two rows matter most: 50 → 150 concurrent degrades median e2e
3.33s → 5.49s, 65%, while p95 queue_wait never leaves 0.00s. Occupancy slows the
batch long before it refuses anyone — a queue-depth alert would not have moved.

Which limit binds is a property of the traffic, not the config. The same 400
requests, three pools:

| KV pool | peak running | peak KV | binds | median e2e |
| --- | --- | --- | --- | --- |
| 150000 | 103/256 | 149995 (100%) | KV pool | 15.24 |
| 396000 | 253/256 | 395988 (100%) | KV pool | 11.00 |
| 900000 | 256/256 | 447519 (50%) | sequence cap | 12.88 |

At 150k the pool refuses everyone: 7.84s median queue_wait, where neither other
pool queues at all. But the configured pool stops three
sequences short of the cap — within ~1% of the crossover, so traffic shifting
shorter hands the constraint to the cap, longer keeps it with the pool. Neither is
the knob to turn here; 900k is inside run-to-run noise of 396k.

## Metrics

Names mirror vLLM's own, so a dashboard written here transfers to a real
deployment. The `sim:` prefix is deliberate — a simulator emitting the real names
silently poisons any dashboard that scrapes both. Seconds throughout, per
Prometheus convention, so `/generate` converts for itself. Latency buckets come
from the runs above; the client defaults stop at 10s, where much of this workload
lands. TPOT's come from the model: step time cannot leave 9.8–41.5ms, so the
buckets divide that span, and one above the ceiling as overflow detection — the
~1% landing there is `asyncio` overshoot, not the model. Queue time only looks
badly bucketed: 78% are admitted on arrival, the rest wait seconds for a slot —
which is why its median reads 0.00s against an 8.4s p95. Gauges are read at scrape
time, not mirrored on state change: a second copy is a second source of truth, and
it drifts. The `size` label on TTFT and e2e is mine, not vLLM's.

## What I'd alert on

In priority order. `e2e` and TTFT p95 against an SLO: the earliest honest signal,
for the reason the load table shows — latency degrades before anything queues, so
everything downstream pages late. `num_requests_waiting` rising monotonically:
arrivals exceed service rate, and the backlog will not drain on its own.
`sim:request_total{status="rejected"}` above zero: load that spilled, and the
trigger for failing over to a hosted provider.

Explicitly not a high `kv_cache_usage_perc` — at saturation it sits at 100%, which
is the goal, not the incident. And readiness must not fail on a full queue:
shedding with a 503 is correct, taking `/health` down with it pulls every
replica out of the load balancer at once.

## Simplifications

KV is reserved in full at admission; vLLM allocates incrementally, so this
under-estimates capacity. Conservative, so I left it.

Prefill does not block decode, so a large prompt's cost shows in the queue, not
everyone's step.

No preemption, hence no `num_preemptions_total`: vLLM evicts and recomputes under
KV pressure, this refuses admission. `decode_per_seq_step_ms` is constant, though
real KV grows during generation.

Wall-clock rather than discrete-event: contention was worth more than determinism,
at the cost of spread — repeats of the same 400 requests agree on the median to
~1%, at p95 only ~6%.

The tokenizer is a character ratio. It drifts on code and non-English text, but a
consistent bias moves the scale, not the shape.
