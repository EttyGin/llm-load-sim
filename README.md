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

| constant | value | what it is |
| --- | --- | --- |
| `decode_base_step_ms` | 9.8 | one decode step with a single sequence resident |
| `decode_per_seq_step_ms` | 0.124 | what each further resident sequence adds to that step |
| `prefill_tokens_per_sec` | 15000 | sustained prefill rate for an 8B model on one H100 |
| `kv_cache_tokens` | 396000 | tokens the KV pool holds once weights and working memory are off the top |
| `chars_per_token` | 3.8 | characters per token, converting the dataset's text to token counts |
| `max_num_seqs` | 256 | vLLM's default cap on the decode batch |
| `max_queue_depth` | 512 | waiters admitted before shedding load with a 503 |

These are realistic, conservative values for this hardware and model, not
derivations; they live in `config.json`, so disagreeing means changing a number
and re-running. The pool follows vLLM's `gpu_memory_utilization=0.90` default,
not the whole card.

## Design choices

The engine is a plain async library, HTTP and metrics thin adapters, so `demo.py`
load-tests it without an HTTP client queueing over its own.

Admission uses a `Condition`, not a `Semaphore`: KV reservations vary in size where
a semaphore counts interchangeable units, so a waiter re-tests a predicate rather
than decrementing a count. Strictly FIFO and head-only — if the head does not fit,
nothing enters, even when a smaller request behind it would. That mirrors vLLM's
scheduler and keeps head-of-line blocking visible rather than hidden behind
backfill. No locks: single-threaded loop, no `await` between check and mutate.

Decode advances 8 tokens per sleep, not one: `asyncio.sleep` overshoots by ~0.6ms,
6% of a 9.9ms step, and chunking amortises that rather than paying it per token.
Step time is still re-read from occupancy per chunk, so the property worth
simulating survives.

The dataset is synthetic: 750 records spanning long documents with short answers,
prefill-bound, and short prompts with long generations, decode-bound — the two
regimes the simulator exists to tell apart.

| tokens | min | p50 | p90 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| prompt | 67 | 255 | 3932 | 5802 | 6048 |
| completion | 40 | 214 | 871 | 1841 | 1966 |

## What the numbers show

All `demo.py` output. Prefill and decode are independent axes:

| | prompt | completion | ttft_ms | decode_ms | e2e_ms |
| --- | --- | --- | --- | --- | --- |
| short prompt, short output | 68 | 89 | 5.7 | 887.3 | 893.0 |
| short prompt, long output | 175 | 1966 | 12.4 | 19801.0 | 19813.3 |
| long prompt, short output | 6048 | 83 | 404.2 | 825.9 | 1230.2 |

The ordering inverts in the last two rows: the 6048-token prompt costs 32× the
TTFT and a sixteenth of the e2e, which is why both are reported.

A 324-token record decodes at 10.08 ms/token alone against 9.92 predicted, and
34.69 with 200 resident against 34.6 predicted: the batch model holds at both ends
of the occupancy range.

| concurrent | peak running | peak KV | queue_wait med/p95 | TTFT med/p95 | e2e med/p95 |
| --- | --- | --- | --- | --- | --- |
| 50 | 50/256 | 78804 (20%) | 0.00 / 0.00 | 0.02 / 0.32 | 3.33 / 12.24 |
| 150 | 150/256 | 256320 (65%) | 0.00 / 0.00 | 0.02 / 0.33 | 5.48 / 19.59 |
| 400 | 253/256 | 395983 (100%) | 0.00 / 8.35 | 0.24 / 8.42 | 11.00 / 25.77 |

Seconds. The first two rows matter most: 50 → 150 concurrent degrades median e2e
3.33s → 5.48s, 65%, while p95 queue_wait never leaves 0.00s. Occupancy slows the
batch long before it refuses anyone; a queue-depth alert would not have moved.

Which limit binds is a property of the traffic, not the config. The same 400
requests, three pools:

| KV pool | peak running | peak KV | binds | median e2e |
| --- | --- | --- | --- | --- |
| 150000 | 103/256 | 149995 (100%) | KV pool | 13.51 |
| 396000 | 253/256 | 395983 (100%) | KV pool | 11.00 |
| 900000 | 256/256 | 448631 (50%) | sequence cap | 10.78 |

At 150k the pool refuses everyone: 7.84s median queue_wait, where neither other
pool queues at all. But the configured pool stops three sequences short of the cap,
so traffic shifting shorter hands the constraint to the cap, longer keeps it with
the pool. The measured peak of 253 runs above the config's predicted 221 because
the resident set skews short; `DESIGN.md` plans on the lower figure. Neither is the
knob to turn: 900k buys three sequences and 2% of the median.

## Metrics

Metric shapes mirror vLLM's own, so a dashboard written here ports with a prefix
change — deliberate, since a simulator emitting the real names silently poisons any
dashboard that scrapes both. Seconds throughout, per Prometheus convention, so
`/generate` converts for itself. Latency buckets come from the runs above;
client defaults stop at 10s, where much of this workload lands. TPOT's come from
the model: step time cannot leave 9.8–41.5ms, so the buckets divide that span, plus
one above the ceiling as overflow detection, empty in these runs. Queue time only
looks badly bucketed: at 400 concurrent 58% are admitted on arrival and the rest
wait, which is why its median reads 0.00s against an 8.4s p95. Gauges are read at
scrape time, not mirrored on state change: a second copy is a second source of
truth and it drifts. The `size` label on TTFT and e2e is mine, not vLLM's.

## What I'd alert on

In priority order. `e2e` and TTFT p95 against an SLO: the earliest honest signal,
for the reason the load table shows — latency degrades before anything queues, so
everything downstream pages late. `num_requests_waiting` rising monotonically:
arrivals exceed service rate, and the backlog will not drain on its own.
`sim:request_total{status="rejected"}` above zero: load that spilled, and the
trigger for failing over to a hosted provider.

Not a high `kv_cache_usage_perc` — at saturation it sits at 100%, which
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
at the cost of spread — repeats of the same 400 requests agree on the median to ~1%,
p95 to ~6%.

The tokenizer is a character ratio: it drifts on code and non-English text, but a
consistent bias moves the scale, not the shape.
