# Self-hosted inference: system design

Memo to the AI Platforms team. Section 1 of 3.

Every figure below comes from `config.json`, `data/dataset.json`, or a
`scripts/demo.py` run in this repo; anything that does not is labelled an
assumption.

## 1. Proposed architecture

Our traffic is the 750-record dataset: mean prompt 1464 tokens, mean completion
328 — assumed representative, and it is the section's largest assumption, since
that dataset is synthetic and every figure below scales with its means. At the
target of 10,000 requests/min — 167/s — that lands on the card as two separate
demands, and they are not the same size.

| stage | demand at 167 req/s | rate per GPU | GPUs |
| --- | --- | --- | --- |
| prefill | 244,055 tok/s | 15,000 tok/s | 16.27 |
| decode | 54,616 tok/s | 5,940 tok/s | 9.19 |
| total | | | 25.46 |

**Prefill dominates decode 1.77 to 1**, 64% of the fleet against 36%. One average
request costs 0.153 GPU-seconds, of which 0.098 is prefill and 0.055 is decode.
That ratio drives the rest of this section: our bottleneck is prompt processing,
not generation, because prompts run 4.5× longer than completions. If the real mix
skews shorter, prefill's share falls and the fleet with it: prompts 30% shorter put
the same arithmetic at 20.3 GPUs, four nodes rather than five.

The decode rate is derived rather than measured. The pool holds
`396000 / 1792 = 221` average sequences, below the 256-sequence cap, so KV binds
first: `N = 221`, step time `9.8 + 0.124 × 221 = 37.2ms`, giving `221 / 0.0372s =`
**5940 output tokens/sec**, or 1088 requests/min per GPU if a card did nothing but
decode. Scenario 3 actually reaches a peak of 253 concurrent sequences, above the
estimate, because the resident set skews short — 1565 tokens per resident sequence
at that instant against the dataset's 1792 mean. That is 3% more throughput and
does not change the node count, so we plan on the config-derived 5940: the choice
is about which number we can defend, not which is larger.

We assume a 70% target utilisation. That gives `25.46 / 0.7 = 36.4` GPUs, so
**five 8×H100 nodes, 40 GPUs**, at a realised 64%. The 30% is not slack for its
own sake: our load runs put p95 e2e at 2.3× the median at saturation, so it is
p99 headroom first, then rolling deploys, then node maintenance. Pulling a node
leaves 32 GPUs against a 25.46 requirement — 80% utilisation, degraded but
serving, without a capacity re-plan.

Qwen3-8B is a capacity decision as much as a quality one. At bf16 its weights are
15.3 GiB (README's provenance derivation), so one model plus the 396,000-token KV
pool fits inside a single 80 GB H100 with no tensor parallelism. The scaling unit
is therefore one replica per
GPU: 40 independent replicas, no collectives on the request path, no NVLink
dependency between them, and a failed card costs 1/40 of capacity — 2.5%, drained
by the router without touching its seven neighbours. A 70B model does not fit one
card at bf16 — 140 GB of weights — so tensor parallelism across four makes half a
node the scaling unit, four cards the blast radius of a single failure, and the
interconnect a dependency of every request.

The request path is application → API gateway → router → vLLM replicas on
Kubernetes. The gateway owns identity: authentication, per-tenant rate limiting,
and tagging each request with a traffic class. The router owns replica choice
only. Kubernetes owns placement, restarts, and node lifecycle. That split matters
under pressure: shedding load and choosing where to send it are different
decisions.

Round-robin is the wrong default here, and our own numbers say so. Scenario 1, one
request at a time on an otherwise idle engine: a 68-token prompt with an 89-token
completion finishes in 892ms; a 175-token prompt with a 1966-token completion
takes 22.9s. Same engine, no contention, **26× apart** — and TTFT spreads
similarly, 5.9ms against 405ms for a 6048-token prompt. Round-robin equalises
request counts, which is not the quantity that matters: a replica that draws a few
long completions saturates while its neighbour idles, and nothing in the request
says which it will be. We should route on queue depth or least outstanding
requests, both of which every replica already exports as
`vllm:num_requests_running` and `vllm:num_requests_waiting`.

On top of that, prefix-cache-aware routing. Document QA re-sends the same
documents across many questions, and since 64% of our GPU time is prefill, sending
a request to a replica that already holds its prefix saves precisely the resource
we are short of. Assumption, flagged: this simulator models no prefix cache, so
the saving is not measured here — but the prefill share that makes it the right
lever is.

For observability, each replica exposes `/metrics`, Prometheus scrapes per pod,
Grafana sits on top; the metric shapes mirror vLLM's, so dashboards port with a
prefix change. OpenTelemetry traces should span
gateway → router → replica, because latency has to be attributable to a stage:
time queued in the router and time queued inside a replica are indistinguishable
from the client, and they have different fixes. Finally, per-team token accounting
from `vllm:prompt_tokens_total` and `vllm:generation_tokens_total`, labelled by
tenant. We are migrating off an external provider, and that case is only settled
after the fact — tokens served per GPU-hour against the invoice we stopped paying.

## 2. Scaling challenges and proposed solutions

Four, ordered by how much of the fleet each one is worth. Each comes out of a
number this repo produced rather than from a list of things that can go wrong.

### Prefill blocks decode

Prefill is 64% of the fleet by section 1's split, and on real vLLM it costs more
than its share. Without chunked prefill the scheduler runs a prefill to completion
before the next decode step, so every sequence resident on that replica stops for
its duration. Scenario 1 measures a 6048-token prompt at 405ms of prefill on an
idle engine; against the 37.2ms step time at N=221 that is roughly eleven decode
steps, and at scenario 3's peak of 253 resident sequences it is eleven tokens that
252 other users do not receive. Our simulator does not model this — prefill and
decode run concurrently here, so a long prompt surfaces as queueing rather than in
everyone's step time, and no figure in this memo prices it.

Three fixes, doing different jobs. Prefix caching reduces how much prefill there
is: document QA re-sends the same documents, and a cached prefix is prefill we
never run. Chunked prefill reduces the damage prefill does, splitting the prompt
across steps and interleaving it with decode — a little TTFT traded for a bounded
stall. They compose, and we recommend both, noting that neither is simulated here,
so we have no measured benefit to quote for either. Disaggregation — separate
prefill and decode fleets with KV shipped between them — attacks the same problem
structurally and is worth revisiting at several hundred GPUs; at 40 it buys a
network hop and a second control plane to fix a stall that chunked prefill already
bounds.

### Autoscaling cannot be reactive

A new replica loads 15.3 GiB of weights and captures CUDA graphs before it serves
a token. Assumption, flagged: one to two minutes to ready, since nothing in this
repo measures pod startup. Scenario 3 says what happens meanwhile — at 400
concurrent, p95 queue_wait is 8.35s. The spike arrives, queues and drains well
inside the window in which the new pod is still loading weights, so a reactive
autoscaler is late by construction. CPU utilisation is the wrong signal for a GPU
service, and latency, though the right SLO, only moves once we are already
missing it.

So: scale on queue depth, which leads latency rather than trailing it; hold a warm
pool of loaded replicas for the first minute of a spike; and keep the headroom
generous. That is the second reason for section 1's 70% assumption — the spare 30%
is what serves the spike while pods come up. At 90% utilisation the autoscaler's
response time becomes the outage.

### The binding limit moves with the traffic mix

Scenario 4 runs the same 400 requests against three KV pools. At 150k the pool
binds at 103 of 256 sequences; at 900k the sequence cap binds at 256 with the pool
half empty, 448631 of 900000; at our configured 396k we reach 253 of 256 with the
pool at 100% — three sequences short of where the two constraints swap. At this
workload's 1792-token mean they all but coincide, which is why neither knob does
anything on its own: tripling the pool to 900k moved median e2e from 11.01s to
10.78s, inside run-to-run noise, because the cap took over the instant the pool
stopped binding. Raising `max_num_seqs` at 396k would be the same non-event in the
other direction.

The planning consequence is that our capacity is a function of traffic shape, not
request count, and we sit exactly where the answer changes. A mix skewing shorter
puts us on the sequence cap, longer on the KV pool, and the two have different
fixes. We should therefore track the prompt and completion token-mix distribution
as a first-class signal beside request rate: it is the input to every number in
section 1, and it decides which constraint we would be tuning against.

### Degradation starts before queueing does

Scenario 3 from 50 to 150 concurrent: p95 queue_wait stays at 0.00s — nothing ever
waits — while median e2e rises 3.33s to 5.49s, 65% worse against an empty queue.
The mechanism is in `config.json` rather than in the queue: step time is
9.8 + 0.124 × running, so 16.0ms at 50 resident against 28.4ms at 150, a 1.78×
slowdown applied to every sequence and needing no contention for admission at all.
We can breach an SLO at a third of the pool's capacity with nothing queued.

Queue depth is therefore a saturation alarm, not an early warning, and the SLO has
to sit on latency percentiles. Capacity planning should follow: section 1's 70% is
a proxy for a latency target, and the honest version measures where p95 e2e crosses
the SLO and sizes to that. The tension with the challenge above is real and worth
stating — scaling on queue depth is right for spikes and blind to this, latency is
right for this and late for spikes. We need both, for different jobs: queue depth
to add capacity quickly, latency percentiles to decide how much of it we need.
