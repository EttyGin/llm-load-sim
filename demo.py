"""A guided tour of the simulator's behaviour, driving the engine directly.

Usage: uv run python scripts/demo.py
"""

import asyncio
import random
import statistics

from app.config import load_config
from app.dataset import load_dataset
from app.engine import Engine

SEED = 0
BACKGROUND = 200
LOADS = (50, 150, 400)
TIGHT_KV = 150000
ROOMY_KV = 900000


def header(number, title, look_for):
    print(f"\n{'=' * 92}\nScenario {number} — {title}\nlook for: {look_for}\n{'=' * 92}")


def med_p95(values):
    values = sorted(values)
    return statistics.median(values), values[min(len(values) - 1, int(0.95 * len(values)))]


async def run_load(config, records, n):
    engine = Engine(config)
    picks = random.Random(SEED).choices(records, k=n)
    results = await asyncio.gather(*(engine.generate(r) for r in picks))
    return engine, results


def load_row(label, config, engine, results):
    q = med_p95([r.timings.queue_wait for r in results])
    t = med_p95([r.timings.ttft for r in results])
    e = med_p95([r.timings.e2e for r in results])
    kv_pct = 100 * engine.peak_kv_used / config.kv_cache_tokens
    print(
        f"  {label:>7}  {engine.peak_running:>3}/{config.max_num_seqs:<4}"
        f"  {engine.peak_kv_used:>6}/{config.kv_cache_tokens:<6} {kv_pct:>3.0f}%"
        f"  {q[0]:>6.2f} {q[1]:>6.2f}  {t[0]:>6.2f} {t[1]:>6.2f}  {e[0]:>6.2f} {e[1]:>6.2f}"
    )


def load_header():
    print(f"\n  {'load':>7}  {'running':<8}  {'kv_used':<17}"
          f"  {'queue_wait':^13}  {'ttft':^13}  {'e2e':^13}")
    print(f"  {'':>7}  {'':<8}  {'':<17}  {'med':>6} {'p95':>6}"
          f"  {'med':>6} {'p95':>6}  {'med':>6} {'p95':>6}   (seconds)")


async def scenario_1(config, records):
    header(1, "prefill and decode, one request at a time",
           "ttft tracks prompt tokens, decode tracks completion tokens. "
           "Nothing else is in flight, so ttft is pure prefill.")
    picks = [
        ("short prompt, short output",
         min((r for r in records if r.completion_tokens >= 50),
             key=lambda r: r.prompt_tokens + r.completion_tokens)),
        ("short prompt, long output",
         max((r for r in records if r.prompt_tokens < 300),
             key=lambda r: r.completion_tokens)),
        ("long prompt, short output",
         max((r for r in records if r.completion_tokens < 150),
             key=lambda r: r.prompt_tokens)),
    ]
    print(f"\n  {'':<28} {'prompt':>7} {'completion':>11}"
          f" {'ttft_ms':>9} {'decode_ms':>10} {'e2e_ms':>10}")
    for label, record in picks:
        engine = Engine(config)
        t = (await engine.generate(record)).timings
        print(f"  {label:<28} {record.prompt_tokens:>7} {record.completion_tokens:>11}"
              f" {t.ttft * 1000:>9.1f} {t.decode_time * 1000:>10.1f} {t.e2e * 1000:>10.1f}")


async def scenario_2(config, records):
    header(2, "one request, alone and then under load",
           f"the same record decodes slower with {BACKGROUND} others resident. "
           "This is why step time is re-read from occupancy on every chunk.")
    target = min(records,
                 key=lambda r: abs(r.prompt_tokens - 1000) + abs(r.completion_tokens - 300))

    engine = Engine(config)
    alone = (await engine.generate(target)).timings.decode_time

    engine = Engine(config)
    # Longest completions in the dataset, so they stay resident for the measurement.
    filler = sorted(records, key=lambda r: -r.completion_tokens)[:BACKGROUND]
    tasks = [asyncio.create_task(engine.generate(r)) for r in filler]
    while engine.running < BACKGROUND:
        await asyncio.sleep(0)
    loaded = (await engine.generate(target)).timings.decode_time
    still_running = engine.running
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\n  record {target.id}: {target.prompt_tokens} prompt tokens,"
          f" {target.completion_tokens} completion tokens")
    for label, decode in (("alone", alone), (f"with {BACKGROUND} in flight", loaded)):
        print(f"  {label:<28} decode {decode * 1000:>9.1f} ms"
              f"  ({decode / (target.completion_tokens - 1) * 1000:.2f} ms/token)")
    print(f"  {'ratio':<28}        {loaded / alone:>9.2f} x"
          f"  ({still_running} still resident at the end)")


async def scenario_3(config, records):
    header(3, "where saturation starts",
           "e2e degrading well before queue_wait leaves zero: occupancy slows the "
           "batch long before the pool refuses anyone.")
    load_header()
    for n in LOADS:
        engine, results = await run_load(config, records, n)
        load_row(n, config, engine, results)
    return engine, results


async def scenario_4(config, records, baseline):
    header(4, "which limit binds",
           "the same 400 requests against three KV pools. The binding limit is the "
           "one sitting at its ceiling: KV at 150k, the sequence cap at 900k, and "
           "the configured pool a few sequences short of where the two swap over.")
    rows = []
    for label, kv in (("150k", TIGHT_KV), ("396k", None), ("900k", ROOMY_KV)):
        if kv is None:
            rows.append((label, config, *baseline))
            continue
        cfg = config.model_copy(update={"kv_cache_tokens": kv})
        rows.append((label, cfg, *await run_load(cfg, records, LOADS[-1])))

    print(f"\n  all three rows are the same {LOADS[-1]} requests;"
          " the label is the pool size")
    load_header()
    for row in rows:
        load_row(*row)

    print()
    for label, cfg, engine, _ in rows:
        bound = "sequence cap" if engine.peak_running == cfg.max_num_seqs else "KV pool"
        print(f"  {label} pool: peak running {engine.peak_running}/{cfg.max_num_seqs},"
              f" peak kv {engine.peak_kv_used}/{cfg.kv_cache_tokens}"
              f" — {bound} binds first")

    reached = baseline[0].peak_running
    print(f"\n  the configured pool stops {config.max_num_seqs - reached} sequences short"
          f" of the {config.max_num_seqs}-slot cap. At this workload's length the two"
          " constraints all but coincide, so a shift in traffic mix — shorter requests"
          "\n  or longer — moves which one governs, and tuning either alone buys nothing.")


async def main():
    config = load_config()
    dataset = load_dataset(config.dataset_path, config.chars_per_token)
    records = [dataset.get(i) for i in dataset.ids()]
    print(f"{config.model_name} on {config.gpu}: {config.max_num_seqs} sequence slots,"
          f" {config.kv_cache_tokens} KV tokens, {len(records)} records")

    await scenario_1(config, records)
    await scenario_2(config, records)
    baseline = await scenario_3(config, records)
    await scenario_4(config, records, baseline)


asyncio.run(main())
