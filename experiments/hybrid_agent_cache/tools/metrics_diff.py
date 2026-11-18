# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scrape /metrics twice (before/after file) and print cache/latency deltas."""

import collections
import sys

import regex as re


def load(p):
    d = collections.defaultdict(float)
    for line in open(p):
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+|NaN|\+Inf)", line)
        if not m:
            continue
        name, labels, v = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(v)
        except ValueError:
            continue
        labels = re.sub(r'(engine|model_name)="[^"]*",?', "", labels).replace("{}", "")
        d[name + labels] += v
    return d


a, b = load(sys.argv[1]), load(sys.argv[2])
delta = {k: b[k] - a.get(k, 0.0) for k in b}


def g(k):
    return delta.get(k, 0.0)


q, h = g("vllm:prefix_cache_queries_total"), g("vllm:prefix_cache_hits_total")
pt, pc = g("vllm:prompt_tokens_total"), g("vllm:prompt_tokens_cached_total")
print(f"prefix_cache_queries  {q:.0f} tokens")
print(
    f"prefix_cache_hits     {h:.0f} tokens  -> hit rate {h / q * 100 if q else 0:.2f}%"
)
print(f"prompt_tokens         {pt:.0f}")
print(
    f"prompt_tokens_cached  {pc:.0f}  -> {pc / pt * 100 if pt else 0:.2f}% of prompt tokens served from cache"
)
print(f"recomputed prompt tok {pt - pc:.0f}")
print(f"preemptions           {g('vllm:num_preemptions_total'):.0f}")
print(f"generation_tokens     {g('vllm:generation_tokens_total'):.0f}")
for name in [
    "time_to_first_token_seconds",
    "request_prefill_time_seconds",
    "request_queue_time_seconds",
    "inter_token_latency_seconds",
]:
    s, c = g(f"vllm:{name}_sum"), g(f"vllm:{name}_count")
    if c:
        print(f"{name:32s} mean {s / c * 1000:8.2f} ms  (n={c:.0f})")
print(
    f"kv_cache_usage_perc (now) {b.get('vllm:kv_cache_usage_perc', float('nan')) * 100:.2f}%"
)
for k in sorted(b):
    if "by_source" in k or "kv_block" in k:
        if delta[k]:
            print(f"  {k} {delta[k]:.0f}")
