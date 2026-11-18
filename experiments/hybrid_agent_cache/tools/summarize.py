# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize sweep runs: python summarize.py <scratch dir> <nb>..."""

import collections
import json
import os
import statistics
import sys

import regex as re

S = sys.argv[1]


def load(p):
    d = collections.defaultdict(float)
    for line in open(p):
        if line.startswith("#"):
            continue
        m = re.match(r"([a-zA-Z_:]+)(\{[^}]*\})?\s+(\S+)", line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        labels = re.sub(r'(engine|model_name)="[^"]*",?', "", m.group(2) or "").replace(
            "{}", ""
        )
        d[m.group(1) + labels] += v
    return d


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


rows = []
for nb in sys.argv[2:]:  # nb may carry a suffix tag, e.g. 160_pmu16
    t = f"nb{nb}"
    fa, fb, fs = (
        f"{S}/m_{t}_before.txt",
        f"{S}/m_{t}_after.txt",
        f"{S}/stats_{t}.json",
    )
    if not all(os.path.exists(f) for f in (fa, fb, fs)):
        rows.append((nb, None))
        continue
    a, b = load(fa), load(fb)
    g = lambda k: b.get(k, 0) - a.get(k, 0)
    st = json.load(open(fs))
    ttft = [r["ttft_ms"] for r in st]
    blk = int(
        re.search(
            r"GPU KV cache size: ([\d,]+) tokens", open(f"{S}/serve_{t}.log").read()
        )
        .group(1)
        .replace(",", "")
    )
    rows.append(
        (
            nb,
            dict(
                cap_tokens=blk,
                hit=g("vllm:prefix_cache_hits_total")
                / g("vllm:prefix_cache_queries_total")
                * 100,
                recompute=g("vllm:prompt_tokens_total")
                - g("vllm:prompt_tokens_cached_total"),
                preempt=g("vllm:num_preemptions_total"),
                reuse=g("vllm:kv_block_lifetime_seconds_count"),
                ttft_mean=statistics.mean(ttft),
                ttft_p50=pct(ttft, 0.5),
                ttft_p90=pct(ttft, 0.9),
                tpot=statistics.mean(r["tpot_ms"] for r in st),
                n=len(st),
                rps=len(st)
                / (
                    (
                        max(r["start_time_ms"] + r["latency_ms"] for r in st)
                        - min(r["start_time_ms"] for r in st)
                    )
                    / 1000
                ),
            ),
        )
    )
hdr = f"{'blocks':>6} {'cap(tok)':>9} {'hit%':>6} {'recomp':>8} {'preempt':>7} {'blkReuse':>8} {'TTFTavg':>8} {'p50':>6} {'p90':>6} {'TPOT':>5} {'req/s':>6} {'n':>4}"
print(hdr)
for nb, r in rows:
    if r is None:
        print(f"{nb:>6}  (missing)")
        continue
    print(
        f"{nb:>6} {r['cap_tokens']:>9} {r['hit']:>6.2f} {r['recompute']:>8.0f} {r['preempt']:>7.0f} {r['reuse']:>8.0f} {r['ttft_mean']:>8.1f} {r['ttft_p50']:>6.1f} {r['ttft_p90']:>6.1f} {r['tpot']:>5.2f} {r['rps']:>6.2f} {r['n']:>4}"
    )
