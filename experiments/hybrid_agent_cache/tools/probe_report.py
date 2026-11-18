# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize probe_<tag>.jsonl files: python probe_report.py <dir> <tag>..."""

import json
import sys

S = sys.argv[1]
cols = [
    "inuse_attn",
    "inuse_mamba_live",
    "inuse_mamba_superseded",
    "inuse_uncached",
    "free_attn",
    "free_mamba_live",
    "free_mamba_superseded",
    "free_uncached",
]
print("Average blocks over the run (snapshots every 0.25 s while allocating)")
print(
    f"{'run':<22}"
    + "".join(
        f"{c.replace('inuse_', 'use:').replace('free_', 'free:'):>16}" for c in cols
    )
)
cums = {}
for tag in sys.argv[2:]:
    recs = [json.loads(l) for l in open(f"{S}/probe_{tag}.jsonl")]
    # Skip warmup/idle snapshots: keep those after the first cache insertion.
    recs = [
        r
        for r in recs
        if sum(v for k, v in r["cum"].items() if k.startswith("insert")) > 0
    ]
    avg = {c: sum(r["snap"].get(c, 0) for r in recs) / len(recs) for c in cols}
    print(f"{tag:<22}" + "".join(f"{avg[c]:>16.1f}" for c in cols))
    cums[tag] = recs[-1]["cum"]
keys = sorted({k for c in cums.values() for k in c if not k.endswith("_sum")})
print("\nCumulative counters")
print(f"{'counter':<34}" + "".join(f"{t:>22}" for t in cums))
for k in keys:
    print(f"{k:<34}" + "".join(f"{cums[t].get(k, 0):>22}" for t in cums))
print("\nMean age at eviction (s)")
for t, c in cums.items():
    parts = []
    for st in ("attn", "live", "superseded"):
        n = sum(c.get(f"evict_{st}_{h}", 0) for h in ("hit", "nohit"))
        if n:
            parts.append(
                f"{st} {c.get(f'evict_{st}_age_ms_sum', 0) / n / 1000:.1f}s (n={n})"
            )
    print(f"  {t:<22} " + ", ".join(parts))
