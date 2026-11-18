# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Split the gap between ideal and actual prefix-cache hits into A vs B.

A = attention blocks themselves gone (ideal-576 - attention-only hit).
B = attention blocks present but no usable GDN checkpoint (attention-only hit - hit).
B contains a structural part of at most one block per request (the checkpoint
sits at floor((P-1)/576)*576 of the previous prompt while attention may hold
one more full block); it is reported separately as B_struct (gap <= 1 block).

Usage: lookup_decompose.py <traces.jsonl> <res_cc_tag.json> <probe_cc_tag.jsonl.lookups>
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cc_analyze import BLOCK, ideal_reuse  # noqa: E402
from cc_replay import load_sessions  # noqa: E402


def main():
    traces, res_path, lookups_path = sys.argv[1:4]
    d = json.load(open(res_path))
    a = d["meta"]["args"]
    sessions = load_sessions(traces, a["sessions"], a["max_in"], a["min_req"])
    sessions = [(sid, reqs[: a["max_req"]]) for sid, reqs in sessions]
    ideal = ideal_reuse(sessions)
    rs = [r for r in d["results"] if "error" not in r]
    prompt = sum(r["prompt_tokens"] for r in rs)
    served = sum(r["cached"] or 0 for r in rs)
    ideal576 = sum(ideal[(r["session"], r["i"])] // BLOCK * BLOCK for r in rs)

    last = {}
    for line in open(lookups_path):
        rec = json.loads(line)
        if rec["preempted"] == 0:
            last[rec["rid"]] = rec
    lk = list(last.values())
    hit = sum(x["hit"] for x in lk)
    attn = sum(x["attn"] for x in lk)
    b_struct = sum(x["attn"] - x["hit"] for x in lk if x["attn"] - x["hit"] <= BLOCK)
    b_evict = sum(x["attn"] - x["hit"] for x in lk if x["attn"] - x["hit"] > BLOCK)
    n_b_evict = sum(1 for x in lk if x["attn"] - x["hit"] > BLOCK)
    lk_prompt = sum(x["prompt"] for x in lk)

    pct = lambda v: f"{v / prompt * 100:6.2f}%"  # noqa: E731
    print(
        f"=== {os.path.basename(res_path)}: {len(rs)} requests, {len(lk)} admitted lookups"
    )
    print(f"prompt tokens (replay) {prompt:,}  (lookups {lk_prompt:,})")
    print(f"ideal-576 hit           {pct(ideal576)}")
    print(f"attention-only hit      {pct(attn)}")
    print(f"actual hit (lookup)     {pct(hit)}   (server-reported {pct(served)})")
    print("gap decomposition, share of prompt tokens:")
    print(f"  A  attention blocks missing          {pct(ideal576 - attn)}")
    print(f"  B  attention present, GDN missing    {pct(attn - hit)}")
    print(f"     of which structural (<=1 block)   {pct(b_struct)}")
    print(
        f"     of which >1 block (evicted ckpt)  {pct(b_evict)}  in {n_b_evict} requests"
    )


if __name__ == "__main__":
    main()
