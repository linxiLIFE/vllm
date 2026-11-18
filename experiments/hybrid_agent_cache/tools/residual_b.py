# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Classify cause-B loss (attention cached, GDN checkpoint unusable) per request.

For request i, G = best GDN checkpoint that could exist with infinite cache:
the max over earlier requests j of c_j = floor((P_j - 1) / 576) * 576 such that
request i shares at least c_j prefix tokens with request j (prompt-tail
checkpoints only; junction checkpoints are ignored). Then, with attn = the
attention-only hit and hit = the reconciled hit from the lookup log:

  ckpt_evicted     attn >= G > hit : attention reaches G, checkpoint at G gone
  attn_truncated   hit <= attn < G : attention tail evicted below G, so the
                                     checkpoint (if still cached) is unreachable
  structural       attn > G        : attention beyond any possible checkpoint

Usage: residual_b.py <traces.jsonl> <res_cc_tag.json> <lookups> [<more res> <more lookups> ...]
"""

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cc_replay import load_sessions  # noqa: E402

BLOCK = 576


def shared_tokens(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n * 64


def best_checkpoint(sessions):
    out = {}
    for idx, (_, reqs) in enumerate(sessions):
        for i, r in enumerate(reqs):
            g = 0
            for j in range(i):
                c = (reqs[j]["in"] - 1) // BLOCK * BLOCK
                if c > g and shared_tokens(r["hash_ids"], reqs[j]["hash_ids"]) >= c:
                    g = c
            out[(idx, i)] = min(g, (r["in"] - 1) // BLOCK * BLOCK)
    return out


def match(res, lookups):
    """Pair replay results with admitted lookups by prompt length, then time."""
    by_len = collections.defaultdict(list)
    for x in lookups:
        by_len[x["prompt"]].append(x)
    pairs, bad = {}, 0
    groups = collections.defaultdict(list)
    for r in res:
        groups[r["prompt_tokens"]].append(r)
    for plen, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["t_send"])
        ls = sorted(by_len[plen], key=lambda x: x["t"])
        # Prefer pairing on equal hit counts, falling back to time order.
        remaining = list(ls)
        for r in rs:
            pick = next(
                (x for x in remaining if x["hit"] == (r["cached"] or 0)), remaining[0]
            )
            if pick["hit"] != (r["cached"] or 0):
                bad += 1
            remaining.remove(pick)
            pairs[(r["session"], r["i"])] = pick
    return pairs, bad


def main():
    traces, args = sys.argv[1], sys.argv[2:]
    for res_path, lk_path in zip(args[::2], args[1::2]):
        d = json.load(open(res_path))
        a = d["meta"]["args"]
        sessions = load_sessions(traces, a["sessions"], a["max_in"], a["min_req"])
        sessions = [(sid, reqs[: a["max_req"]]) for sid, reqs in sessions]
        G = best_checkpoint(sessions)
        res = [r for r in d["results"] if "error" not in r]
        last = {}
        for line in open(lk_path):
            x = json.loads(line)
            if x["preempted"] == 0:
                last[x["rid"]] = x
        pairs, bad = match(res, list(last.values()))
        total = sum(r["prompt_tokens"] for r in res)
        cat = collections.Counter()
        cnt = collections.Counter()
        examples = collections.defaultdict(list)
        for r in res:
            x = pairs[(r["session"], r["i"])]
            g, attn, hit = G[(r["session"], r["i"])], x["attn"], x["hit"]
            if attn - hit <= 0:
                continue
            if attn > g:
                cat["structural"] += attn - max(g, hit)
                cnt["structural"] += 1
            if hit < g <= attn:
                cat["ckpt_evicted"] += g - hit
                cnt["ckpt_evicted"] += 1
                examples["ckpt_evicted"].append(
                    (r["session"], r["i"], r["prompt_tokens"], g, attn, hit)
                )
            elif attn < g:
                cat["attn_truncated"] += attn - hit
                cnt["attn_truncated"] += 1
                examples["attn_truncated"].append(
                    (r["session"], r["i"], r["prompt_tokens"], g, attn, hit)
                )
        b_total = sum(
            pairs[(r["session"], r["i"])]["attn"] - pairs[(r["session"], r["i"])]["hit"]
            for r in res
        )
        print(f"=== {os.path.basename(res_path)}  (pairing mismatches on hit: {bad})")
        print(f"B loss total {b_total / total * 100:6.2f}% of prompt tokens")
        for k in ("ckpt_evicted", "attn_truncated", "structural"):
            print(f"  {k:15s} {cat[k] / total * 100:6.2f}%  in {cnt[k]:3d} requests")
        for k in ("ckpt_evicted", "attn_truncated"):
            for e in examples[k][:4]:
                print(
                    f"    e.g. {k}: s{e[0]:02d} i={e[1]:2d} prompt={e[2]} best_ckpt={e[3]} attn={e[4]} hit={e[5]}"
                )


if __name__ == "__main__":
    main()
