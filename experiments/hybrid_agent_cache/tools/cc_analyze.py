# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Analyze cc_replay results: python cc_analyze.py <traces.jsonl> <res_*.json>..."""

import json
import statistics
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from cc_replay import load_sessions  # noqa: E402

BLOCK = 576


def ideal_reuse(sessions):
    """Per (session, i): tokens reusable with a perfect 64-token prefix cache."""
    out = {}
    for idx, (_, reqs) in enumerate(sessions):
        seen = set()
        for i, r in enumerate(reqs):
            n = 0
            for h in r["hash_ids"]:
                if h not in seen:
                    break
                n += 1
            out[(idx, i)] = min(n * 64, r["in"] - 1)
            seen.update(r["hash_ids"])
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def main():
    traces, files = sys.argv[1], sys.argv[2:]
    for f in files:
        d = json.load(open(f))
        a = d["meta"]["args"]
        sessions = load_sessions(traces, a["sessions"], a["max_in"], a["min_req"])
        sessions = [(sid, reqs[: a["max_req"]]) for sid, reqs in sessions]
        ideal = ideal_reuse(sessions)
        rs = [r for r in d["results"] if "error" not in r]
        errs = len(d["results"]) - len(rs)
        tot = sum(r["prompt_tokens"] for r in rs)
        cached = sum(r["cached"] or 0 for r in rs)
        i64 = sum(ideal[(r["session"], r["i"])] for r in rs)
        i576 = sum(ideal[(r["session"], r["i"])] // BLOCK * BLOCK for r in rs)
        ttft = [r["ttft"] for r in rs if r["ttft"] is not None]
        print(
            f"=== {f.split('/')[-1]}  requests={len(rs)} errors={errs} wall={d['meta']['wall']:.0f}s"
        )
        print(
            f"prompt tokens {tot:,}  cached {cached:,}  -> hit {cached / tot * 100:.2f}%"
        )
        print(
            f"ideal (64-tok perfect cache) {i64 / tot * 100:.2f}%   ideal floored to {BLOCK}: {i576 / tot * 100:.2f}%"
        )
        print(
            f"recomputed tokens {tot - cached:,}  (ideal-576 would recompute {tot - i576:,})"
        )
        print(
            f"TTFT mean {statistics.mean(ttft):.3f}s p50 {pct(ttft, 0.5):.3f} p90 {pct(ttft, 0.9):.3f} p99 {pct(ttft, 0.99):.3f}"
        )
        # Per-request "lost" = ideal-576 reuse that the server failed to deliver.
        lost = [
            (ideal[(r["session"], r["i"])] // BLOCK * BLOCK) - (r["cached"] or 0)
            for r in rs
        ]
        missed = [x for x in lost if x > BLOCK]
        print(
            f"requests losing > 1 block vs ideal-576: {len(missed)} / {len(rs)}; tokens lost {sum(max(0, x) for x in lost):,}"
        )
        # Hit rate vs scaled idle gap before the request (think time actually slept).
        bins = [(0, 0.5), (0.5, 3), (3, 12), (12, 31)]
        print(
            "by think gap (replayed seconds = trace think_time x time_scale, capped):"
        )
        for lo, hi in bins:
            sel = [
                r
                for r in rs
                if r["i"] > 0
                and r["think"] is not None
                and lo <= min(r["think"] * a["time_scale"], a["max_think"]) < hi
            ]
            if not sel:
                continue
            p = sum(r["prompt_tokens"] for r in sel)
            c = sum(r["cached"] or 0 for r in sel)
            idl = sum(ideal[(r["session"], r["i"])] // BLOCK * BLOCK for r in sel)
            real_lo, real_hi = lo / a["time_scale"], hi / a["time_scale"]
            print(
                f"  gap {lo:>4}-{hi:<4}s (trace {real_lo:>5.0f}-{real_hi:<5.0f}s) n={len(sel):3d}  "
                f"hit {c / p * 100:6.2f}%  ideal-576 {idl / p * 100:6.2f}%"
            )
        print("per session (requests, hit%, ideal-576%):")
        for idx, (sid, reqs) in enumerate(sessions):
            sel = [r for r in rs if r["session"] == idx]
            p = sum(r["prompt_tokens"] for r in sel)
            c = sum(r["cached"] or 0 for r in sel)
            idl = sum(ideal[(idx, r["i"])] // BLOCK * BLOCK for r in sel)
            print(
                f"  s{idx:02d} {sid[:10]} n={len(sel):3d} max_in={max(r['in'] for r in reqs):6d} "
                f"hit {c / p * 100:6.2f}%  ideal-576 {idl / p * 100:6.2f}%"
            )


if __name__ == "__main__":
    main()
