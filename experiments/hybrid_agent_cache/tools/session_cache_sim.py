# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline session-cache simulator: how much hit rate can retention policy buy?

Model (hybrid model after #61282 + #66881, i.e. no dead weight):
- Cached state is a set of checkpoint entries. An entry at length L of a
  session needs the attention blocks of its prefix [0, L) plus 3 GDN state
  blocks. Attention blocks are shared between entries with a common prefix
  (counted once, as in vLLM's block pool).
- After a request with prompt P finishes, its session caches an entry at its
  prompt-tail checkpoint floor((P - 1) / 576) * 576. If its prompt diverged
  from an existing entry part way (the attention hit ran past its own hit),
  it also caches a junction entry at the divergence point, like vLLM's
  `shared_prefix_boundary` checkpoint.
- A request hits an entry only if it shares the entry's whole prefix (the
  only checkpoint is at the entry's end): all or nothing, longest entry wins.
- A running request holds ceil((P + O) / 576) + 6 blocks from when it is
  scheduled (send time + TTFT - fitted prefill time) until it finishes. The
  timeline comes from a recorded run and is held fixed (open loop).

Policies choose which cached entry to evict when a starting request needs room:
  lru        least recently used
  belady     farthest next use (offline optimal for unit sizes)
  value      lowest (tokens saved at next use) / (blocks freed * time to next use)

Usage: session_cache_sim.py <traces.jsonl> <res_cc_tag.json> [capacity...]
"""

import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cc_replay import load_sessions  # noqa: E402

BLOCK = 576
HASHES_PER_BLOCK = BLOCK // 64
GDN_BLOCKS = 3
# Prefill cost fitted on the 3060 (results/cctrace/prefill_cost_fit.json).
D, E = 6.14e-5, 2.44e-4


def shared_tokens(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n * 64


def block_ids(s, hashes, length):
    """Attention block identities of prefix [0, length): the hash id ending each block."""
    return [(s, hashes[HASHES_PER_BLOCK * (j + 1) - 1]) for j in range(length // BLOCK)]


def build(traces, res_path):
    d = json.load(open(res_path))
    a = d["meta"]["args"]
    sessions = load_sessions(traces, a["sessions"], a["max_in"], a["min_req"])
    sessions = [(sid, reqs[: a["max_req"]]) for sid, reqs in sessions]
    timing = {(r["session"], r["i"]): r for r in d["results"] if "error" not in r}
    reqs = []
    for s, (_, rs) in enumerate(sessions):
        for i, r in enumerate(rs):
            t = timing.get((s, i))
            if t is None:
                continue
            c = t["cached"] or 0
            n = r["in"] - c
            prefill = D * n + E * n * (2 * c + n) / 2e5
            ttft = t["ttft"] if t["ttft"] is not None else 0.0
            reqs.append(
                dict(
                    s=s,
                    P=r["in"],
                    O=max(1, r["out"]),
                    hashes=r["hash_ids"],
                    start=t["t_send"] + max(0.0, ttft - prefill),
                    end=t["t_send"] + t["latency"],
                    L=(r["in"] - 1) // BLOCK * BLOCK,
                )
            )
    reqs.sort(key=lambda r: r["start"])
    return reqs


class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = {}
        self.refs = collections.Counter()
        self.n_cached = 0
        self.n_running = 0
        self.next_id = 0

    def usage(self):
        # Every attention block counted once, whoever holds it; a cached entry
        # adds its 3 state blocks, a running request its 6.
        return (
            len(self.refs)
            + GDN_BLOCKS * self.n_cached
            + 2 * GDN_BLOCKS * self.n_running
        )

    def free(self):
        return self.capacity - self.usage()

    def hold(self, blocks, sign):
        for b in blocks:
            self.refs[b] += sign
            if self.refs[b] == 0:
                del self.refs[b]
        self.n_running += sign

    def _account(self, e, sign):
        for b in e["blocks"]:
            self.refs[b] += sign
            if self.refs[b] == 0:
                del self.refs[b]
        self.n_cached += sign

    def add(self, s, L, hashes, now):
        for e in self.entries.values():
            if (
                e["s"] == s
                and e["L"] == L
                and e["hashes"][: L // 64] == hashes[: L // 64]
            ):
                e["last_use"] = now
                return
        e = dict(
            s=s,
            L=L,
            hashes=hashes,
            blocks=block_ids(s, hashes, L),
            last_use=now,
            pinned=False,
        )
        self.entries[self.next_id] = e
        self.next_id += 1
        self._account(e, +1)

    def remove(self, eid):
        e = self.entries.pop(eid)
        if not e["pinned"]:
            self._account(e, -1)

    def pin(self, eid):
        self._account(self.entries[eid], -1)
        self.entries[eid]["pinned"] = True

    def unpin(self, eid, now):
        e = self.entries[eid]
        e["pinned"] = False
        e["last_use"] = now
        self._account(e, +1)

    def freed_by(self, eid):
        e = self.entries[eid]
        return GDN_BLOCKS + sum(1 for b in e["blocks"] if self.refs[b] == 1)


def simulate(reqs, capacity, policy):
    cache = Cache(capacity)
    events = []
    for idx, r in enumerate(reqs):
        events.append((r["start"], 1, idx))
        events.append((r["end"], 0, idx))
    events.sort()
    hit_tok = total_tok = overflow = 0
    using, attn_match, held = {}, {}, {}

    def next_use(e, now):
        for r in reqs:
            if r["start"] > now and r["s"] == e["s"] and e["L"] <= r["P"] - 1:
                if shared_tokens(r["hashes"], e["hashes"]) >= e["L"]:
                    return r["start"]
        return math.inf

    def evict_for(need, now):
        while cache.free() < need:
            cand = [k for k, e in cache.entries.items() if not e["pinned"]]
            if not cand:
                return False
            if policy == "lru":
                victim = min(cand, key=lambda k: cache.entries[k]["last_use"])
            elif policy == "belady":
                victim = max(cand, key=lambda k: next_use(cache.entries[k], now))
            else:

                def score(k):
                    t = next_use(cache.entries[k], now)
                    if t == math.inf:
                        return -1.0
                    return cache.entries[k]["L"] / (
                        cache.freed_by(k) * max(t - now, 1e-6)
                    )

                victim = min(cand, key=score)
            cache.remove(victim)
        return True

    for now, kind, idx in events:
        r = reqs[idx]
        need = math.ceil((r["P"] + r["O"]) / BLOCK) + 2 * GDN_BLOCKS
        if kind == 1:
            total_tok += r["P"]
            best, longest_attn = None, 0
            for eid, e in cache.entries.items():
                if e["s"] != r["s"] or e["pinned"]:
                    continue
                shared = min(shared_tokens(r["hashes"], e["hashes"]), r["P"] - 1)
                longest_attn = max(longest_attn, min(shared, e["L"]) // BLOCK * BLOCK)
                if e["L"] <= shared and (
                    best is None or e["L"] > cache.entries[best]["L"]
                ):
                    best = eid
            hit = cache.entries[best]["L"] if best is not None else 0
            hit_tok += hit
            if best is not None:
                cache.pin(best)
            using[idx] = best
            attn_match[idx] = longest_attn
            full = r["P"] // BLOCK * BLOCK
            held[idx] = block_ids(r["s"], r["hashes"], full) + [
                ("run", idx, j) for j in range(need - 2 * GDN_BLOCKS - full // BLOCK)
            ]
            cache.hold(held[idx], +1)
            if not evict_for(0, now):
                overflow += 1
        else:
            cache.hold(held.pop(idx), -1)
            old = using.pop(idx)
            if old is not None:
                # Keep the resumed-from entry: a later request may continue an
                # older branch of the session. It shares its attention blocks
                # with this request's entry, so it costs only its state blocks.
                cache.unpin(old, now)
            if r["L"] > 0:
                cache.add(r["s"], r["L"], r["hashes"], now)
            junction = attn_match.pop(idx)
            hit_len = 0 if old is None else min(r["L"], junction)
            if 0 < junction < r["L"] and junction > hit_len:
                cache.add(r["s"], junction, r["hashes"], now)
            evict_for(0, now)
    return hit_tok / total_tok * 100, overflow


def main():
    traces, res_path = sys.argv[1], sys.argv[2]
    caps = [int(x) for x in sys.argv[3:]] or [871]
    reqs = build(traces, res_path)
    inf, _ = simulate(reqs, 10**9, "lru")
    print(f"timeline: {os.path.basename(res_path)}  requests: {len(reqs)}")
    print(f"infinite cache: {inf:.2f}%")
    print(
        f"{'capacity':>9} " + " ".join(f"{p:>16}" for p in ("lru", "belady", "value"))
    )
    for cap in caps:
        row = []
        for p in ("lru", "belady", "value"):
            hit, ov = simulate(reqs, cap, p)
            row.append(f"{hit:6.2f}% (ovf {ov:2d})")
        print(f"{cap:>9} " + " ".join(f"{x:>16}" for x in row))


if __name__ == "__main__":
    main()
