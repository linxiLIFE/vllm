# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEMPORARY instrumentation: classify prefix-cache contents of hybrid models.

Enabled only when VLLM_CACHE_PROBE=<output .jsonl path>. A Mamba checkpoint
is "superseded" when the same group also caches a checkpoint for a strictly
longer prefix of it: a linear conversation can then never resume from it.
"""

import json
import os
import time
from collections import Counter

from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id


class CacheProbe:
    @classmethod
    def maybe_create(cls, pool) -> "CacheProbe | None":
        path = os.environ.get("VLLM_CACHE_PROBE")
        return cls(pool, path) if path else None

    def __init__(self, pool, path: str):
        self.pool = pool
        self.out = open(path, "w", buffering=1)
        self.mamba_groups: set[int] = set()
        self.parent: dict[bytes, bytes | None] = {}
        self.req_done: dict[str, int] = {}
        self.desc: Counter = Counter()
        self.bumped: dict[bytes, list[bytes]] = {}
        self.birth: dict[bytes, float] = {}
        self.hits: Counter = Counter()
        self.evicting = False
        self.c: Counter = Counter()
        self.last_snap = 0.0
        self.t0 = time.monotonic()
        self.block_size = pool.hash_block_size
        # Counterfactual: move superseded Mamba checkpoints to the eviction front.
        self.evict_superseded = (
            os.environ.get("VLLM_CACHE_PROBE_EVICT_SUPERSEDED") == "1"
        )

    def kind(self, key: bytes) -> str:
        return "mamba" if get_group_id(key) in self.mamba_groups else "attn"

    def is_superseded(self, block) -> bool:
        if block.block_hash is None or self.kind(block.block_hash) != "mamba":
            return False
        extra = self.pool.cached_block_hashes_by_block.get(block.block_id, ())
        return all(self.desc.get(k) for k in (block.block_hash, *extra))

    def _demote(self, akey: bytes) -> None:
        block_id = self.pool.cached_block_hash_to_block._cache.get(akey)
        if block_id is None:
            return
        block = self.pool.blocks[block_id]
        if block.ref_cnt == 0 and self.is_superseded(block):
            self.pool.free_block_queue.remove(block)
            self.pool.free_block_queue.prepend_n([block])
            self.c["demote_superseded"] += 1

    def on_request(self, request) -> None:
        hashes = request.block_hashes
        start = self.req_done.get(request.request_id, 0)
        for i in range(start, len(hashes)):
            self.parent.setdefault(hashes[i], hashes[i - 1] if i else None)
        self.req_done[request.request_id] = len(hashes)

    def on_insert(self, key: bytes, partial: bool) -> None:
        kind = self.kind(key)
        self.c[f"insert_{kind}{'_partial' if partial else ''}"] += 1
        self.birth[key] = time.monotonic()
        if kind != "mamba":
            return
        suffix = key[-4:]
        cache = self.pool.cached_block_hash_to_block._cache
        bumped = []
        h = self.parent.get(get_block_hash(key))
        while h is not None:
            akey = h + suffix
            if akey in cache:
                self.desc[akey] += 1
                bumped.append(akey)
            h = self.parent.get(h)
        if bumped:
            self.bumped[key] = bumped
            if self.evict_superseded:
                for akey in bumped:
                    self._demote(akey)

    def on_remove(self, key: bytes, block) -> None:
        for akey in self.bumped.pop(key, ()):
            self.desc[akey] -= 1
            if self.desc[akey] <= 0:
                del self.desc[akey]
        born = self.birth.pop(key, None)
        superseded = self.desc.pop(key, 0) > 0
        if not self.evicting:
            self.c["remove_noevict"] += 1
            return
        kind = self.kind(key)
        state = ("superseded" if superseded else "live") if kind == "mamba" else "attn"
        hit = "hit" if self.hits[block.block_id] else "nohit"
        self.c[f"evict_{state}_{hit}"] += 1
        if born is not None:
            self.c[f"evict_{state}_age_ms_sum"] += int((time.monotonic() - born) * 1e3)

    def on_evict_done(self, block) -> None:
        self.hits.pop(block.block_id, None)

    def on_hit(self, block) -> None:
        if block.block_hash is not None:
            self.hits[block.block_id] += 1
            self.c[f"hit_{self.kind(block.block_hash)}"] += 1

    def maybe_snapshot(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_snap < 0.25:
            return
        self.last_snap = now
        snap: Counter = Counter()
        extra = self.pool.cached_block_hashes_by_block
        for b in self.pool.blocks:
            if b.is_null:
                continue
            if b.block_hash is None:
                snap["inuse_uncached" if b.ref_cnt else "free_uncached"] += 1
                continue
            kind = self.kind(b.block_hash)
            if kind == "mamba":
                keys = [b.block_hash, *extra.get(b.block_id, ())]
                kind += "_superseded" if all(self.desc.get(k) for k in keys) else "_live"
            snap[("inuse_" if b.ref_cnt else "free_") + kind] += 1
        rec = {"t": round(now - self.t0, 3), "snap": snap, "cum": self.c}
        self.out.write(json.dumps(rec) + "\n")

    def on_lookup(self, request, max_len: int, hit: int, attn_extra: int) -> None:
        """Per-lookup decomposition: attention-only hit = hit + attn_extra."""
        if not hasattr(self, "lookup_out"):
            self.lookup_out = open(self.out.name + ".lookups", "w", buffering=1)
        rec = {
            "t": round(time.monotonic() - self.t0, 3),
            "rid": request.request_id,
            "prompt": request.num_prompt_tokens,
            "max": max_len,
            "hit": hit,
            "attn": hit + attn_extra,
            "preempted": request.num_preemptions,
        }
        self.lookup_out.write(json.dumps(rec) + "\n")
