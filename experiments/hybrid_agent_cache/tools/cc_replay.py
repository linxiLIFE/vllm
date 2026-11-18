# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Closed-loop replay of Claude Code WekaTrace sessions against a vLLM server.

Each session's main stream (dominant model, type "s") is replayed in order:
send request i, wait for it to finish, sleep think_time * time_scale (capped),
send request i+1. Every 64-token hash id maps to a fixed random token block,
so identical hash ids give identical tokens and the trace's prefix reuse is
preserved. Hash ids are session-local, so sessions share nothing.

Usage: cc_replay.py traces.jsonl out.json --sessions 12 --max-in 128000
"""

import argparse
import asyncio
import collections
import json
import random
import time

import aiohttp


def load_sessions(path, n, max_in, min_req):
    picked = []
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            reqs = t["requests"]
            if any(r.get("type") == "subagent" for r in reqs):
                continue
            main = [r for r in reqs if r.get("type") == "s"]
            if not main:
                continue
            model = collections.Counter(r["model"] for r in main).most_common(1)[0][0]
            main = [r for r in main if r["model"] == model]
            if len(main) < min_req or max(r["in"] for r in main) > max_in:
                continue
            picked.append((t["id"], main))
    picked.sort(key=lambda x: -len(x[1]))
    return picked[:n]


class Blocks:
    def __init__(self, vocab_size, block):
        self.vocab_size, self.block, self.cache = vocab_size, block, {}

    def tokens(self, sid, hash_ids):
        out = []
        for h in hash_ids:
            key = (sid, h)
            if key not in self.cache:
                rng = random.Random(f"{sid}:{h}")
                # Stay well inside the normal-text id range of the Qwen vocab.
                self.cache[key] = [
                    rng.randrange(1000, self.vocab_size) for _ in range(self.block)
                ]
            out.extend(self.cache[key])
        return out


async def run_session(idx, sid, reqs, blocks, args, http, results, t0):
    await asyncio.sleep(idx * args.stagger)
    for i, r in enumerate(reqs):
        if i:
            think = r.get("think_time") or 0.0
            await asyncio.sleep(min(think * args.time_scale, args.max_think))
        prompt = blocks.tokens(idx, r["hash_ids"])
        out = max(1, min(r["out"], args.max_out))
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": out,
            "min_tokens": out,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = time.perf_counter()
        ttft, usage = None, None
        async with http.post(f"{args.url}/v1/completions", json=body) as resp:
            if resp.status != 200:
                results.append(
                    dict(
                        session=idx,
                        i=i,
                        error=f"{resp.status} {(await resp.text())[:200]}",
                    )
                )
                continue
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if (
                    ttft is None
                    and chunk.get("choices")
                    and chunk["choices"][0].get("text")
                ):
                    ttft = time.perf_counter() - start
                if chunk.get("usage"):
                    usage = chunk["usage"]
        end = time.perf_counter()
        details = (usage or {}).get("prompt_tokens_details") or {}
        results.append(
            dict(
                session=idx,
                sid=sid[:10],
                i=i,
                t_send=start - t0,
                ttft=ttft,
                latency=end - start,
                prompt_tokens=(usage or {}).get("prompt_tokens"),
                cached=details.get("cached_tokens"),
                out=out,
                think=r.get("think_time"),
                trace_in=r["in"],
            )
        )
        print(
            f"s{idx:02d} {i:3d}/{len(reqs)} in={r['in']:6d} cached={details.get('cached_tokens')} "
            f"ttft={ttft if ttft is None else round(ttft, 3)}",
            flush=True,
        )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("out")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--max-in", type=int, default=128_000)
    ap.add_argument("--min-req", type=int, default=30)
    ap.add_argument("--max-req", type=int, default=10_000)
    ap.add_argument("--time-scale", type=float, default=0.1)
    ap.add_argument("--max-think", type=float, default=30.0)
    ap.add_argument("--max-out", type=int, default=4096)
    ap.add_argument("--stagger", type=float, default=2.0)
    ap.add_argument("--vocab-size", type=int, default=240_000)
    args = ap.parse_args()

    sessions = load_sessions(args.traces, args.sessions, args.max_in, args.min_req)
    sessions = [(sid, reqs[: args.max_req]) for sid, reqs in sessions]
    print("sessions:", [(sid[:10], len(r)) for sid, r in sessions], flush=True)
    blocks = Blocks(args.vocab_size, 64)
    results = []
    t0 = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=None)
    # A fresh connection per request: reusing an idle keep-alive connection
    # races with the server closing it and raises ServerDisconnectedError.
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http:
        await asyncio.gather(
            *(
                run_session(i, sid, reqs, blocks, args, http, results, t0)
                for i, (sid, reqs) in enumerate(sessions)
            )
        )
    meta = dict(
        args=vars(args),
        sessions=[(sid, len(r)) for sid, r in sessions],
        wall=time.perf_counter() - t0,
    )
    json.dump(dict(meta=meta, results=results), open(args.out, "w"))
    print(f"done in {meta['wall']:.0f}s, {len(results)} requests", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
