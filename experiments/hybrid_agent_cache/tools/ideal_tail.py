# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ideal hit rate if only the previous turn's prompt tail (rounded down to `unit`) can be reused."""

import collections
import json
import sys

reqs = json.load(open(sys.argv[1]))
conv = collections.defaultdict(list)
for r in reqs:
    conv[r["conversation_id"]].append(r)
tot = sum(r["input_num_tokens"] for r in reqs)
for unit in map(int, sys.argv[2:]):
    hit = 0
    for rs in conv.values():
        rs.sort(key=lambda r: r["input_num_turns"])
        for prev, cur in zip(rs, rs[1:]):
            hit += (
                min(prev["input_num_tokens"] - 1, cur["input_num_tokens"] - 1) // unit
            ) * unit
    print(f"unit {unit:>4}: prompt-tail ideal hit {hit / tot * 100:.2f}%")
