# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import collections
import json
import sys

reqs = json.load(open(sys.argv[1]))
B = int(sys.argv[2])
conv = collections.defaultdict(list)
for r in reqs:
    conv[r["conversation_id"]].append(r)
tot = tok_ideal = blk_ideal = 0
first_turn = 0
for cid, rs in conv.items():
    rs.sort(key=lambda r: r["input_num_turns"])
    prev = None
    for r in rs:
        n = r["input_num_tokens"]
        tot += n
        if prev is None:
            first_turn += n
            reuse = 0
        else:
            reuse = min(prev["input_num_tokens"] + prev["output_num_tokens"], n - 1)
        tok_ideal += reuse
        blk_ideal += (reuse // B) * B
        prev = r
print(f"requests {len(reqs)}  conversations {len(conv)}  prompt tokens {tot}")
print(
    f"first-turn tokens (never cacheable without cross-conv sharing) {first_turn} ({first_turn / tot * 100:.1f}%)"
)
print(f"ideal hit, token granularity     {tok_ideal / tot * 100:.2f}%")
print(f"ideal hit, {B}-token block align  {blk_ideal / tot * 100:.2f}%")
