# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check GPU timing variation on the unchanged logits-projection shape alone."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    inputs = torch.ones((4, 1536), device="cuda", dtype=torch.float16)
    weight = torch.ones((151936, 1536), device="cuda", dtype=torch.float16)
    output = torch.empty((4, 151936), device="cuda", dtype=torch.float16)

    def operation():
        torch.mm(inputs, weight.T, out=output)

    for _ in range(10):
        operation()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 1536))
    records = []
    torch.cuda.profiler.start()
    for repeat in range(2):
        for synchronized in (False, True):
            torch.cuda.nvtx.range_push(f"repeat{repeat}/sync{synchronized}")
            deadline = time.perf_counter() + 1
            while time.perf_counter() < deadline:
                graph.replay()
                torch.accelerator.synchronize()
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(1000)
            ]
            begin = time.perf_counter()
            for start, end in events:
                start.record()
                graph.replay()
                end.record()
                if synchronized:
                    end.synchronize()
            torch.accelerator.synchronize()
            durations = [start.elapsed_time(end) for start, end in events]
            records.append(
                dict(
                    repeat=repeat,
                    synchronized=synchronized,
                    wall_ms=(time.perf_counter() - begin) * 1000,
                    samples_ms=durations,
                    percentiles_ms=np.percentile(durations, [10, 50, 90, 99]).tolist(),
                    effective_gbps=weight.numel()
                    * weight.element_size()
                    / np.median(durations)
                    / 1e6,
                )
            )
            torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()
    args.output.write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                shape=[4, 151936, 1536],
                records=records,
            ),
            indent=2,
        )
    )
    for record in records:
        print(record["repeat"], record["synchronized"], record["percentiles_ms"])


if __name__ == "__main__":
    main()
