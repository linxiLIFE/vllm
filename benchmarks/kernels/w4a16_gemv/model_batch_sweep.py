# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure each decode batch with real weights and plot the dispatch comparison."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def summarize(output: Path, maximum: int, rounds: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measurements = {}
    for backend in ("humming", "hybrid"):
        for round_index in range(rounds):
            records = json.loads(
                (output / f"round{round_index}-{backend}.json").read_text()
            )
            assert len(records) == maximum
            for record in records:
                batch = len(record["requests"])
                rank = record["ranks"][0]
                assert rank["finite"], (backend, batch)
                assert rank["graph_deterministic"], (backend, batch)
                assert rank["num_tokens_after_padding"] == batch
                values = rank["latency_us"]["samples_us"]
                measurements.setdefault((backend, batch), []).append(values)

    rows = []
    for batch in range(1, maximum + 1):
        row = {"batch": batch}
        for backend in ("humming", "hybrid"):
            runs = measurements[backend, batch]
            medians = [float(np.median(values)) for values in runs]
            row[f"{backend}_us"] = float(np.median(medians))
            row[f"{backend}_round_min_us"] = min(medians)
            row[f"{backend}_round_max_us"] = max(medians)
        row["speedup"] = row["humming_us"] / row["hybrid_us"]
        ratios = [
            float(np.median(measurements["humming", batch][round_index]))
            / float(np.median(measurements["hybrid", batch][round_index]))
            for round_index in range(rounds)
        ]
        row["speedup_round_min"] = min(ratios)
        row["speedup_round_max"] = max(ratios)
        rows.append(row)
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    batches = np.arange(1, maximum + 1)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        layout="constrained",
    )
    for backend, label, color in (
        ("humming", "All Humming", "#d97706"),
        ("hybrid", "Custom before boundary, Humming after", "#2563eb"),
    ):
        axes[0].plot(
            batches,
            [row[f"{backend}_us"] / 1000 for row in rows],
            label=label,
            color=color,
            marker="o",
            markersize=3,
        )
        axes[0].fill_between(
            batches,
            [row[f"{backend}_round_min_us"] / 1000 for row in rows],
            [row[f"{backend}_round_max_us"] / 1000 for row in rows],
            color=color,
            alpha=0.15,
        )
    axes[0].set_ylabel("Transformer forward latency (ms)")
    axes[0].legend(loc="upper left")
    axes[0].set_title(
        "Real Qwen2.5-1.5B GPTQ INT4 | RTX 3060 | FP16 | TP=1 | KV=128\n"
        "Exact decode batches, CUDA graphs, identical fusion boundaries"
    )
    axes[1].plot(
        batches,
        [row["speedup"] for row in rows],
        color="#15803d",
        marker="o",
        markersize=3,
    )
    axes[1].axhline(1, color="#555555", linewidth=1)
    axes[1].fill_between(
        batches,
        [row["speedup_round_min"] for row in rows],
        [row["speedup_round_max"] for row in rows],
        color="#15803d",
        alpha=0.15,
    )
    axes[1].set_title(
        "Shading: range across rounds, not a confidence interval",
        fontsize=9,
        loc="left",
    )
    axes[1].set_ylabel("Humming / hybrid latency\n(above 1 = hybrid faster)")
    axes[1].set_xlabel("Decode batch size (one token per request)")
    for boundary, label in (
        (9, "MLP → Humming"),
        (25, "O → Humming"),
        (49, "QKV → Humming"),
    ):
        if boundary <= maximum:
            for axis in axes:
                axis.axvline(boundary, color="#777777", linestyle="--", alpha=0.6)
            axes[0].text(
                boundary + 0.3,
                0.03,
                label,
                rotation=90,
                transform=axes[0].get_xaxis_transform(),
                fontsize=9,
            )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(1, maximum)
    ticks = {1, *range(4, maximum + 1, 4), maximum}
    for boundary in (9, 25, 49):
        if boundary <= maximum:
            ticks.discard(boundary - 1)
            ticks.add(boundary)
    axes[1].set_xticks(sorted(ticks))
    figure.savefig(output / "batch-sweep.png", dpi=180)
    figure.savefig(output / "batch-sweep.svg")
    plt.close(figure)
    print(f"SAVED {output / 'batch-sweep.png'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--humming-root", type=Path, required=True)
    parser.add_argument("--max-batch", type=int, default=52)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        source = Path(__file__).resolve().parent
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(args.humming_root.resolve()), environment.get("PYTHONPATH", "")]
        )
        for round_index in range(args.rounds):
            order = (
                ("humming", "hybrid") if round_index % 2 == 0 else ("hybrid", "humming")
            )
            for backend in order:
                batches = list(range(1, args.max_batch + 1))
                if (round_index + (backend == "hybrid")) % 2:
                    batches.reverse()
                prefix = args.output_dir / f"round{round_index}-{backend}"
                command = [
                    sys.executable,
                    str(source / "model_dispatch.py"),
                    "--backend",
                    backend,
                    "--dispatch-build",
                    str(args.output_dir / "build"),
                    "--dispatch-stats",
                    str(prefix.with_suffix(".stats.json")),
                    "--model",
                    str(args.model),
                    "--load-format",
                    "auto",
                    "--quantization",
                    "humming",
                    "--dtype",
                    "float16",
                    "--max-model-len",
                    "1024",
                    "--max-num-seqs",
                    "64",
                    "--max-num-batched-tokens",
                    "512",
                    "--gpu-memory-utilization",
                    "0.5",
                    "--kv-len-jitter",
                    "0",
                    "--iters",
                    str(args.iters),
                    "--samples",
                    str(args.samples),
                    "--output-json",
                    str(prefix.with_suffix(".json")),
                ]
                if backend == "hybrid":
                    command.append("--validate-layers")
                for batch in batches:
                    command.extend(["--case", f"decode{batch}={batch}*1@128"])
                print(f"START round={round_index} backend={backend}", flush=True)
                with prefix.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        command,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                print(f"DONE round={round_index} backend={backend}", flush=True)
    summarize(args.output_dir, args.max_batch, args.rounds)


if __name__ == "__main__":
    main()
