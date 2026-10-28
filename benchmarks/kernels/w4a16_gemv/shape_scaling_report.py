# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Plot the independent shape_scaling.py confirmation run."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main(directory):
    with (directory / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    shapes = list(dict.fromkeys(row["shape"] for row in rows))
    batches = sorted({int(row["m"]) for row in rows})
    panels = [("fused", "True"), ("fused", "False")]
    if any(row["bias"] == "none" for row in rows):
        panels.append(("none", "True"))
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels), 7), constrained_layout=True
    )
    norm = TwoSlopeNorm(vmin=0.5, vcenter=1, vmax=1.5)
    for ax, (bias, cold) in zip(axes, panels):
        values = np.array(
            [
                [
                    float(
                        next(
                            row["speedup"]
                            for row in rows
                            if row["shape"] == shape
                            and int(row["m"]) == m
                            and row["bias"] == bias
                            and row["cold"] == cold
                        )
                    )
                    for m in batches
                ]
                for shape in shapes
            ]
        )
        image = ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(batches)), batches)
        ax.set_yticks(range(len(shapes)), shapes)
        ax.set_xlabel("M (rows in this GEMM)")
        ax.set_title(
            f"{'Fused bias' if bias == 'fused' else 'No bias'}, "
            f"{'cold' if cold == 'True' else 'warm'} L2"
        )
        for i in range(len(shapes)):
            for j in range(len(batches)):
                ax.text(
                    j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=9
                )
    fig.colorbar(
        image, ax=axes, label="Humming / custom latency (higher is better)", shrink=0.7
    )
    fig.suptitle("RTX 3060 · synthetic W4A16 · fixed configurations · TP=1")
    fig.savefig(directory / "speedup.png", dpi=180)
    fig.savefig(directory / "speedup.svg")
    plt.close(fig)

    checks = json.loads((directory / "checks.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    rounds = metadata.get("rounds", 2)
    warmup = metadata.get("warmup_ms", 10)
    repeats = metadata.get("repeats", 10)
    lines = [
        "# DeepSeek V4.1 Flash projection shape simulation",
        "",
        "Synthetic symmetric INT4 group128, FP16 activations/dequant/output, "
        "FP32 accumulation, TP=1, RTX 3060. No checkpoint weights were downloaded.",
        "",
        "Shape definitions follow the [official inference implementation]("
        + metadata.get(
            "shape_source",
            "https://huggingface.co/deepseek-ai/"
            "DeepSeek-V4.1-Flash/blob/main/inference/model.py",
        )
        + "). Expert gate/up is the concatenation of w1/w3; "
        "o_a_group uses the actual block-diagonal group dimensions.",
        "",
        "Tuning sweeps custom configurations (K warps, output width, row reuse "
        "and optional register prefetch) and Humming "
        "default/stage-2/no-Stream-K variants "
        "where applicable. This is a coarse sweep, not an exhaustive search.",
        "",
        "Configurations were selected in a separate tuning run. Confirmation "
        f"uses {rounds} rounds with reversed implementation order, {warmup} ms "
        f"warmup and {repeats} CUPTI graph samples per "
        "implementation/cache/shape/bias case. "
        "The reported value is the median of round medians. Allocation, initial "
        "weight transform, compilation, and cold-L2 flush are excluded.",
        "",
        "Both implementations read the same Humming weight/scale allocation and "
        "use the same transformed nonzero bias in the fused-bias comparison. "
        "No extra bias-add kernel is included in either main comparison arm.",
        "",
        f"Correctness: {len(checks)} comparisons against independent FP32 GEMM "
        "of decoded FP16 weights; relative L2 < 0.001 and elementwise "
        "atol=max(0.002, 0.002*reference_RMS), rtol=0.02. "
        f"Max relative L2: {max(c['relative_l2_error'] for c in checks):.6g}.",
        "",
        "## Fused bias, cold L2",
        "",
        "Each cell is custom / Humming in microseconds, followed by speedup.",
        "",
        "| Projection (K × N) | " + " | ".join(f"M={m}" for m in batches) + " |",
        "| --- | " + " | ".join("---:" for _ in batches) + " |",
    ]
    for shape in shapes:
        selected = [
            row
            for row in rows
            if row["shape"] == shape
            and row["bias"] == "fused"
            and row["cold"] == "True"
        ]
        first = selected[0]
        cells = []
        for m in batches:
            row = next(row for row in selected if int(row["m"]) == m)
            cells.append(
                f"{float(row['custom_us']):.2f} / "
                f"{float(row['humming_us']):.2f} "
                f"({float(row['speedup']):.3f}×)"
            )
        lines.append(
            f"| {shape} ({first['k']} × {first['n']}) | " + " | ".join(cells) + " |"
        )
    if rows[0].get("previous_us"):
        lines += [
            "",
            "## Previous custom versus optimized custom, cold L2",
            "",
            "Both custom versions and Humming were measured in the same run.",
            "",
            "| Shape | M | Previous (µs) | Optimized (µs) | Previous / optimized |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            if row["cold"] == "True" and row["bias"] == "fused":
                old, new = float(row["previous_us"]), float(row["custom_us"])
                lines.append(
                    f"| {row['shape']} | {row['m']} | {old:.2f} | "
                    f"{new:.2f} | {old / new:.3f}× |"
                )
    lines += [
        "",
        "## Scope",
        "",
        "The official model uses native FP8/FP4 and bias-free projections. "
        "This experiment borrows its matrix shapes and tests our current "
        "W4A16 implementation; it is not a measurement of native model inference.",
        "",
        "`v41_o_a_group` is one of eight independent group projections; "
        "`v41_expert_gate_up` stacks gate/up for one expert. Routed experts are "
        "not merged into a dense MLP. M means tokens assigned to this expert "
        "when interpreting the expert rows. No routing, grouped-MoE scheduling, "
        "communication, attention, KV cache, or nonlinear operations are timed.",
        "",
        "GB/s in summary.csv uses ideal bytes: K*N/2 + 2*(K/128)*N + "
        "2*M*(K+N), plus 2*N for bias. TFLOPS = 2*M*N*K / elapsed time. "
        "These are useful-work metrics, not measured DRAM traffic.",
        "",
        "Full cold/warm, bias/no-bias results, round medians and selected "
        "configurations are in summary.csv. Raw samples are in samples.json.",
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    main(parser.parse_args().input_dir)
