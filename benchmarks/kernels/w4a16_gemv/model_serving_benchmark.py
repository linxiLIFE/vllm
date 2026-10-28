# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare HTTP streaming latency and throughput for Humming and hybrid."""

import argparse
import csv
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def summarize(args):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for input_length in args.input_lengths:
        for concurrency in args.concurrencies:
            row = dict(
                input_tokens=input_length,
                output_tokens=args.output_length,
                concurrency=concurrency,
            )
            for backend in ("humming", "hybrid"):
                records = [
                    json.loads(
                        (
                            args.output_dir
                            / (
                                f"round{round_index}-{backend}-in{input_length}-"
                                f"c{concurrency}.json"
                            )
                        ).read_text()
                    )
                    for round_index in range(args.rounds)
                ]
                assert all(record["failed"] == 0 for record in records)
                assert all(
                    all(
                        length == args.output_length for length in record["output_lens"]
                    )
                    for record in records
                )
                row[f"{backend}_requests"] = sum(
                    record["completed"] for record in records
                )
                ttfts = [
                    value * 1000 for record in records for value in record["ttfts"]
                ]
                tpots = [
                    (latency - ttft) * 1000 / (length - 1)
                    for record in records
                    for latency, ttft, length in zip(
                        record["latencies"], record["ttfts"], record["output_lens"]
                    )
                ]
                itls = [
                    value * 1000
                    for record in records
                    for request in record["itls"]
                    for value in request
                ]
                e2els = [
                    value * 1000 for record in records for value in record["latencies"]
                ]
                for metric, values in (
                    ("ttft", ttfts),
                    ("tpot", tpots),
                    ("itl", itls),
                    ("e2el", e2els),
                ):
                    for percentile in (50, 95, 99):
                        row[f"{backend}_{metric}_p{percentile}_ms"] = float(
                            np.percentile(values, percentile)
                        )
                for metric in ("output_throughput", "request_throughput"):
                    values = [record[metric] for record in records]
                    row[f"{backend}_{metric}"] = float(np.median(values))
                    row[f"{backend}_{metric}_min"] = min(values)
                    row[f"{backend}_{metric}_max"] = max(values)
                for metric in ("ttft", "tpot"):
                    values = [record[f"median_{metric}_ms"] for record in records]
                    row[f"{backend}_{metric}_p50_ms_min"] = min(values)
                    row[f"{backend}_{metric}_p50_ms_max"] = max(values)
                row[f"{backend}_decode_tokens_per_s_per_request"] = (
                    1000 / row[f"{backend}_tpot_p50_ms"]
                )
            for metric in ("ttft", "tpot", "e2el"):
                row[f"{metric}_p50_reduction_pct"] = 100 * (
                    1
                    - (row[f"hybrid_{metric}_p50_ms"] / row[f"humming_{metric}_p50_ms"])
                )
            row["output_throughput_gain_pct"] = 100 * (
                row["hybrid_output_throughput"] / row["humming_output_throughput"] - 1
            )
            rows.append(row)
    with (args.output_dir / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2))

    figure, axes = plt.subplots(
        len(args.input_lengths),
        3,
        squeeze=False,
        figsize=(15, 4 * len(args.input_lengths)),
        layout="constrained",
    )
    for input_index, input_length in enumerate(args.input_lengths):
        selected = [row for row in rows if row["input_tokens"] == input_length]
        for metric_index, (metric, title) in enumerate(
            (
                ("ttft_p50_ms", "TTFT P50 (ms), lower is better"),
                ("tpot_p50_ms", "TPOT P50 (ms/token), lower is better"),
                ("output_throughput", "Output tokens/s, higher is better"),
            )
        ):
            axis = axes[input_index, metric_index]
            for backend, label, color in (
                ("humming", "All Humming", "#d97706"),
                ("hybrid", "Automatic dispatch", "#2563eb"),
            ):
                axis.plot(
                    args.concurrencies,
                    [row[f"{backend}_{metric}"] for row in selected],
                    marker="o",
                    color=color,
                    label=label,
                )
                axis.fill_between(
                    args.concurrencies,
                    [row[f"{backend}_{metric}_min"] for row in selected],
                    [row[f"{backend}_{metric}_max"] for row in selected],
                    color=color,
                    alpha=0.15,
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks(args.concurrencies, labels=args.concurrencies)
            axis.set_xlabel("Maximum concurrent requests")
            axis.set_title(
                f"Input {input_length}, output {args.output_length}\n{title}"
            )
            axis.grid(alpha=0.2)
            axis.legend()
    figure.suptitle(
        "Real streaming HTTP serving | Qwen2.5-1.5B INT4 | RTX 3060\n"
        "Identical fusion, decode CUDA graphs, no prefix caching"
    )
    figure.savefig(args.output_dir / "serving-metrics.png", dpi=180)
    figure.savefig(args.output_dir / "serving-metrics.svg")
    plt.close(figure)
    source = Path(__file__).resolve().parent
    metadata = dict(
        gpu=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        vllm_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        model=str(args.model),
        humming_root=str(args.humming_root),
        rounds=args.rounds,
        source_sha256={
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "model_serving_benchmark.py",
                "model_serving_worker.py",
                "model_dispatch.py",
                "model_dispatch.cu",
                "shared_layout.cu",
            )
        },
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"SAVED {args.output_dir / 'serving-metrics.png'}", flush=True)


def wait_ready(process, port: int):
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("Server did not become ready")


def run(args):
    source = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as connection:
        assert connection.connect_ex(("127.0.0.1", args.port)) != 0, "Port in use"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(source),
            str(args.humming_root.resolve()),
            environment.get("PYTHONPATH", ""),
        ]
    )
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    environment["W4A16_SERVING_BUILD"] = str(args.output_dir / "build")
    server_base = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "qwen-int4",
        "--quantization",
        "humming",
        "--dtype",
        "float16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "512",
        "--gpu-memory-utilization",
        "0.5",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--disable-log-stats",
        "--worker-cls",
        "model_serving_worker.Worker",
        "--compilation-config",
        json.dumps(
            {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 48, 64],
            }
        ),
    ]
    commands = []
    for round_index in range(args.rounds):
        backends = (
            ("humming", "hybrid") if round_index % 2 == 0 else ("hybrid", "humming")
        )
        for backend in backends:
            prefix = args.output_dir / f"round{round_index}-{backend}"
            environment["W4A16_SERVING_BACKEND"] = backend
            environment["W4A16_SERVING_STATS"] = str(prefix) + "-dispatch.json"
            print(f"START SERVER round={round_index} backend={backend}", flush=True)
            with Path(str(prefix) + "-server.log").open("w") as server_log:
                server = subprocess.Popen(
                    server_base,
                    env=environment,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    wait_ready(server, args.port)
                    cases = [
                        (length, concurrency)
                        for length in args.input_lengths
                        for concurrency in args.concurrencies
                    ]
                    if round_index % 2:
                        cases.reverse()
                    for input_length, concurrency in cases:
                        requests = max(32, concurrency * 8)
                        name = (
                            f"round{round_index}-{backend}-"
                            f"in{input_length}-c{concurrency}"
                        )
                        command = [
                            sys.executable,
                            "-m",
                            "vllm.entrypoints.cli.main",
                            "bench",
                            "serve",
                            "--backend",
                            "vllm",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(args.port),
                            "--model",
                            "qwen-int4",
                            "--tokenizer",
                            str(args.model),
                            "--dataset-name",
                            "random",
                            "--random-input-len",
                            str(input_length),
                            "--random-output-len",
                            str(args.output_length),
                            "--random-range-ratio",
                            "0",
                            "--num-prompts",
                            str(requests),
                            "--max-concurrency",
                            str(concurrency),
                            "--request-rate",
                            "inf",
                            "--ignore-eos",
                            "--temperature",
                            "0",
                            "--seed",
                            "0",
                            "--num-warmups",
                            str(max(4, concurrency)),
                            "--percentile-metrics",
                            "ttft,tpot,itl,e2el",
                            "--metric-percentiles",
                            "50,95,99",
                            "--disable-tqdm",
                            "--save-result",
                            "--save-detailed",
                            "--result-dir",
                            str(args.output_dir),
                            "--result-filename",
                            name + ".json",
                        ]
                        commands.append(
                            dict(name=name, command=command, server_command=server_base)
                        )
                        (args.output_dir / "commands.json").write_text(
                            json.dumps(commands, indent=2)
                        )
                        print(f"START {name} requests={requests}", flush=True)
                        with (args.output_dir / (name + ".log")).open("w") as log:
                            subprocess.run(
                                command,
                                env=environment,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                                timeout=300,
                            )
                        result = json.loads(
                            (args.output_dir / (name + ".json")).read_text()
                        )
                        assert result["completed"] == requests, name
                        assert result["total_output_tokens"] == (
                            requests * args.output_length
                        ), name
                        print(f"DONE {name}", flush=True)
                finally:
                    server.send_signal(signal.SIGINT)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()
            print(f"STOP SERVER round={round_index} backend={backend}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--humming-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8108)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
