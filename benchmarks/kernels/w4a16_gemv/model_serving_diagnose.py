# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare fixed request waves and immediate refill, with step/clock telemetry."""

import argparse
import asyncio
import dataclasses
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import aiohttp
from model_serving_benchmark import wait_ready


async def client(model, output, port):
    from vllm.benchmarks.datasets import RandomDataset
    from vllm.benchmarks.lib.endpoint_request_func import (
        RequestFuncInput,
        async_request_openai_completions,
    )
    from vllm.tokenizers import get_tokenizer

    prompts = RandomDataset(random_seed=0).sample(
        get_tokenizer(str(model)),
        num_requests=16,
        input_len=128,
        output_len=256,
        range_ratio=0,
    )
    records = []
    semaphore = asyncio.Semaphore(4)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:

        async def request(index):
            async with semaphore:
                prompt = prompts[index % len(prompts)]
                result = await async_request_openai_completions(
                    RequestFuncInput(
                        prompt=prompt.prompt,
                        api_url=f"http://127.0.0.1:{port}/v1/completions",
                        prompt_len=prompt.prompt_len,
                        output_len=256,
                        model="qwen-int4",
                        model_name="qwen-int4",
                        ignore_eos=True,
                        extra_body={"temperature": 0},
                    ),
                    session,
                )
                assert result.success and result.output_tokens == 256, result.error
                return dataclasses.asdict(result)

        await asyncio.gather(*(request(index) for index in range(4)))
        for repeat in range(3):
            order = ("waves", "refill") if repeat % 2 == 0 else ("refill", "waves")
            for protocol in order:
                begin = time.perf_counter()
                results = []
                if protocol == "waves":
                    for first in range(0, 16, 4):
                        results.extend(
                            await asyncio.gather(
                                *(request(index) for index in range(first, first + 4))
                            )
                        )
                else:
                    results = await asyncio.gather(
                        *(request(index) for index in range(16))
                    )
                records.append(
                    dict(
                        protocol=protocol,
                        repeat=repeat,
                        start=begin,
                        end=time.perf_counter(),
                        requests=results,
                    )
                )
                output.write_text(json.dumps(records, indent=2))
                print(protocol, repeat, records[-1]["end"] - begin, flush=True)


def telemetry(stop, path):
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    rows = []
    while not stop.is_set():
        rows.append(
            dict(
                time=time.perf_counter(),
                sm_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                memory_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
                gpu_util=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                power_mw=pynvml.nvmlDeviceGetPowerUsage(handle),
            )
        )
        stop.wait(0.2)
    path.write_text(json.dumps(rows, indent=2))
    pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--humming-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8108)
    parser.add_argument("--client", action="store_true")
    args = parser.parse_args()
    if args.client:
        asyncio.run(client(args.model, args.output_dir, args.port))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as connection:
        assert connection.connect_ex(("127.0.0.1", args.port)) != 0, "Port in use"
    source = Path(__file__).resolve().parent
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source), str(args.humming_root), environment.get("PYTHONPATH", "")]
    )
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    environment["W4A16_SERVING_BUILD"] = str(args.output_dir / "build")
    commands = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=telemetry,
        args=(stop, args.output_dir / "gpu-telemetry.json"),
        daemon=True,
    )
    sampler.start()
    try:
        for index, backend in enumerate(
            ("humming", "hybrid", "wrapped", "hybrid", "humming")
        ):
            prefix = args.output_dir / f"arm{index}-{backend}"
            environment["W4A16_SERVING_BACKEND"] = backend
            environment["W4A16_SERVING_STATS"] = str(prefix) + "-dispatch.json"
            environment["W4A16_STEP_DIAGNOSTICS"] = str(prefix) + "-steps.json"
            command = [
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
                "--shutdown-timeout",
                "15",
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
            commands.append(command)
            (args.output_dir / "commands.json").write_text(
                json.dumps(commands, indent=2)
            )
            print("START", index, backend, flush=True)
            with Path(str(prefix) + "-server.log").open("w") as log:
                server = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    wait_ready(server, args.port)
                    with Path(str(prefix) + "-client.log").open("w") as client_log:
                        subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--client",
                                "--model",
                                str(args.model),
                                "--humming-root",
                                str(args.humming_root),
                                "--output-dir",
                                str(prefix) + "-requests.json",
                                "--port",
                                str(args.port),
                            ],
                            env=environment,
                            stdout=client_log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=180,
                        )
                finally:
                    server.send_signal(signal.SIGINT)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()
            assert Path(str(prefix) + "-steps.json").is_file(), prefix
            print("DONE", index, backend, flush=True)
    finally:
        stop.set()
        sampler.join(timeout=10)


if __name__ == "__main__":
    main()
