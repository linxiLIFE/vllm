# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Detailed result collection for the offline throughput benchmark."""

import argparse
import csv
import importlib.metadata
import math
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

REQUEST_METRIC_FIELDS = (
    "ttft_s",
    "e2e_latency_s",
    "queue_time_s",
    "prefill_time_s",
    "decode_time_s",
    "inference_time_s",
    "mean_tpot_s",
)

REQUEST_METRIC_DEFINITIONS = {
    "ttft_s": "RequestStateStats.first_token_latency.",
    "e2e_latency_s": (
        "TTFT plus the engine-core interval from first to last generated token."
    ),
    "queue_time_s": "First scheduled timestamp minus queued timestamp.",
    "prefill_time_s": "First-token timestamp minus first scheduled timestamp.",
    "decode_time_s": "Last-token timestamp minus first-token timestamp.",
    "inference_time_s": "Last-token timestamp minus first scheduled timestamp.",
    "mean_tpot_s": (
        "Decode time divided by generated tokens minus one; recorded for n=1."
    ),
}

REQUEST_CSV_FIELDS = (
    "request_index",
    "request_id",
    "input_tokens",
    "requested_output_tokens",
    "actual_output_tokens",
    "num_sequences",
    "output_characters",
    "finish_reasons",
    "arrival_time_unix_s",
    "request_start_s",
    *REQUEST_METRIC_FIELDS,
    "num_preemptions",
    "num_cached_tokens",
    "timing_available",
)

HOST_CSV_FIELDS = (
    "elapsed_s",
    "timestamp_local",
    "stage",
    "cpu_percent_process_tree",
    "cpu_percent_process_tree_of_host",
    "cpu_percent_system",
    "rss_bytes_process_tree",
    "process_count",
    "process_read_bytes",
    "process_write_bytes",
    "process_read_bytes_per_s",
    "process_write_bytes_per_s",
    "system_memory_total_bytes",
    "system_memory_available_bytes",
    "error",
)

GPU_CSV_FIELDS = (
    "elapsed_s",
    "timestamp_local",
    "stage",
    "gpu_index",
    "gpu_name",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "memory_used_mb",
    "memory_total_mb",
    "power_w",
    "power_limit_w",
    "temperature_c",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "sm_clock_mhz",
    "pstate",
    "error",
)

TOKEN_CSV_FIELDS = (
    "request_index",
    "sequence_index",
    "token_index",
    "token_id",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(slots=True)
class ThroughputBenchmarkConfig:
    """Serializable configuration and invocation details for one benchmark."""

    run_id: str
    started_at: str
    model: str
    backend: str
    dataset_name: str
    num_prompts: int
    num_warmups: int
    seed: int | None
    sampling: dict[str, Any]
    engine: dict[str, Any]
    save_detailed: bool
    telemetry_interval_s: float
    output_json: str | None
    parameters: dict[str, Any]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ThroughputBenchmarkConfig":
        parameters = vars(args).copy()
        for field in ("hf_token", "api_key", "authorization"):
            if parameters.get(field) is not None:
                parameters[field] = "***"

        engine_keys = (
            "dtype",
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "disable_log_stats",
        )
        run_id = datetime.now().astimezone().strftime("throughput_%Y%m%d_%H%M%S_%f")
        return cls(
            run_id=run_id,
            started_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
            model=args.model,
            backend=args.backend,
            dataset_name=args.dataset_name,
            num_prompts=args.num_prompts,
            num_warmups=args.num_warmups,
            seed=args.seed,
            sampling={
                "n": args.n,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "ignore_eos": args.ignore_eos,
            },
            engine={key: getattr(args, key, None) for key in engine_keys},
            save_detailed=args.save_detailed,
            telemetry_interval_s=args.telemetry_interval_s,
            output_json=args.output_json,
            parameters=_json_safe(parameters),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float | None:
    finite = sorted(float(value) for value in values if value is not None)
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(finite) - 1)
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def summarize_values(values: list[float | int | None]) -> dict[str, float | int | None]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "min": min(finite),
        "p50": _percentile(finite, 50),
        "p90": _percentile(finite, 90),
        "p95": _percentile(finite, 95),
        "p99": _percentile(finite, 99),
        "max": max(finite),
    }


def collect_request_metrics(
    requests: list[Any],
    request_outputs: list[Any] | None,
) -> list[dict[str, Any]]:
    """Build one detailed record per prompt from final engine outputs."""
    outputs = request_outputs or []
    records = []
    for index, request in enumerate(requests):
        output = outputs[index] if index < len(outputs) else None
        completions = [
            item for item in (getattr(output, "outputs", []) or []) if item is not None
        ]
        output_tokens = sum(len(item.token_ids or []) for item in completions)
        output_characters = sum(len(item.text or "") for item in completions)
        stats = getattr(output, "metrics", None)

        record: dict[str, Any] = {
            "request_index": index,
            "request_id": getattr(output, "request_id", None),
            "input_tokens": (
                len(output.prompt_token_ids)
                if output is not None and output.prompt_token_ids is not None
                else request.prompt_len
            ),
            "requested_output_tokens": request.expected_output_len,
            "actual_output_tokens": (
                output_tokens if output is not None else None
            ),
            "num_sequences": len(completions) if output is not None else 0,
            "output_characters": output_characters,
            "finish_reasons": ",".join(
                str(item.finish_reason) for item in completions
            ),
            "arrival_time_unix_s": None,
            "request_start_s": None,
            "ttft_s": None,
            "e2e_latency_s": None,
            "queue_time_s": None,
            "prefill_time_s": None,
            "decode_time_s": None,
            "inference_time_s": None,
            "mean_tpot_s": None,
            "num_preemptions": None,
            "num_cached_tokens": getattr(output, "num_cached_tokens", None),
            "timing_available": stats is not None,
        }
        if stats is not None:
            arrival = stats.arrival_time
            first_token = stats.first_token_ts
            last_token = stats.last_token_ts
            scheduled = stats.scheduled_ts
            queued = stats.queued_ts
            record["arrival_time_unix_s"] = arrival or None
            record["ttft_s"] = stats.first_token_latency or None
            if record["ttft_s"] is not None:
                decode_time = (
                    max(0.0, last_token - first_token)
                    if first_token and last_token
                    else 0.0
                )
                record["e2e_latency_s"] = record["ttft_s"] + decode_time
            if queued and scheduled:
                record["queue_time_s"] = max(0.0, scheduled - queued)
            if scheduled and first_token:
                record["prefill_time_s"] = max(0.0, first_token - scheduled)
            if first_token and last_token:
                record["decode_time_s"] = max(0.0, last_token - first_token)
            if scheduled and last_token:
                record["inference_time_s"] = max(0.0, last_token - scheduled)
            generated_tokens = stats.num_generation_tokens
            if (
                len(completions) == 1
                and generated_tokens > 1
                and first_token
                and last_token
            ):
                record["mean_tpot_s"] = max(
                    0.0, last_token - first_token
                ) / (generated_tokens - 1)
            record["num_preemptions"] = stats.num_preemptions
        records.append(record)
    arrivals = [
        record["arrival_time_unix_s"]
        for record in records
        if record["arrival_time_unix_s"] is not None
    ]
    if arrivals:
        first_arrival = min(arrivals)
        for record in records:
            arrival = record["arrival_time_unix_s"]
            if arrival is not None:
                record["request_start_s"] = arrival - first_arrival
    return records


def collect_output_token_ids(request_outputs: list[Any] | None) -> list[dict[str, int]]:
    rows = []
    for request_index, output in enumerate(request_outputs or []):
        for sequence_index, completion in enumerate(
            getattr(output, "outputs", []) or []
        ):
            if completion is None:
                continue
            for token_index, token_id in enumerate(completion.token_ids or []):
                rows.append(
                    {
                        "request_index": request_index,
                        "sequence_index": sequence_index,
                        "token_index": token_index,
                        "token_id": token_id,
                    }
                )
    return rows


def summarize_request_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        field: summarize_values([record[field] for record in records])
        for field in REQUEST_METRIC_FIELDS
    }


def capture_environment(torch_module: Any) -> dict[str, Any]:
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = "unknown"
    try:
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        driver_version = driver.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        driver_version = None
    devices = []
    if torch_module.cuda.is_available():
        for index in range(torch_module.cuda.device_count()):
            try:
                properties = torch_module.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "compute_capability": (
                            f"{properties.major}.{properties.minor}"
                        ),
                    }
                )
            except Exception as exc:
                devices.append({"index": index, "error": str(exc)})
    return {
        "vllm_version": vllm_version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": torch_module.version.cuda,
        "nvidia_driver_version": driver_version,
        "cuda_available": torch_module.cuda.is_available(),
        "cuda_devices": devices,
        "cpu_count": psutil.cpu_count(logical=True),
        "system_memory_total_bytes": psutil.virtual_memory().total,
    }


def capture_cuda_memory(torch_module: Any) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        return {"available": False}
    try:
        device = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(device)
        return {
            "available": True,
            "scope": "current PyTorch process",
            "device_index": device,
            "device_name": properties.name,
            "total_bytes": properties.total_memory,
            "allocated_bytes": torch_module.cuda.memory_allocated(device),
            "reserved_bytes": torch_module.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch_module.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch_module.cuda.max_memory_reserved(device),
        }
    except Exception as exc:
        return {"available": True, "error": str(exc)}


def write_csv(
    path: str | Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class ThroughputTelemetrySampler:
    """Sample process/system resources and NVIDIA GPU telemetry in the background."""

    GPU_QUERY_FIELDS = (
        "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "power.draw,power.limit,temperature.gpu,clocks.gr,clocks.mem,clocks.sm,pstate"
    )

    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self.started = time.perf_counter()
        self.stage = "engine_startup"
        self.host_rows: list[dict[str, Any]] = []
        self.gpu_rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._root_process = psutil.Process(os.getpid())
        self._previous_host: dict[str, Any] | None = None
        self._previous_host_time: float | None = None
        psutil.cpu_percent(interval=None)

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(
            target=self._run, name="vllm-throughput-telemetry", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(6.0, self.interval_s * 2))

    def _process_snapshot(self) -> dict[str, Any]:
        processes = [self._root_process]
        processes.extend(self._root_process.children(recursive=True))
        cpu_seconds = 0.0
        rss_bytes = 0
        read_bytes = 0
        write_bytes = 0
        process_count = 0
        for process in processes:
            try:
                cpu_times = process.cpu_times()
                cpu_seconds += cpu_times.user + cpu_times.system
                rss_bytes += process.memory_info().rss
                process_count += 1
                try:
                    io = process.io_counters()
                    read_bytes += io.read_bytes
                    write_bytes += io.write_bytes
                except (psutil.Error, AttributeError):
                    pass
            except psutil.Error:
                continue
        return {
            "cpu_seconds": cpu_seconds,
            "rss_bytes": rss_bytes,
            "process_count": process_count,
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
        }

    def _sample_host(self, now: float) -> None:
        row: dict[str, Any] = {
            "elapsed_s": now - self.started,
            "timestamp_local": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "stage": self.stage,
            "cpu_percent_process_tree": None,
            "cpu_percent_process_tree_of_host": None,
            "cpu_percent_system": None,
            "rss_bytes_process_tree": None,
            "process_count": None,
            "process_read_bytes": None,
            "process_write_bytes": None,
            "process_read_bytes_per_s": None,
            "process_write_bytes_per_s": None,
            "system_memory_total_bytes": None,
            "system_memory_available_bytes": None,
            "error": None,
        }
        try:
            snapshot = self._process_snapshot()
            memory = psutil.virtual_memory()
            row.update(
                {
                    "rss_bytes_process_tree": snapshot["rss_bytes"],
                    "process_count": snapshot["process_count"],
                    "process_read_bytes": snapshot["read_bytes"],
                    "process_write_bytes": snapshot["write_bytes"],
                    "system_memory_total_bytes": memory.total,
                    "system_memory_available_bytes": memory.available,
                    "cpu_percent_system": psutil.cpu_percent(interval=None),
                }
            )
            if self._previous_host is not None and self._previous_host_time is not None:
                interval = now - self._previous_host_time
                if interval > 0:
                    cpu_percent = max(
                        0.0,
                        snapshot["cpu_seconds"]
                        - self._previous_host["cpu_seconds"],
                    ) / interval * 100.0
                    row["cpu_percent_process_tree"] = cpu_percent
                    row["cpu_percent_process_tree_of_host"] = cpu_percent / max(
                        1, psutil.cpu_count(logical=True) or 1
                    )
                    row["process_read_bytes_per_s"] = max(
                        0,
                        snapshot["read_bytes"] - self._previous_host["read_bytes"],
                    ) / interval
                    row["process_write_bytes_per_s"] = max(
                        0,
                        snapshot["write_bytes"] - self._previous_host["write_bytes"],
                    ) / interval
            self._previous_host = snapshot
            self._previous_host_time = now
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            self.errors.append(row["error"])
        self.host_rows.append(row)

    @staticmethod
    def _number(value: str) -> float | None:
        try:
            number = float(value.strip())
            return number if number == number and abs(number) != float("inf") else None
        except (ValueError, AttributeError):
            return None

    def _sample_gpu(self, now: float) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={self.GPU_QUERY_FIELDS}",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            for values in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                if len(values) < 13:
                    continue
                self.gpu_rows.append(
                    {
                        "elapsed_s": now - self.started,
                        "timestamp_local": timestamp,
                        "stage": self.stage,
                        "gpu_index": int(values[0]) if values[0].isdigit() else None,
                        "gpu_name": values[1].strip(),
                        "utilization_gpu_pct": self._number(values[2]),
                        "utilization_memory_pct": self._number(values[3]),
                        "memory_used_mb": self._number(values[4]),
                        "memory_total_mb": self._number(values[5]),
                        "power_w": self._number(values[6]),
                        "power_limit_w": self._number(values[7]),
                        "temperature_c": self._number(values[8]),
                        "graphics_clock_mhz": self._number(values[9]),
                        "memory_clock_mhz": self._number(values[10]),
                        "sm_clock_mhz": self._number(values[11]),
                        "pstate": values[12].strip(),
                        "error": None,
                    }
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.errors.append(error)
            self.gpu_rows.append(
                {
                    "elapsed_s": now - self.started,
                    "timestamp_local": timestamp,
                    "stage": self.stage,
                    "error": error,
                }
            )

    def _sample(self) -> None:
        now = time.perf_counter()
        self._sample_host(now)
        self._sample_gpu(now)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()
        self._sample()

    def summarize(self) -> dict[str, Any]:
        host_metrics = (
            "cpu_percent_process_tree",
            "cpu_percent_process_tree_of_host",
            "cpu_percent_system",
            "rss_bytes_process_tree",
            "process_read_bytes_per_s",
            "process_write_bytes_per_s",
            "system_memory_available_bytes",
        )
        gpu_metrics = (
            "utilization_gpu_pct",
            "utilization_memory_pct",
            "memory_used_mb",
            "power_w",
            "temperature_c",
        )
        gpu_indices = sorted(
            {
                row["gpu_index"]
                for row in self.gpu_rows
                if row.get("gpu_index") is not None
            }
        )
        stages = sorted(
            {
                row["stage"]
                for row in [*self.host_rows, *self.gpu_rows]
                if row.get("stage") is not None
            }
        )
        return {
            "interval_s": self.interval_s,
            "host_sample_count": len(self.host_rows),
            "gpu_sample_count": len(self.gpu_rows),
            "host": {
                key: summarize_values([row.get(key) for row in self.host_rows])
                for key in host_metrics
            },
            "host_by_stage": [
                {
                    "stage": stage,
                    "sample_count": sum(
                        row.get("stage") == stage for row in self.host_rows
                    ),
                    "metrics": {
                        key: summarize_values(
                            [
                                row.get(key)
                                for row in self.host_rows
                                if row.get("stage") == stage
                            ]
                        )
                        for key in host_metrics
                    },
                }
                for stage in stages
                if any(row.get("stage") == stage for row in self.host_rows)
            ],
            "gpu_by_device": [
                {
                    "gpu_index": gpu_index,
                    "gpu_name": next(
                        (
                            row["gpu_name"]
                            for row in self.gpu_rows
                            if row.get("gpu_index") == gpu_index
                        ),
                        None,
                    ),
                    "metrics": {
                        key: summarize_values(
                            [
                                row.get(key)
                                for row in self.gpu_rows
                                if row.get("gpu_index") == gpu_index
                            ]
                        )
                        for key in gpu_metrics
                    },
                    "by_stage": [
                        {
                            "stage": stage,
                            "sample_count": sum(
                                row.get("stage") == stage
                                and row.get("gpu_index") == gpu_index
                                for row in self.gpu_rows
                            ),
                            "metrics": {
                                key: summarize_values(
                                    [
                                        row.get(key)
                                        for row in self.gpu_rows
                                        if row.get("stage") == stage
                                        and row.get("gpu_index") == gpu_index
                                    ]
                                )
                                for key in gpu_metrics
                            },
                        }
                        for stage in stages
                        if any(
                            row.get("stage") == stage
                            and row.get("gpu_index") == gpu_index
                            for row in self.gpu_rows
                        )
                    ],
                }
                for gpu_index in gpu_indices
            ],
            "errors": self.errors,
        }
