# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run the real vLLM model with experimental SM86 W4A16 dispatch.

Use --backend humming|hybrid, followed by vllm bench forward arguments.
The patch is process-local; unsupported shapes retain the Humming path.
"""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from benchmark import checked
from cuda.bindings import driver
from shared_layout_benchmark import CONFIGS, ROW_CAPACITY


def install_dispatch(
    build: Path, fuse_bias: bool, validate: bool, custom_enabled: bool = True
):
    from vllm.model_executor.layers.quantization.humming import HummingLinearMethod

    source = Path(__file__).resolve().parent
    selections = json.loads((source / "results/original-selection.json").read_text())
    for relative in (
        "shared-qkv-reuse-tune/selection.json",
        "shared-qkv-crossover-tune/selection.json",
    ):
        selections.update(json.loads((source / "results" / relative).read_text()))
    build.mkdir(parents=True, exist_ok=True)
    cubin = build / "model_dispatch.cubin"
    subprocess.run(
        [
            "nvcc",
            "--cubin",
            "-std=c++17",
            "-O3",
            "-lineinfo",
            "-arch=sm_86",
            str(source / "model_dispatch.cu"),
            "-o",
            str(cubin),
        ],
        check=True,
    )
    torch.empty(0, device="cuda")
    module = checked(driver.cuModuleLoad(str(cubin).encode()))
    functions = {
        name: checked(driver.cuModuleGetFunction(module, ("model_" + name).encode()))
        for name in {value["custom"] for value in selections.values()}
    }
    counts = Counter()
    validated = set()
    plain_biases = {}

    @torch.library.custom_op("w4a16_experiment::linear", mutates_args=())
    def linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        name: str,
        n: int,
    ) -> torch.Tensor:
        m, k = x.shape
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        tiles, nw, kw, _ = CONFIGS[name]
        values = [
            ctypes.c_void_p(tensor.data_ptr() if tensor is not None else 0)
            for tensor in (x, weight, scales, out, bias)
        ]
        values += [ctypes.c_int(value) for value in (m, k, n)]
        pointers = (ctypes.c_void_p * len(values))(
            *(ctypes.addressof(value) for value in values)
        )
        checked(
            driver.cuLaunchKernel(
                functions[name],
                n // (tiles * nw * 16),
                (m + ROW_CAPACITY.get(name, 8) - 1) // ROW_CAPACITY.get(name, 8),
                1,
                nw * kw * 32,
                1,
                1,
                0,
                torch.cuda.current_stream().cuda_stream,
                ctypes.addressof(pointers),
                0,
            )
        )
        return out

    @linear.register_fake
    def linear_fake(x, weight, scales, bias, name, n):
        return x.new_empty((x.shape[0], n))

    original = HummingLinearMethod.apply
    shapes = {
        (1536, 2048): ("qkv_proj", 48),
        (1536, 1536): ("o_proj", 24),
        (1536, 17920): ("gate_up_proj", 8),
        (8960, 1536): ("down_proj", 8),
    }

    def apply(self, layer, x, bias=None):
        config = self.layer_config
        shape = (config.shape_k, config.shape_n)
        flat = x.reshape(-1, x.shape[-1])
        m = flat.shape[0]
        eligible = (
            torch.cuda.get_device_capability(x.device) == (8, 6)
            and x.dtype == torch.float16
            and x.is_contiguous()
            and shape in shapes
            and not config.has_zero_point
            and not any(
                (
                    config.use_packed_k_layout,
                    config.use_native_dequant,
                    config.use_flat_weight,
                    config.use_raw_weight,
                    config.use_int_weight_scale,
                    config.should_apply_bs_on_c,
                )
            )
            and layer.weight.dtype == torch.int32
            and tuple(layer.weight.shape) == (shape[0] // 16, shape[1] * 2)
            and tuple(layer.weight_scale.shape) == (shape[0] // 128, shape[1])
            and self.weight_schema.hadamard_block_size == 0
        )
        if eligible:
            projection, cutoff = shapes[shape]
            key = f"{projection}/M{m}"
            if m <= cutoff and key in selections and custom_enabled:
                counts[(projection, m, "custom")] += 1
                out = linear(
                    flat,
                    layer.weight,
                    layer.weight_scale,
                    bias if fuse_bias else None,
                    selections[key]["custom"],
                    shape[1],
                )
                if bias is not None and not fuse_bias:
                    if id(layer) not in plain_biases:
                        columns = torch.arange(shape[1], device=bias.device)
                        permutation = (
                            columns // 32 * 32
                            + columns % 2
                            + columns % 8 // 2 * 8
                            + columns % 32 // 8 * 2
                        )
                        plain_biases[id(layer)] = bias[permutation]
                    out = out + plain_biases[id(layer)]
                validation_key = (id(layer), m)
                if (
                    validate
                    and validation_key not in validated
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    reference = original(self, layer, x, bias).reshape_as(out)
                    relative_l2 = (
                        out.float() - reference.float()
                    ).norm() / reference.float().norm().clamp_min(1e-8)
                    assert torch.isfinite(out).all() and relative_l2 < 0.002, (
                        projection,
                        m,
                        relative_l2.item(),
                    )
                    validated.add(validation_key)
                    counts[(projection, m, "validated")] += 1
                return out.view(*x.shape[:-1], shape[1])
        counts[(str(shape), m, "humming")] += 1
        return original(self, layer, x, bias)

    HummingLinearMethod.apply = apply
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("humming", "hybrid"), required=True)
    parser.add_argument("--dispatch-build", type=Path, required=True)
    parser.add_argument("--unfused-bias", action="store_true")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--generation-logprobs", type=int, default=5)
    parser.add_argument("--generation-batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--validate-layers", action="store_true")
    parser.add_argument("--dispatch-stats", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.dispatch_stats.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    metadata = dict(
        command=sys.argv,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        vllm_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        backend=args.backend,
        fused_bias=not args.unfused_bias,
        layer_validation_relative_l2=0.002,
        source_sha256={
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("model_dispatch.py", "model_dispatch.cu", "shared_layout.cu")
        },
    )
    args.dispatch_stats.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    counts = (
        install_dispatch(
            args.dispatch_build, not args.unfused_bias, args.validate_layers
        )
        if args.backend == "hybrid"
        else Counter()
    )
    from vllm.benchmarks.forward.main import add_cli_args
    from vllm.benchmarks.forward.main import main as forward_main
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    forward_parser = FlexibleArgumentParser()
    add_cli_args(forward_parser)
    forward_args = forward_parser.parse_args(remaining)
    if args.generation:
        from vllm import LLM, SamplingParams
        from vllm.config.compilation import CompilationMode, CUDAGraphMode
        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs.from_cli_args(forward_args)
        engine_args.compilation_config.mode = CompilationMode.NONE
        engine_args.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
        engine_args.compilation_config.cudagraph_capture_sizes = [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
        ]
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        llm = LLM.from_engine_args(engine_args)
        prompts = [
            "Explain why the sky is blue in simple terms.",
            "计算 17 乘以 23，并解释计算过程。",
            "Write a Python function that returns Fibonacci numbers.",
            "Name three differences between CPUs and GPUs.",
        ]
        records = []
        for batch in args.generation_batches:
            inputs = [prompts[index % len(prompts)] for index in range(batch)]
            for length in (1, 65):
                sampling = SamplingParams(
                    temperature=0,
                    max_tokens=length,
                    ignore_eos=True,
                    logprobs=args.generation_logprobs or None,
                )
                llm.generate(inputs, sampling, use_tqdm=False)
                durations = []
                for _ in range(3):
                    start = time.perf_counter()
                    outputs = llm.generate(inputs, sampling, use_tqdm=False)
                    durations.append(time.perf_counter() - start)
                records.append(
                    dict(
                        batch=batch,
                        output_tokens=length,
                        seconds=durations,
                        outputs=[
                            dict(
                                tokens=output.outputs[0].token_ids,
                                text=output.outputs[0].text,
                                logprobs=[
                                    {
                                        str(token): entry.logprob
                                        for token, entry in row.items()
                                    }
                                    for row in (output.outputs[0].logprobs or [])
                                ],
                            )
                            for output in outputs
                        ],
                    )
                )
            Path(forward_args.output_json).write_text(json.dumps(records, indent=2))
    else:
        forward_main(forward_args)
    args.dispatch_stats.write_text(
        json.dumps({str(key): value for key, value in counts.items()}, indent=2)
    )


if __name__ == "__main__":
    main()
