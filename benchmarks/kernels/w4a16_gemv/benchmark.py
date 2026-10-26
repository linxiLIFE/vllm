# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare independent-output CUDA GEMV kernels with Marlin on real weights."""

import argparse
import csv
import ctypes
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from cuda.bindings import driver
from flashinfer.testing import bench_gpu_time_with_cupti
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.scalar_type import scalar_types

VARIANTS = {
    "gemv_scalar_w4": (4, 1),
    "gemv_scalar_w8": (8, 1),
    "gemv_pair_w4_o1": (4, 1),
    "gemv_pair_w8_o1": (8, 1),
    "gemv_pair_w4_o2": (4, 2),
    "gemv_pair_w8_o2": (8, 2),
    "gemv_pair_w4_o4": (4, 4),
    "gemv_pair_w8_o4": (8, 4),
    "gemv_vector_w4_o2": (4, 2),
    "gemv_vector_w8_o2": (8, 2),
    "gemv_spread_w4_o1": (4, 1),
    "gemv_spread_w8_o1": (8, 1),
    "gemv_spread_w8_o2": (8, 2),
    "gemv_spread_w4_o4": (4, 4),
    "gemv_spread_w8_o4": (8, 4),
}
THREADS = {}
SPLITS = {}
LOCAL_SPLITS = {}
for warps, outputs, split_counts in ((4, 2, (2, 4, 8)), (4, 1, (2, 4)), (2, 2, (2,))):
    for splits in split_counts:
        name = f"gemv_local{splits}_w{warps}_o{outputs}"
        VARIANTS[name] = (warps, outputs)
        LOCAL_SPLITS[name] = splits
for warps, outputs, split_counts in (
    (4, 2, (2, 4, 8, 16)),
    (4, 1, (2, 4, 8)),
    (8, 2, (2, 4, 8)),
):
    for splits in split_counts:
        name = f"gemv_split{splits}_w{warps}_o{outputs}"
        VARIANTS[name] = (warps, outputs)
        SPLITS[name] = splits
for threads in (8, 16):
    for warps in (4, 8):
        for outputs in (1, 2):
            name = f"gemv_t{threads}_w{warps}_o{outputs}"
            VARIANTS[name] = (warps, outputs)
            THREADS[name] = threads
for kind in ("broadcast", "float", "k1536"):
    for threads in (16, 32):
        name = f"gemv_{kind}_t{threads}"
        VARIANTS[name] = (4, 2)
        THREADS[name] = threads
for kind in ("spread", "prefetch", "spread_prefetch", "spread_k1536"):
    for threads in (8, 16, 32) if kind == "spread" else (16, 32):
        name = f"gemv_{kind}_t{threads}"
        VARIANTS[name] = (4, 2)
        THREADS[name] = threads
for name, threads in (
    ("gemv_ilp_t32", 32),
    ("gemv_ilp_spread_t32", 32),
    ("gemv_ilp_spread_t16", 16),
):
    VARIANTS[name] = (4, 2)
    THREADS[name] = threads
LAYERS = {
    "qkv_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "o_proj": ["self_attn.o_proj"],
    "gate_up_proj": ["mlp.gate_proj", "mlp.up_proj"],
    "down_proj": ["mlp.down_proj"],
}
DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct-GPTQ-Int4"
    / "snapshots/4f5a0e31008e2966e8f787964bcc4e1105e03dc3"
)


def checked(result):
    status, *values = result
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {status}")
    return values[0] if len(values) == 1 else values


class GemvModule:
    def __init__(self, build_dir):
        source = Path(__file__).resolve().parent
        capability = torch.cuda.get_device_capability()
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build_dir),
                "-G",
                "Ninja",
                f"-DGEMV_ARCH={capability[0]}{capability[1]}",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(build_dir)], check=True)
        self.module = checked(
            driver.cuModuleLoad(str(build_dir / "gemv.cubin").encode())
        )
        self.functions = {
            name: checked(driver.cuModuleGetFunction(self.module, name.encode()))
            for name in VARIANTS
        }
        self.reduce = checked(
            driver.cuModuleGetFunction(self.module, b"gemv_reduce_split")
        )

    def launch(self, name, x, weight, scales, out, partial=None):
        warps, outputs = VARIANTS[name]
        threads = THREADS.get(name, 32)
        k, n = x.numel(), out.numel()
        assert k % 128 == 0
        if "k1536" in name:
            assert k == 1536
        if "prefetch" in name:
            assert k // 8 >= threads
        assert x.dtype == out.dtype == scales.dtype == torch.float16
        assert weight.dtype == torch.int32
        assert weight.shape == (n, k // 8)
        assert scales.shape == (n, k // 128)
        splits = SPLITS.get(name, 1)
        local_splits = LOCAL_SPLITS.get(name, 1)
        if local_splits > 1:
            assert k // 128 >= local_splits
            assert n % (warps * outputs) == 0
        if splits > 1:
            assert k // 128 >= splits
            assert partial.shape == (splits, n) and partial.dtype == torch.float32
        args = [ctypes.c_void_p(t.data_ptr()) for t in (x, weight, scales, out)] + [
            ctypes.c_int(k),
            ctypes.c_int(n),
        ]
        if splits > 1:
            args.append(ctypes.c_void_p(partial.data_ptr()))
        pointers = (ctypes.c_void_p * len(args))(
            *(ctypes.addressof(arg) for arg in args)
        )
        checked(
            driver.cuLaunchKernel(
                self.functions[name],
                (n + warps * 32 // threads * outputs - 1)
                // (warps * 32 // threads * outputs),
                splits,
                1,
                warps * 32 * local_splits,
                1,
                1,
                0,
                torch.cuda.current_stream().cuda_stream,
                ctypes.addressof(pointers),
                0,
            )
        )
        if splits > 1:
            args = [
                ctypes.c_void_p(partial.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int(n),
                ctypes.c_int(splits),
            ]
            pointers = (ctypes.c_void_p * len(args))(
                *(ctypes.addressof(arg) for arg in args)
            )
            checked(
                driver.cuLaunchKernel(
                    self.reduce,
                    (n + 255) // 256,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    torch.cuda.current_stream().cuda_stream,
                    ctypes.addressof(pointers),
                    0,
                )
            )


def load_weights(model, names, layer_index=0):
    packed, scales = [], []
    with safe_open(model / "model.safetensors", framework="pt") as checkpoint:
        for name in names:
            prefix = f"model.layers.{layer_index}.{name}"
            q = checkpoint.get_tensor(prefix + ".qweight")
            s = checkpoint.get_tensor(prefix + ".scales")
            zero = checkpoint.get_tensor(prefix + ".qzeros")
            group = checkpoint.get_tensor(prefix + ".g_idx")
            assert torch.all(zero == 0x77777777)
            torch.testing.assert_close(
                group.long(), torch.arange(q.shape[0] * 8) // 128
            )
            packed.append(q)
            scales.append(s)
    packed = torch.cat(packed, dim=1).contiguous()
    scales = torch.cat(scales, dim=1).contiguous()
    k, n = packed.shape[0] * 8, packed.shape[1]
    unsigned = packed.numpy().view(np.uint32)
    shifts = np.arange(8, dtype=np.uint32)[None, :, None] * 4
    integers = ((unsigned[:, None, :] >> shifts) & 15).reshape(k, n)
    reference = (integers.astype(np.float32) - 8) * np.repeat(
        scales.numpy().astype(np.float32), 128, axis=0
    )
    reference = torch.from_numpy(reference.astype(np.float16)).cuda()
    return {
        "k": k,
        "n": n,
        "custom_weight": packed.T.contiguous().cuda(),
        "custom_scales": scales.T.contiguous().cuda(),
        "marlin_weight": ops.gptq_marlin_repack(
            packed.cuda(), size_k=k, size_n=n, num_bits=4
        ),
        "marlin_scales": marlin_permute_scales(
            scales.cuda(), size_k=k, size_n=n, group_size=128
        ),
        "reference": reference,
    }


def verify(actual, reference):
    relative = (
        (actual.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-20)
    ).item()
    assert relative < 0.001, relative
    torch.testing.assert_close(actual.float(), reference.float(), atol=0.002, rtol=0.02)
    return relative


def measure(fn, inputs, cold, repeats):
    for _ in range(10):
        fn(*inputs)
    torch.accelerator.synchronize()
    return (
        np.asarray(
            bench_gpu_time_with_cupti(
                fn,
                input_args=inputs,
                use_cuda_graph=True,
                cold_l2_cache=cold,
                dry_run_time_ms=100,
                repeat_iters=repeats,
            )
        )
        * 1000
    )


def main(args):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.init()
    torch.empty(0, device="cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = GemvModule(args.output_dir / "build")
    workspace = marlin_make_workspace_new(torch.device("cuda:0"), max_blocks_per_sm=4)
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": str(args.model),
        "layer_index": args.layer_index,
        "configuration": "M=1; group128; FP16 input/output; FP32 accumulation",
        "timing": "GPU graph replay, cold rotating arguments and warm buffers",
        "excluded": "packing, compilation, allocation, Python/driver dispatch",
        "split_k": (
            "FP32 global partials plus timed reduction kernel, or shared partials "
            "with block barrier; complete operation timed"
        ),
        "correctness": "FP32 reference using identical FP16 dequantized weights",
        "profile": args.profile,
        "variants": args.variants,
        "repeats": args.repeats,
    }
    try:
        metadata["cupti"] = version("cupti-python")
    except Exception:
        metadata["cupti"] = "absent; FlashInfer CUDA graph/event fallback"
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    rows, correctness = [], []
    for layer in args.layers:
        weights = load_weights(args.model, LAYERS[layer], args.layer_index)
        k, n = weights["k"], weights["n"]
        x = torch.randn(k, device="cuda", dtype=torch.float16)
        out = torch.empty(n, device="cuda", dtype=torch.float16)

        def marlin(a, b, scales, output, locks, atomic=False, k=k, n=n):
            return ops.marlin_gemm(
                a.view(1, k),
                output.view(1, n),
                b,
                None,
                scales,
                None,
                None,
                None,
                locks,
                scalar_types.uint4b8,
                1,
                n,
                k,
                atomic,
                True,
                False,
            )

        variants = {}
        for atomic in (False, True):
            variants[f"marlin_atomic{int(atomic)}"] = (
                marlin,
                (
                    x,
                    weights["marlin_weight"],
                    weights["marlin_scales"],
                    out,
                    workspace,
                    atomic,
                ),
            )
        for name in VARIANTS:
            if "k1536" in name and k != 1536:
                continue
            if SPLITS.get(name, 1) > k // 128:
                continue
            if LOCAL_SPLITS.get(name, 1) > k // 128:
                continue

            def run(a, b, scales, output, *partial, name=name):
                module.launch(name, a, b, scales, output, *partial)

            variants[name] = (
                run,
                (x, weights["custom_weight"], weights["custom_scales"], out),
            )
            if name in SPLITS:
                fn, inputs = variants[name]
                partial = torch.empty(
                    (SPLITS[name], n), device="cuda", dtype=torch.float32
                )
                variants[name] = (fn, (*inputs, partial))
        if args.variants:
            variants = {name: variants[name] for name in args.variants}

        for input_kind in ("zero", "one", "random0", "random1", "random2"):
            if input_kind == "zero":
                x.zero_()
            elif input_kind == "one":
                x.fill_(1)
            else:
                torch.manual_seed(int(input_kind[-1]))
                x.normal_()
            reference = (x.float() @ weights["reference"].float()).half()
            for name, (fn, inputs) in variants.items():
                fn(*inputs)
                relative = verify(out, reference)
                correctness.append(
                    {
                        "layer": layer,
                        "input": input_kind,
                        "impl": name,
                        "relative_l2_error": relative,
                    }
                )
        print(f"CORRECT {layer} K={k} N={n}", flush=True)

        if args.check_only:
            continue
        if args.profile:
            for name in args.profile:
                fn, inputs = variants[name]
                measure(fn, inputs, False, 10)
                print("PROFILE", layer, name, flush=True)
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push(f"{layer}_{name}")
                fn(*inputs)
                torch.accelerator.synchronize()
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
        else:
            for cold in (True, False):
                samples = {name: [] for name in variants}
                for round_index in range(2):
                    order = list(variants)
                    if round_index % 2:
                        order.reverse()
                    for name in order:
                        fn, inputs = variants[name]
                        samples[name].extend(
                            measure(fn, inputs, cold, args.repeats).tolist()
                        )
                for name, times in samples.items():
                    us = float(np.median(times))
                    row = {
                        "layer": layer,
                        "M": 1,
                        "K": k,
                        "N": n,
                        "cache": "cold" if cold else "warm",
                        "impl": name,
                        "median_us": us,
                        "round0_us": float(np.median(times[: args.repeats])),
                        "round1_us": float(np.median(times[args.repeats :])),
                        "effective_gbps": (k * n * (0.5 + 2 / 128) + 2 * (k + n))
                        / us
                        / 1000,
                    }
                    rows.append(row)
                    print(row, flush=True)
    (args.output_dir / "correctness.json").write_text(json.dumps(correctness, indent=2))
    if rows:
        with (args.output_dir / "results.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    print("EXPERIMENT_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--layers", nargs="+", choices=LAYERS, default=["qkv_proj", "o_proj"]
    )
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile", nargs="+", choices=["marlin_atomic0", "marlin_atomic1", *VARIANTS]
    )
    parser.add_argument(
        "--variants", nargs="+", choices=["marlin_atomic0", "marlin_atomic1", *VARIANTS]
    )
    main(parser.parse_args())
