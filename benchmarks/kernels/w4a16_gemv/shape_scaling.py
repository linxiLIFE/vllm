# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Synthetic W4A16 projection sweep; no checkpoint weights are downloaded."""

import argparse
import csv
import ctypes
import dataclasses
import hashlib
import json
import subprocess
import sys
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from batch_benchmark import COMPUTE, check
from benchmark import checked
from cuda.bindings import driver
from flashinfer.testing import bench_gpu_time_with_cupti
from shared_layout_benchmark import CONFIGS, ROW_CAPACITY

# (K, N), TP=1. wo_a is eight independent group GEMMs, not one dense GEMM.
SHAPES = {
    "qwen_qkv": (1536, 2048),
    "qwen_o": (1536, 1536),
    "qwen_gate_up": (1536, 17920),
    "qwen_down": (8960, 1536),
    "v41_q_a": (5120, 1280),
    "v41_q_b": (1280, 32768),
    "v41_kv": (5120, 512),
    "v41_o_a_group": (4096, 1024),
    "v41_o_b": (8192, 5120),
    "v41_expert_gate_up": (5120, 4608),
    "v41_expert_down": (2304, 5120),
    "v41_index_q_b": (1280, 4096),
}
VARIANTS = (
    "shareds4_t2_n1_k8_c0_rows",
    "shareds4_t2_n1_k16_c0_rows",
    "shareds4_t2_n2_k8_c0_rows",
    "sharedv_t2_n1_k16_c0_rows",
    "reuse2_t2_k4_s4",
)
MODEL_REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
WIDE_CONFIGS = {
    f"wide_r{rows}_t{tiles}_k{warps}_s{steps}": (tiles, 1, warps, 0)
    for rows in (1, 2, 4, 8)
    for tiles in (2, 4)
    for warps in (1, 2, 4)
    for steps in (4, 8)
    if rows * tiles * warps <= 96
}
CONFIGS.update(WIDE_CONFIGS)
ROW_CAPACITY.update({name: int(name.split("_")[1][1:]) * 8 for name in WIDE_CONFIGS})
for _rows, _tiles, _warps, _steps in (
    (4, 4, 1, 8),
    (4, 4, 4, 8),
):
    _name = f"prefetch_r{_rows}_t{_tiles}_k{_warps}_s{_steps}"
    WIDE_CONFIGS[_name] = CONFIGS[_name] = (_tiles, 1, _warps, 0)
    ROW_CAPACITY[_name] = _rows * 8
for _rows, _tiles, _warps in (
    (2, 4, 1),
    (2, 4, 2),
    (4, 4, 1),
    (4, 4, 2),
    (4, 4, 4),
    (8, 4, 1),
    (8, 4, 2),
    (8, 2, 2),
):
    for _steps in (1, 2):
        _name = f"wide_r{_rows}_t{_tiles}_k{_warps}_s{_steps}"
        WIDE_CONFIGS[_name] = CONFIGS[_name] = (_tiles, 1, _warps, 0)
        ROW_CAPACITY[_name] = _rows * 8


class FusedModule:
    def __init__(self, output, build=None):
        source = Path(__file__).resolve().parent
        build = build or output / "build"
        cubin = build / "wide_gemm.cubin"
        assert torch.cuda.get_device_capability() == (8, 6)
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-G",
                "Ninja",
                "-DGEMV_ARCH=86",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--target", "wide_gemm"], check=True
        )
        self.module = checked(driver.cuModuleLoad(str(cubin).encode()))
        self.functions = {
            name: checked(
                driver.cuModuleGetFunction(self.module, ("model_" + name).encode())
            )
            for name in (*VARIANTS, *WIDE_CONFIGS)
        }

    def launch(self, name, x, w, s, out, bias):
        m, k = x.shape
        n = out.shape[1]
        tiles, nw, kw, _ = CONFIGS[name]
        assert k % 128 == 0 and n % (tiles * nw * 16) == 0
        assert w.shape == (k // 16, n * 2) and s.shape == (k // 128, n)
        assert all(t.is_contiguous() for t in (x, w, s, out))
        values = [
            ctypes.c_void_p(t.data_ptr() if t is not None else 0)
            for t in (x, w, s, out, bias)
        ]
        values += [ctypes.c_int(v) for v in (m, k, n)]
        pointers = (ctypes.c_void_p * len(values))(
            *(ctypes.addressof(v) for v in values)
        )
        rows = ROW_CAPACITY.get(name, 8)
        checked(
            driver.cuLaunchKernel(
                self.functions[name],
                n // (tiles * nw * 16),
                (m + rows - 1) // rows,
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


def reference(x, packed, scales):
    """FP32 GEMM of independently decoded FP16 weights, in small N chunks."""
    n, words = packed.shape
    result = torch.empty((x.shape[0], n), device="cuda", dtype=torch.float16)
    shifts = torch.arange(8, device="cuda", dtype=torch.int32) * 4
    for first in range(0, n, 256):
        codes = ((packed[first : first + 256, :, None] >> shifts) & 15) - 8
        dense = codes.reshape(-1, words * 8).half()
        dense *= scales[first : first + 256].repeat_interleave(128, dim=1)
        result[:, first : first + 256] = x.float() @ dense.float().T
    return result


def separate_bias(module, name, x, w, s, out, bias):
    module.launch(name, x, w, s, out, None)
    out.add_(bias)


def measure(fn, cold, repeats, warmup):
    fn()
    torch.accelerator.synchronize()
    return (
        np.asarray(
            bench_gpu_time_with_cupti(
                fn,
                use_cuda_graph=True,
                cold_l2_cache=cold,
                dry_run_time_ms=warmup,
                repeat_iters=repeats,
            )
        )
        * 1000
    ).tolist()


def main(args):
    sys.path.insert(0, str(args.cupti_root.resolve()))
    from cupti import cupti

    assert cupti is not None
    sys.path.insert(0, str(args.humming_root.resolve()))
    import humming
    from humming.tune import get_heuristics_config

    from vllm.utils import humming as hm

    assert Path(humming.__file__).resolve().is_relative_to(args.humming_root.resolve())
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.init()
    torch.empty(0, device="cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    source_files = (
        "shape_scaling.py",
        "model_dispatch.cu",
        "shared_layout.cu",
        "shared_layout_benchmark.py",
        "batch_benchmark.py",
        "benchmark.py",
        "wide_gemm.cu",
        "CMakeLists.txt",
    )
    snapshot = args.output_dir / "source"
    snapshot.mkdir(exist_ok=True)
    for name in source_files:
        (snapshot / name).write_bytes((source / name).read_bytes())
    metadata = {
        "command": sys.argv,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "humming_commit": subprocess.check_output(
            ["git", "-C", str(args.humming_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "shapes": SHAPES,
        "shape_source": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/"
        f"blob/{MODEL_REVISION}/inference/model.py",
        "shape_revision": MODEL_REVISION,
        "seed": args.seed,
        "rounds": args.rounds,
        "repeats": args.repeats,
        "warmup_ms": args.warmup_ms,
        "cupti_version": version("cupti-python"),
        "selection_source": str(args.selection) if args.selection else None,
        "previous_selection_source": str(args.previous_selection)
        if args.previous_selection
        else None,
        "variants": args.variants or list((*VARIANTS, *WIDE_CONFIGS)),
        "timing": "CUPTI graph replay; cold flush excluded; reverse order rounds",
        "precision": "synthetic symmetric INT4 group128, FP16 dequant, FP32 accum",
        "limitations": "TP1 individual GEMMs; no routing/communication/attention; "
        "V4.1 native FP8/FP4 and bias-free projections not reproduced",
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in source_files
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    module = FusedModule(args.output_dir, args.build_dir)
    locks = hm.create_humming_locks("cuda:0")
    selection = json.loads(args.selection.read_text()) if args.selection else {}
    previous = (
        json.loads(args.previous_selection.read_text())
        if args.previous_selection
        else {}
    )
    chosen, records, validations, samples = {}, [], [], []

    def save():
        for name, obj in (
            ("selection", chosen),
            ("checks", validations),
            ("samples", samples),
        ):
            (args.output_dir / f"{name}.json").write_text(json.dumps(obj, indent=2))
        if records:
            with (args.output_dir / "summary.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=records[0])
                writer.writeheader()
                writer.writerows(records)

    for shape in args.shapes:
        k, n = SHAPES[shape]
        packed = torch.randint(
            -(2**31), 2**31 - 1, (n, k // 8), device="cuda", dtype=torch.int32
        )
        scales = (torch.rand((n, k // 128), device="cuda") * 0.02 + 0.01).half()
        plain_bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.1
        hc, ht = hm.prepare_and_transform_humming_tensors(
            tensors={"weight": packed, "weight_scale": scales, "bias": plain_bias},
            shape_n=n,
            shape_k=k,
            has_bias=True,
            weight_schema=hm.HummingWeightSchema(
                b_dtype="int4", weight_scale_group_size=128
            ),
            input_schema=hm.HummingInputSchema(a_dtype="float16"),
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            torch_dtype=torch.float16,
            device=torch.device("cuda:0"),
        )
        assert hc.shape_n == n and hc.shape_k == k and not hc.has_zero_point
        assert not any(
            (
                hc.use_packed_k_layout,
                hc.use_native_dequant,
                hc.use_flat_weight,
                hc.use_raw_weight,
                hc.use_int_weight_scale,
                hc.should_apply_bs_on_c,
            )
        )
        w, s, transformed_bias = ht["weight"], ht["weight_scale"], ht["bias"]
        fingerprint = (w.data_ptr(), s.data_ptr(), w._version, s._version)
        for m in args.batches:
            x = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
            out = torch.empty((m, n), device="cuda", dtype=torch.float16)
            ref = reference(x, packed, scales)
            default = get_heuristics_config(hc, m, use_f16_accum=False)
            configs = {"humming_default": default}
            if default["num_stages"] != 2:
                configs["humming_stages2"] = dict(default, num_stages=2)
            if default.get("use_stream_k"):
                configs["humming_no_streamk"] = dict(default, use_stream_k=False)
            for bias_mode in args.bias_modes:
                bias = transformed_bias if bias_mode != "none" else None
                forward_config = dataclasses.replace(hc, has_bias=bias is not None)
                expected = ref + plain_bias if bias is not None else ref
                candidates = {}
                for name in (*VARIANTS, *WIDE_CONFIGS):
                    candidates[name] = (
                        partial(separate_bias, module, name, x, w, s, out, plain_bias)
                        if bias_mode == "separate"
                        else partial(module.launch, name, x, w, s, out, bias)
                    )
                for name, config in configs.items():
                    candidates[name] = partial(
                        hm.humming_forward,
                        forward_config,
                        inputs=x,
                        weight=w,
                        weight_scale=s,
                        bias=bias,
                        outputs=out,
                        locks=locks,
                        compute_config=COMPUTE,
                        tuning_config=json.dumps(config),
                    )
                for cold in (True, False):
                    key = f"{shape}/M{m}/{bias_mode}/{'cold' if cold else 'warm'}"
                    active = candidates
                    if selection:
                        active = {
                            name: candidates[name] for name in selection[key].values()
                        }
                        if previous:
                            name = previous[key]["custom"]
                            active[name] = candidates[name]
                    elif args.variants:
                        active = {
                            name: fn
                            for name, fn in candidates.items()
                            if name.startswith("humming") or name in args.variants
                        }
                    for name, fn in active.items():
                        fn()
                        validations.append(
                            dict(case=key, impl=name, **check(out, expected))
                        )
                    if args.profile_only:
                        save()
                        continue
                    timings = {name: [] for name in active}
                    for round_id in range(args.rounds):
                        order = list(active)
                        if round_id % 2:
                            order.reverse()
                        for name in order:
                            values = measure(
                                active[name], cold, args.repeats, args.warmup_ms
                            )
                            timings[name].append(float(np.median(values)))
                            samples.append(
                                dict(case=key, impl=name, round=round_id, us=values)
                            )
                    medians = {
                        name: float(np.median(values))
                        for name, values in timings.items()
                    }
                    winners = {
                        family: min(
                            (
                                name
                                for name in active
                                if name.startswith("humming") == (family == "humming")
                            ),
                            key=medians.get,
                        )
                        for family in ("custom", "humming")
                    }
                    if selection:
                        winners = selection[key]
                    chosen[key] = winners
                    a, b = (medians[winners[f]] for f in ("custom", "humming"))
                    traffic = k * n // 2 + k // 128 * n * 2 + 2 * (m * k + m * n)
                    if bias is not None:
                        traffic += n * 2
                    records.append(
                        dict(
                            shape=shape,
                            m=m,
                            k=k,
                            n=n,
                            bias=bias_mode,
                            cold=cold,
                            custom=winners["custom"],
                            humming=winners["humming"],
                            humming_config=json.dumps(configs[winners["humming"]]),
                            custom_us=a,
                            humming_us=b,
                            speedup=b / a,
                            custom_gbps=traffic / a / 1000,
                            humming_gbps=traffic / b / 1000,
                            custom_tflops=2 * m * n * k / a / 1e6,
                            custom_round_us=json.dumps(timings[winners["custom"]]),
                            humming_round_us=json.dumps(timings[winners["humming"]]),
                            previous_custom=previous[key]["custom"] if previous else "",
                            previous_us=medians[previous[key]["custom"]]
                            if previous
                            else "",
                        )
                    )
                    print(f"{key}: {a:.2f} / {b:.2f} us, {b / a:.3f}x", flush=True)
                    assert fingerprint == (
                        w.data_ptr(),
                        s.data_ptr(),
                        w._version,
                        s._version,
                    )
                    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humming-root", type=Path, required=True)
    parser.add_argument("--cupti-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--previous-selection", type=Path)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=(*VARIANTS, *WIDE_CONFIGS))
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    parser.add_argument(
        "--bias-modes",
        nargs="+",
        choices=("none", "fused", "separate"),
        default=["none", "fused"],
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
