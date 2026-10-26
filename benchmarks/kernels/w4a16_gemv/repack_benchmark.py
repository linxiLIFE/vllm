# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure native-layout MMA, offline packing, and GPU-only packing."""

import argparse
import csv
import ctypes
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from batch_benchmark import (
    RAW_CONFIGS,
    TENSOR_CONFIGS,
    WIDE_CONFIGS,
    BatchModule,
    check,
    tensor_pack,
)
from benchmark import DEFAULT_MODEL, LAYERS, checked, load_weights, measure
from cuda.bindings import driver


def cpu_pack_phases(weight, scales):
    """Mirror tensor_pack and time its CPU and transfer stages."""
    n, words = weight.shape
    k = words * 8
    torch.accelerator.synchronize()
    start = time.perf_counter()
    raw = weight.cpu().numpy().view(np.uint32)
    copied = time.perf_counter()
    q = (raw[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15
    q = q.reshape(n // 16, 16, k // 128, 8, 16).transpose(0, 2, 3, 1, 4)
    lane = np.arange(32)
    packed = np.zeros((n // 16, k // 128, 8, 32), dtype=np.uint32)
    for i in range(8):
        reg = i % 4
        row = lane // 4 + reg % 2 * 8
        col = lane % 4 * 2 + reg // 2 * 8 + i // 4
        packed |= q[..., row, col] << (i * 4)
    packed = packed.reshape(n // 16, k // 128, 2, 4, 32)
    packed = packed.transpose(0, 1, 2, 4, 3).copy().reshape(n // 16, k // 128, 8, 32)
    rearranged = time.perf_counter()
    s = scales.reshape(n // 16, 16, k // 128).permute(0, 2, 1).contiguous()
    result = torch.from_numpy(packed.view(np.int32)).cuda()
    torch.accelerator.synchronize()
    end = time.perf_counter()
    return (
        result,
        s,
        {
            "d2h_ms": (copied - start) * 1000,
            "cpu_reorder_ms": (rearranged - copied) * 1000,
            "h2d_and_gpu_scales_ms": (end - rearranged) * 1000,
            "total_ms": (end - start) * 1000,
        },
    )


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    torch.cuda.init()
    torch.empty(0, device="cuda")
    module = BatchModule(args.output_dir / "build")
    pack_fn = checked(
        driver.cuModuleGetFunction(module.tensor_module, b"repack_vector")
    )
    selected = json.loads(args.selection.read_text())
    checks, packing, timing, samples, choices = [], [], [], [], {}
    source = Path(__file__).resolve().parent
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "timing": "FlashInfer CUDA graph/event fallback; two reverse-order rounds",
        "packing_wall": "perf_counter with GPU synchronization; allocation included",
        "gpu_packing_kernel": "preallocated output; excludes allocation/host launch",
        "scope": "native [N,K/8] layout; excludes checkpoint GPTQ/Humming conversion",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.iterdir()
            if p.is_file()
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def save():
        for name, data in (
            ("correctness", checks),
            ("packing", packing),
            ("samples", samples),
            ("selection", choices),
        ):
            (args.output_dir / f"{name}.json").write_text(json.dumps(data, indent=2))
        if timing:
            with (args.output_dir / "results.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=timing[0])
                writer.writeheader()
                writer.writerows(timing)

    for layer in args.layers:
        print(f"LAYER {layer}", flush=True)
        weights = load_weights(args.model, LAYERS[layer], args.layer_index)
        k, n = weights["k"], weights["n"]
        w, s = weights["custom_weight"], weights["custom_scales"]
        pw, ps = tensor_pack(w, s, vector=True)

        def gpu_pack(a, b, c, d, k=k, n=n):
            values = [ctypes.c_void_p(t.data_ptr()) for t in (a, b, c, d)]
            values += [ctypes.c_int(v) for v in (k, n)]
            pointers = (ctypes.c_void_p * len(values))(
                *(ctypes.addressof(v) for v in values)
            )
            checked(
                driver.cuLaunchKernel(
                    pack_fn,
                    (n * (k // 8) + 255) // 256,
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

        gw, gs = torch.empty_like(pw), torch.empty_like(ps)
        gpu_pack(w, s, gw, gs)
        assert torch.equal(gw, pw) and torch.equal(gs, ps)
        cpu_pack_phases(w, s)
        phase_samples, cpu_samples, gpu_samples = [], [], []
        for _ in range(args.pack_repeats):
            rw, rs, phases = cpu_pack_phases(w, s)
            assert torch.equal(rw, pw) and torch.equal(rs, ps)
            phase_samples.append(phases)
            torch.accelerator.synchronize()
            start = time.perf_counter()
            rw, rs = tensor_pack(w, s, vector=True)
            torch.accelerator.synchronize()
            cpu_samples.append((time.perf_counter() - start) * 1000)
            torch.accelerator.synchronize()
            start = time.perf_counter()
            rw, rs = torch.empty_like(pw), torch.empty_like(ps)
            gpu_pack(w, s, rw, rs)
            torch.accelerator.synchronize()
            gpu_samples.append((time.perf_counter() - start) * 1000)
        gpu_kernel = measure(gpu_pack, (w, s, gw, gs), True, args.repeats).tolist()
        packing.append(
            {
                "layer": layer,
                "K": k,
                "N": n,
                "bytes": w.numel() * w.element_size() + s.numel() * s.element_size(),
                "cpu_function_ms": float(np.median(cpu_samples)),
                "gpu_function_ms": float(np.median(gpu_samples)),
                "gpu_kernel_us": float(np.median(gpu_kernel)),
                "phase_medians_ms": {
                    key: float(np.median([r[key] for r in phase_samples]))
                    for key in phase_samples[0]
                },
                "cpu_function_samples_ms": cpu_samples,
                "gpu_function_samples_ms": gpu_samples,
                "cpu_phase_samples_ms": phase_samples,
                "gpu_kernel_samples_us": gpu_kernel,
                "gpu_pack_bitwise_equal": True,
            }
        )
        for m in args.batches:
            key = f"{layer}/M{m}"
            packed_name = selected[key]["custom"]
            nw, kw = TENSOR_CONFIGS[packed_name]
            tiles = WIDE_CONFIGS[packed_name][0] if packed_name in WIDE_CONFIGS else 1
            cache = int(packed_name.startswith("tensorc"))
            matched = f"raw_t{tiles}_n{nw}_k{kw}_c{cache}"
            assert matched in RAW_CONFIGS
            x = torch.empty((m, k), device="cuda", dtype=torch.float16)
            y = torch.empty((m, n), device="cuda", dtype=torch.float16)
            candidates = {}
            for name in [packed_name, *RAW_CONFIGS]:

                def call(a, b, c, d, name=name):
                    module.launch(name, a, b, c, d)

                candidates[name] = (
                    call,
                    (x, w, s, y) if name in RAW_CONFIGS else (x, pw, ps, y),
                )
            for kind in ("zero", "one", "random0", "random1", "random2"):
                if kind == "zero":
                    x.zero_()
                elif kind == "one":
                    x.fill_(1)
                else:
                    torch.manual_seed(int(kind[-1]))
                    x.normal_()
                ref = (x.float() @ weights["reference"].float()).half()
                packed_output = None
                for name, (fn, inputs) in candidates.items():
                    fn(*inputs)
                    error = check(y, ref)
                    assert error["original_elementwise_tolerance_passed"], (key, name)
                    if name == packed_name:
                        packed_output = y.clone()
                    if name == matched:
                        assert torch.equal(y, packed_output), (key, "matched MMA order")
                    checks.append(dict(case=key, impl=name, input=kind, **error))
            if args.check_only:
                save()
                continue
            tuned = {}
            for name, (fn, inputs) in candidates.items():
                if name in RAW_CONFIGS:
                    tuned[name] = float(
                        np.median(measure(fn, inputs, True, args.tune_repeats))
                    )
            best = min(tuned, key=tuned.get)
            choices[key] = dict(
                packed=packed_name,
                raw_matched=matched,
                raw_tuned=best,
                raw_tuning_us=tuned,
            )
            keep = list(dict.fromkeys((packed_name, matched, best)))
            for cold in (True, False):
                rounds = {name: [] for name in keep}
                for order in (keep, list(reversed(keep))):
                    for name in order:
                        fn, inputs = candidates[name]
                        values = measure(fn, inputs, cold, args.repeats).tolist()
                        rounds[name].append(values)
                for name in keep:
                    values = rounds[name][0] + rounds[name][1]
                    timing.append(
                        dict(
                            layer=layer,
                            M=m,
                            K=k,
                            N=n,
                            cache="cold" if cold else "warm",
                            impl=name,
                            median_us=float(np.median(values)),
                            round0_us=float(np.median(rounds[name][0])),
                            round1_us=float(np.median(rounds[name][1])),
                        )
                    )
                    samples.append(
                        dict(case=key, impl=name, cold=cold, rounds=rounds[name])
                    )
            save()
            print(f"DONE {key}: {choices[key]}", flush=True)
    save()
    print("REPACK_EXPERIMENT_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", choices=LAYERS, default=list(LAYERS))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=150)
    parser.add_argument("--tune-repeats", type=int, default=30)
    parser.add_argument("--pack-repeats", type=int, default=7)
    parser.add_argument("--check-only", action="store_true")
    main(parser.parse_args())
