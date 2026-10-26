# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tune and compare CUDA W4A16 row reuse with Marlin and pinned Humming."""

import argparse
import csv
import ctypes
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from benchmark import DEFAULT_MODEL, LAYERS, GemvModule, checked, load_weights, measure
from cuda.bindings import driver

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.scalar_type import scalar_types

BATCH_CONFIGS = {
    f"batch_r{r}_o{o}_w{w}_t32": (r, o, w, 32)
    for r in (1, 2, 4, 8)
    for o in (1, 2)
    for w in (4, 8)
}
BATCH_CONFIGS["batch_r1_o4_w4_t32"] = (1, 4, 4, 32)
for r in (2, 4, 8):
    BATCH_CONFIGS[f"batch_r{r}_o1_w4_t16"] = (r, 1, 4, 16)
TENSOR_CONFIGS = {
    f"tensor_n{nw}_k{kw}": (nw, kw)
    for nw, kw in (
        (1, 1),
        (2, 1),
        (4, 1),
        (1, 2),
        (2, 2),
        (4, 2),
        (1, 4),
        (2, 4),
        (1, 8),
    )
}
for nw, kw in ((1, 2), (1, 4), (2, 4), (1, 8), (1, 16), (1, 32)):
    TENSOR_CONFIGS[f"tensorv_n{nw}_k{kw}"] = (nw, kw)
WIDE_CONFIGS = {
    f"tensorw{t}_n{nw}_k{kw}": (t, nw, kw)
    for t, nw, kw in (
        (2, 1, 4),
        (2, 1, 8),
        (2, 2, 4),
        (2, 1, 16),
        (4, 1, 8),
        (4, 1, 16),
    )
}
for tiles, kw in ((1, 8), (2, 8), (2, 16)):
    WIDE_CONFIGS[f"tensorc{tiles}_n1_k{kw}"] = (tiles, 1, kw)
TENSOR_CONFIGS.update({name: (nw, kw) for name, (_, nw, kw) in WIDE_CONFIGS.items()})
RAW_CONFIGS = {
    f"raw_t{t}_n{nw}_k{kw}_c{cache}": (t, nw, kw, cache)
    for t, nw, kw, cache in (
        (1, 1, 2, 0),
        (1, 1, 4, 0),
        (1, 1, 8, 0),
        (1, 1, 16, 0),
        (1, 1, 32, 0),
        (1, 2, 2, 0),
        (1, 2, 4, 0),
        (1, 4, 2, 0),
        (2, 1, 4, 0),
        (2, 1, 8, 0),
        (2, 1, 16, 0),
        (2, 2, 4, 0),
        (4, 1, 8, 0),
        (4, 1, 16, 0),
        (2, 1, 8, 1),
        (2, 1, 16, 1),
    )
}
COMPUTE = json.dumps(
    {"use_f16_accum": False, "use_batch_invariant": False, "gemm_type": "dense"}
)


class BatchModule:
    def __init__(self, build_dir):
        self.single = GemvModule(build_dir)
        self.module = checked(
            driver.cuModuleLoad(str(build_dir / "batched.cubin").encode())
        )
        self.functions = {
            name: checked(driver.cuModuleGetFunction(self.module, name.encode()))
            for name in BATCH_CONFIGS
        }
        self.tensor_module = checked(
            driver.cuModuleLoad(str(build_dir / "tensorcore.cubin").encode())
        )
        self.functions.update(
            {
                name: checked(
                    driver.cuModuleGetFunction(self.tensor_module, name.encode())
                )
                for name in TENSOR_CONFIGS.keys() | RAW_CONFIGS.keys()
            }
        )

    def launch(self, name, x, weight, scales, out):
        m, k = x.shape
        n = out.shape[1]
        assert k % 128 == 0 and x.is_contiguous() and out.is_contiguous()
        assert x.dtype == out.dtype == scales.dtype == torch.float16
        assert weight.dtype == torch.int32
        if name in RAW_CONFIGS:
            tiles, nw, kw, _ = RAW_CONFIGS[name]
            rows, columns, warps = 8, nw * 16 * tiles, nw * kw
            assert 1 <= m <= 8 and n % columns == 0
            assert weight.shape == (n, k // 8)
            assert scales.shape == (n, k // 128)
        elif name in TENSOR_CONFIGS:
            nw, kw = TENSOR_CONFIGS[name]
            tiles = WIDE_CONFIGS[name][0] if name in WIDE_CONFIGS else 1
            rows, columns, warps = 8, nw * 16 * tiles, nw * kw
            assert 1 <= m <= 8 and n % columns == 0
            assert weight.shape == (n // 16, k // 128, 8, 32)
            assert scales.shape == (n // 16, k // 128, 16)
        else:
            rows, outputs, warps, threads = BATCH_CONFIGS[name]
            columns = warps * 32 // threads * outputs
            assert weight.shape == (n, k // 8)
            assert scales.shape == (n, k // 128)
        args = [ctypes.c_void_p(t.data_ptr()) for t in (x, weight, scales, out)]
        args += [ctypes.c_int(v) for v in (m, k, n)]
        pointers = (ctypes.c_void_p * len(args))(
            *(ctypes.addressof(arg) for arg in args)
        )
        checked(
            driver.cuLaunchKernel(
                self.functions[name],
                (n + columns - 1) // columns,
                (m + rows - 1) // rows,
                1,
                warps * 32,
                1,
                1,
                0,
                torch.cuda.current_stream().cuda_stream,
                ctypes.addressof(pointers),
                0,
            )
        )


def tensor_pack(weight, scales, vector=False):
    """Pack INT4 weights into the A fragments of W @ X.T m16n8k16."""
    n, words = weight.shape
    k = words * 8
    raw = weight.cpu().numpy().view(np.uint32)
    q = (raw[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15
    q = q.reshape(n // 16, 16, k // 128, 8, 16).transpose(0, 2, 3, 1, 4)
    lane = np.arange(32)
    packed = np.zeros((n // 16, k // 128, 8, 32), dtype=np.uint32)
    for i in range(8):
        reg = i % 4
        row = lane // 4 + reg % 2 * 8
        col = lane % 4 * 2 + reg // 2 * 8 + i // 4
        packed |= q[..., row, col] << (i * 4)
    if vector:
        packed = packed.reshape(n // 16, k // 128, 2, 4, 32)
        packed = (
            packed.transpose(0, 1, 2, 4, 3).copy().reshape(n // 16, k // 128, 8, 32)
        )
    s = scales.reshape(n // 16, 16, k // 128).permute(0, 2, 1).contiguous()
    return torch.from_numpy(packed.view(np.int32)).cuda(), s


def check(actual, reference):
    difference = actual.float() - reference.float()
    relative = (difference.norm() / reference.float().norm().clamp_min(1e-20)).item()
    assert relative < 0.001, relative
    rms = reference.float().square().mean().sqrt().item()
    max_abs = difference.abs().max().item()
    scaled_atol = max(0.002, 0.002 * rms)
    torch.testing.assert_close(
        actual.float(), reference.float(), atol=scaled_atol, rtol=0.02
    )
    strict = (
        torch.isclose(actual.float(), reference.float(), atol=0.002, rtol=0.02)
        .all()
        .item()
    )
    return dict(
        relative_l2_error=relative,
        max_abs_error=max_abs,
        output_rms=rms,
        max_error_over_rms=max_abs / max(rms, 1e-20),
        elementwise_atol=scaled_atol,
        original_elementwise_tolerance_passed=strict,
    )


def main(args):
    sys.path.insert(0, str(args.humming_root.resolve()))
    import humming
    from humming.tune import get_heuristics_config

    from vllm.utils import humming as hm

    assert Path(humming.__file__).resolve().is_relative_to(args.humming_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.init()
    torch.empty(0, device="cuda")
    module = BatchModule(args.output_dir / "build")
    marlin_locks = marlin_make_workspace_new(
        torch.device("cuda:0"), max_blocks_per_sm=4
    )
    humming_locks = hm.create_humming_locks(torch.device("cuda:0"))
    source = Path(__file__).resolve().parent
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "humming_version": version("humming-kernels"),
        "humming_import": humming.__file__,
        "humming_commit": subprocess.check_output(
            ["git", "-C", str(args.humming_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "vllm_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": str(args.model),
        "layer_index": args.layer_index,
        "phase": args.phase,
        "selection_file": str(args.selection) if args.selection else None,
        "wsl_build_environment": "LIBRARY_PATH=/usr/lib/wsl/lib",
        "batches": args.batches,
        "tensorcore": args.tensorcore,
        "variants": args.variants,
        "layers": args.layers,
        "repeats": args.repeats,
        "dtype": "W4A16 group128, FP16 dequantized weights/output, FP32 accumulation",
        "correctness": (
            "relative L2 < 0.001; elementwise atol=max(0.002, 0.002*output RMS), "
            "rtol=0.02; "
            "also record original atol=0.002 rtol=0.02 elementwise pass flag"
        ),
        "reduction_precision": (
            "custom FP32; Marlin atomic1 uses FP16 global atomic reduction; "
            "library settings disable FP16 MMA accumulation"
        ),
        "timing": (
            "GPU graph replay; two reverse-order rounds; "
            "cold rotated args/warm reused buffers"
        ),
        "excluded": "packing, compilation, allocation, host dispatch",
        "dispatch": (
            "cold-cache tuning selects one config per family/shape "
            "for both cache conditions"
        ),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.iterdir()
            if p.is_file()
        },
    }
    try:
        metadata["cupti"] = version("cupti-python")
    except Exception:
        metadata["cupti"] = "absent; FlashInfer CUDA graph/event fallback"
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    selected = json.loads(args.selection.read_text()) if args.selection else {}
    selections, results, checks, failures, configs, raw_samples = {}, [], [], [], {}, []
    if args.resume:
        assert args.phase == "tune"
        selections = json.loads((args.output_dir / "selection.json").read_text())
        with (args.output_dir / "results.csv").open() as f:
            results = list(csv.DictReader(f))
        for row in results:
            row["M"] = int(row["M"])
        checks = [
            c
            for c in json.loads((args.output_dir / "correctness.json").read_text())
            if f"{c['layer']}/M{c['M']}" in selections
        ]
        failures = json.loads((args.output_dir / "failures.json").read_text())
        configs = json.loads((args.output_dir / "humming_configs.json").read_text())
        raw_samples = json.loads((args.output_dir / "samples_us.json").read_text())

    def save():
        for name, value in (
            ("correctness", checks),
            ("failures", failures),
            ("humming_configs", configs),
            ("samples_us", raw_samples),
            ("selection", selections),
        ):
            (args.output_dir / f"{name}.json").write_text(json.dumps(value, indent=2))
        if results:
            with (args.output_dir / "results.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=results[0])
                writer.writeheader()
                writer.writerows(results)

    for layer in args.layers:
        weights = load_weights(args.model, LAYERS[layer], args.layer_index)
        k, n = weights["k"], weights["n"]
        if args.tensorcore:
            tensor_weight, tensor_scales = tensor_pack(
                weights["custom_weight"], weights["custom_scales"]
            )
            vector_weight, _ = tensor_pack(
                weights["custom_weight"], weights["custom_scales"], vector=True
            )
        hc, ht = hm.prepare_and_transform_humming_tensors(
            tensors={
                "weight": weights["custom_weight"],
                "weight_scale": weights["custom_scales"],
            },
            shape_n=n,
            shape_k=k,
            weight_schema=hm.HummingWeightSchema(
                b_dtype="int4", weight_scale_group_size=128
            ),
            input_schema=hm.HummingInputSchema(a_dtype="float16"),
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            torch_dtype=torch.float16,
            device=torch.device("cuda:0"),
        )
        assert hc.b_dtype.num_bits == 4 and not hc.has_zero_point
        assert hc.a_dtype.num_bits == 16 and not hc.input_quant_mode.should_quantize
        for m in args.batches:
            key = f"{layer}/M{m}"
            if args.resume and key in selections:
                print(f"RESUME_SKIP {key}", flush=True)
                continue
            print(f"START {key}", flush=True)
            x = torch.empty((m, k), dtype=torch.float16, device="cuda")
            out = torch.empty((m, n), dtype=torch.float16, device="cuda")

            def marlin(a, b, s, y, locks, atomic=False, m=m, k=k, n=n):
                return ops.marlin_gemm(
                    a,
                    y,
                    b,
                    None,
                    s,
                    None,
                    None,
                    None,
                    locks,
                    scalar_types.uint4b8,
                    m,
                    n,
                    k,
                    atomic,
                    True,
                    False,
                )

            def humming_run(a, b, s, y, locks, tuning, hc=hc):
                return hm.humming_forward(
                    hc,
                    inputs=a,
                    weight=b,
                    weight_scale=s,
                    outputs=y,
                    locks=locks,
                    compute_config=COMPUTE,
                    tuning_config=tuning,
                )

            candidates = {}
            for atomic in (False, True):
                candidates[f"marlin_atomic{int(atomic)}"] = (
                    "marlin",
                    marlin,
                    (
                        x,
                        weights["marlin_weight"],
                        weights["marlin_scales"],
                        out,
                        marlin_locks,
                        atomic,
                    ),
                )
            default = get_heuristics_config(hc, m, use_f16_accum=False)
            tuning = {"humming_default": default}
            for stages in (2, 3):
                if default["num_stages"] != stages:
                    tuning[f"humming_stages{stages}"] = dict(default, num_stages=stages)
            if default.get("use_stream_k"):
                tuning["humming_no_streamk"] = dict(default, use_stream_k=False)
            configs[key] = tuning
            for name, config in tuning.items():
                candidates[name] = (
                    "humming",
                    humming_run,
                    (
                        x,
                        ht["weight"],
                        ht["weight_scale"],
                        out,
                        humming_locks,
                        json.dumps(config),
                    ),
                )
            for name, config in BATCH_CONFIGS.items():
                if config[0] > m:
                    continue

                def batch(a, b, s, y, name=name):
                    module.launch(name, a, b, s, y)

                candidates[name] = (
                    "custom",
                    batch,
                    (x, weights["custom_weight"], weights["custom_scales"], out),
                )
            if m == 1:
                original_name = (
                    "gemv_spread_w4_o4"
                    if layer == "gate_up_proj"
                    else "gemv_spread_t32"
                )

                def original(a, b, s, y, name=original_name):
                    module.single.launch(name, a.view(-1), b, s, y.view(-1))

                candidates[original_name] = (
                    "custom",
                    original,
                    (x, weights["custom_weight"], weights["custom_scales"], out),
                )
            if args.tensorcore and m <= 8:
                for name in TENSOR_CONFIGS:

                    def tensor(a, b, s, y, name=name):
                        module.launch(name, a, b, s, y)

                    candidates[name] = (
                        "custom",
                        tensor,
                        (
                            x,
                            vector_weight
                            if name.startswith(("tensorv_", "tensorw", "tensorc"))
                            else tensor_weight,
                            tensor_scales,
                            out,
                        ),
                    )
            if args.variants:
                candidates = {
                    name: value
                    for name, value in candidates.items()
                    if name in args.variants or value[0] != "custom"
                }
            if selected:
                keep = set(selected[key].values()) | {
                    "marlin_atomic0",
                    "humming_default",
                }
                candidates = {
                    name: value for name, value in candidates.items() if name in keep
                }
            valid = {}
            eligible = {}
            for name, (family, fn, inputs) in candidates.items():
                try:
                    strict_pass = True
                    for kind in ("zero", "one", "random0", "random1", "random2"):
                        if kind == "zero":
                            x.zero_()
                        elif kind == "one":
                            x.fill_(1)
                        else:
                            torch.manual_seed(int(kind[-1]))
                            x.normal_()
                        ref = (x.float() @ weights["reference"].float()).half()
                        fn(*inputs)
                        error = check(out, ref)
                        strict_pass &= error["original_elementwise_tolerance_passed"]
                        checks.append(
                            dict(
                                layer=layer,
                                M=m,
                                impl=name,
                                input=kind,
                                **error,
                            )
                        )
                    valid[name] = (family, fn, inputs)
                    eligible[name] = strict_pass
                    print(f"CORRECT {key} {name}", flush=True)
                except Exception as exc:
                    failures.append(dict(case=key, impl=name, error=repr(exc)))
                    save()
                    if family == "custom" or name == "marlin_atomic0":
                        raise
                    print(f"SKIP {key} {name}: {exc}", flush=True)
            for family in ("custom", "marlin", "humming"):
                assert any(
                    v[0] == family and eligible[name] for name, v in valid.items()
                ), (key, family)
            if selected:
                assert set(selected[key].values()).issubset(valid), key
                assert all(eligible[name] for name in selected[key].values()), key
            if args.phase == "check":
                save()
                continue
            if args.phase == "profile":
                assert selected
                for name in selected[key].values():
                    _, fn, inputs = valid[name]
                    measure(fn, inputs, False, 10)
                    print(f"PROFILE {key} {name}", flush=True)
                    torch.cuda.profiler.start()
                    fn(*inputs)
                    torch.accelerator.synchronize()
                    torch.cuda.profiler.stop()
                save()
                continue
            medians = {}
            for cold in (True,) if args.phase == "tune" else (True, False):
                samples = {name: [] for name in valid}
                rounds = {name: [] for name in valid}
                for round_id in range(2):
                    order = list(valid)
                    if round_id:
                        order.reverse()
                    for name in order:
                        _, fn, inputs = valid[name]
                        times = measure(fn, inputs, cold, args.repeats).tolist()
                        samples[name].extend(times)
                        rounds[name].append(float(np.median(times)))
                for name, times in samples.items():
                    family = valid[name][0]
                    us = float(np.median(times))
                    nb = k * n * (0.5 + 2 / 128) + 2 * m * (k + n)
                    results.append(
                        dict(
                            layer=layer,
                            M=m,
                            K=k,
                            N=n,
                            cache="cold" if cold else "warm",
                            family=family,
                            impl=name,
                            median_us=us,
                            round0_us=rounds[name][0],
                            round1_us=rounds[name][1],
                            us_per_token=us / m,
                            effective_gbps=nb / us / 1000,
                            tflops=2 * m * k * n / us / 1e6,
                        )
                    )
                    raw_samples.append(
                        dict(case=key, impl=name, cold=cold, samples=times)
                    )
                    if cold:
                        medians[name] = us
                print(f"TIMED {key} {'cold' if cold else 'warm'}", flush=True)
            selections[key] = (
                selected[key]
                if selected
                else {
                    family: min(
                        (
                            name
                            for name in valid
                            if valid[name][0] == family and eligible[name]
                        ),
                        key=lambda name: medians[name],
                    )
                    for family in ("custom", "marlin", "humming")
                }
            )
            print(f"BEST {key}: {selections[key]}", flush=True)
            save()
    save()
    print("BATCH_EXPERIMENT_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--humming-root", type=Path, required=True)
    parser.add_argument("--humming-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("tune", "compare", "check", "profile"), default="tune"
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--tensorcore", action="store_true")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--layers", nargs="+", choices=LAYERS, default=list(LAYERS))
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=40)
    main(parser.parse_args())
