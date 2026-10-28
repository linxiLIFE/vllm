# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare kernels sharing exactly one Humming W4A16 parameter allocation."""

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
from batch_benchmark import COMPUTE, check
from benchmark import DEFAULT_MODEL, LAYERS, checked, load_weights
from cuda.bindings import driver
from flashinfer.testing import bench_gpu_time_with_cupti

CONFIGS = {
    f"shared_t{t}_n{nw}_k{kw}_c{cache}": (t, nw, kw, cache)
    for t, nw, kw, cache in (
        (1, 1, 2, 0),
        (1, 1, 4, 0),
        (1, 1, 8, 0),
        (1, 1, 16, 0),
        (1, 2, 4, 0),
        (1, 2, 8, 0),
        (2, 1, 4, 0),
        (2, 1, 8, 0),
        (2, 1, 16, 0),
        (2, 2, 4, 0),
        (2, 2, 8, 0),
        (4, 1, 4, 0),
        (4, 1, 8, 0),
        (4, 1, 16, 0),
        (2, 1, 8, 1),
        (4, 1, 8, 1),
        (4, 1, 16, 1),
    )
}
for t, nw, kw, cache in (
    (1, 1, 8, 0),
    (1, 2, 4, 0),
    (2, 1, 4, 0),
    (2, 1, 8, 0),
    (2, 1, 16, 0),
    (2, 2, 4, 0),
    (2, 2, 8, 0),
    (4, 1, 4, 0),
    (4, 1, 8, 0),
    (4, 1, 16, 0),
    (2, 1, 8, 1),
    (4, 1, 8, 1),
    (4, 1, 16, 1),
):
    CONFIGS[f"sharedv_t{t}_n{nw}_k{kw}_c{cache}"] = (t, nw, kw, cache)

for steps in (1, 2, 4):
    for t, nw, kw in ((2, 1, 8), (2, 1, 16), (2, 2, 8), (4, 1, 8), (4, 1, 16)):
        CONFIGS[f"shareds{steps}_t{t}_n{nw}_k{kw}_c0"] = (t, nw, kw, 0)
for base in (
    "sharedv_t2_n1_k16_c0",
    "shareds4_t2_n1_k8_c0",
    "shareds4_t2_n2_k8_c0",
    "shareds4_t2_n1_k16_c0",
):
    CONFIGS[base + "_rows"] = CONFIGS[base]

ROW_CAPACITY = {}
for row_tiles, kw, steps in (
    (2, 4, 4),
    (2, 4, 8),
    (2, 8, 4),
    (2, 16, 8),
    (2, 8, 8),
    (4, 8, 4),
    (4, 8, 8),
    (4, 4, 8),
    (4, 4, 4),
):
    name = f"reuse{row_tiles}_t2_k{kw}_s{steps}"
    CONFIGS[name] = (2, 1, kw, 0)
    ROW_CAPACITY[name] = row_tiles * 8
for kw, steps in ((8, 4), (16, 8)):
    CONFIGS[f"staged_t2_k{kw}_s{steps}"] = (2, 1, kw, 0)
    CONFIGS[f"pipelined_t2_k{kw}_s{steps}"] = (2, 1, kw, 0)

RESERVE_CONTROLS = {
    "reserved_t2_k8_s4": "shareds4_t2_n1_k8_c0",
    "reserved_t2_k16_s8": "sharedv_t2_n1_k16_c0",
}
for name, base in RESERVE_CONTROLS.items():
    CONFIGS[name] = CONFIGS[base]


class SharedModule:
    def __init__(self, build):
        source = Path(__file__).resolve().parent
        capability = torch.cuda.get_device_capability()
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-G",
                "Ninja",
                f"-DGEMV_ARCH={capability[0]}{capability[1]}",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--target", "shared_layout"], check=True
        )
        self.module = checked(
            driver.cuModuleLoad(str(build / "shared_layout.cubin").encode())
        )
        self.functions = {
            name: checked(
                driver.cuModuleGetFunction(
                    self.module, RESERVE_CONTROLS.get(name, name).encode()
                )
            )
            for name in CONFIGS
        }

    def call(self, name, tensors, integers, blocks, threads, grid_y=1, reserve=0):
        values = [ctypes.c_void_p(t.data_ptr()) for t in tensors]
        values += [ctypes.c_int(v) for v in integers]
        pointers = (ctypes.c_void_p * len(values))(
            *(ctypes.addressof(v) for v in values)
        )
        checked(
            driver.cuLaunchKernel(
                self.functions[name],
                blocks,
                grid_y,
                1,
                threads,
                1,
                1,
                reserve,
                torch.cuda.current_stream().cuda_stream,
                ctypes.addressof(pointers),
                0,
            )
        )

    def launch(self, name, x, weight, scales, out):
        tiles, nw, kw, _ = CONFIGS[name]
        m, k = x.shape
        n = out.shape[1]
        row_tiled = name.endswith("_rows")
        row_capacity = ROW_CAPACITY.get(name, 8)
        assert 1 <= m <= (64 if row_tiled or name in ROW_CAPACITY else 8)
        assert k % 128 == 0 and n % (tiles * nw * 16) == 0
        assert x.dtype == scales.dtype == out.dtype == torch.float16
        assert weight.dtype == torch.int32 and weight.shape == (k // 16, n * 2)
        assert scales.shape == (k // 128, n)
        assert all(t.is_contiguous() for t in (x, weight, scales, out))
        self.call(
            name,
            (x, weight, scales, out),
            (m, k, n),
            n // (tiles * nw * 16),
            nw * kw * 32,
            (m + row_capacity - 1) // row_capacity
            if row_tiled or name in ROW_CAPACITY
            else 1,
            tiles * kw * 32 * 4 if name in RESERVE_CONTROLS else 0,
        )

    def measure(self, fn, inputs, cold, repeats):
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
        ).tolist()


def main(args):
    sys.path.insert(0, str(args.cupti_root.resolve()))
    from cupti import cupti

    assert cupti is not None and int(version("cupti-python").split(".")[0]) >= 13
    sys.path.insert(0, str(args.humming_root.resolve()))
    import humming
    from humming.tune import get_heuristics_config

    direct = None
    if args.humming_direct:
        from humming_direct import DirectExperiment

        direct = DirectExperiment()

    from vllm.utils import humming as hm

    assert Path(humming.__file__).resolve().is_relative_to(args.humming_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    torch.empty(0, device="cuda")
    module = SharedModule(args.output_dir / "build")
    locks = hm.create_humming_locks(torch.device("cuda:0"))
    selected = json.loads(args.selection.read_text()) if args.selection else {}
    reference_selected = (
        json.loads(args.reference_selection.read_text())
        if args.reference_selection
        else {}
    )
    custom_seed = (
        json.loads(args.custom_selection.read_text()) if args.custom_selection else {}
    )
    selections, results, samples, checks, failures, parameters, switches = (
        {},
        [],
        [],
        [],
        [],
        [],
        [],
    )
    source = Path(__file__).resolve().parent
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "humming_import": humming.__file__,
        "humming_commit": subprocess.check_output(
            ["git", "-C", str(args.humming_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "vllm_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "timing": "CUPTI; fixed pointers; graph replay; 100ms warmup; reversed rounds",
        "cupti_version": version("cupti-python"),
        "cold": "FlashInfer 2xL2 flush outside profiled kernel; no parameter rotation",
        "correctness": "CPU FP32; atol=.002 rtol=.02; relative L2<.001",
        "resident_parameters": "one Humming allocation; reference on CPU",
        "excluded": "initial transform, compilation, allocation, host, cache flush",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.iterdir()
            if p.is_file()
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def save():
        for name, value in (
            ("selection", selections),
            ("correctness", checks),
            ("failures", failures),
            ("parameters", parameters),
            ("switches", switches),
            ("samples", samples),
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
        reference_cpu = weights["reference"].cpu().float()
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
        assert hc.shape_n == n and hc.shape_k == k and not hc.has_zero_point
        w, s = ht["weight"], ht["weight_scale"]
        del weights, ht
        fingerprint = (w.data_ptr(), s.data_ptr(), w._version, s._version)
        parameters.append(
            dict(
                layer=layer,
                weight_ptr=w.data_ptr(),
                scale_ptr=s.data_ptr(),
                weight_shape=list(w.shape),
                scale_shape=list(s.shape),
                bytes=w.numel() * w.element_size() + s.numel() * s.element_size(),
                layouts=1,
                reference_device="cpu",
            )
        )
        buffers = {}
        for m in args.batches:
            key = f"{layer}/M{m}"
            print(f"START {key}", flush=True)
            x = torch.empty((m, k), device="cuda", dtype=torch.float16)
            out = torch.empty((m, n), device="cuda", dtype=torch.float16)
            configs = {
                "humming_default": get_heuristics_config(hc, m, use_f16_accum=False)
            }
            default = configs["humming_default"]
            if default["num_stages"] != 2:
                configs["humming_stages2"] = dict(default, num_stages=2)
            if default.get("use_stream_k"):
                configs["humming_no_streamk"] = dict(default, use_stream_k=False)
            candidates = {}
            for name, config in configs.items():
                tuning = json.dumps(config)

                def humming_call(a, b, c, d, tuning=tuning, hc=hc):
                    return hm.humming_forward(
                        hc,
                        inputs=a,
                        weight=b,
                        weight_scale=c,
                        outputs=d,
                        locks=locks,
                        compute_config=COMPUTE,
                        tuning_config=tuning,
                    )

                candidates[name] = ("humming", humming_call)
                if direct is not None:
                    for family, mode, compact in (
                        ("humming", 0, 0),
                        ("humming_direct_b", 1, 1),
                        ("humming_direct_all", 2, 2),
                    ):
                        variant = (
                            name
                            if family == "humming"
                            else family + "_" + name.removeprefix("humming_")
                        )
                        fn, kernel = direct.prepare(
                            hc, COMPUTE, tuning, mode, compact, locks
                        )
                        candidates[variant] = (family, fn)
                        from humming.ops.launcher import get_kernel_smem_size

                        parameters.append(
                            dict(
                                case=key,
                                impl=variant,
                                config=config,
                                cubin=kernel.kernel_filename,
                                shared_bytes=get_kernel_smem_size(kernel.kernel_id),
                                registers=checked(
                                    driver.cuFuncGetAttribute(
                                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
                                        kernel.load_cubin(),
                                    )
                                ),
                            )
                        )
            if m <= args.max_custom_m:
                for name in CONFIGS:
                    if name not in ROW_CAPACITY and name.endswith("_rows") != (m > 8):
                        continue
                    if custom_seed and name != custom_seed[key]["custom"]:
                        continue
                    if args.variants and name not in args.variants:
                        continue

                    def custom_call(a, b, c, d, name=name):
                        module.launch(name, a, b, c, d)

                    candidates[name] = ("custom", custom_call)
            if selected:
                keep = set(selected[key].values()) | {"humming_default"}
                if direct is not None:
                    suffixes = set()
                    for family in ("humming", "humming_direct_b", "humming_direct_all"):
                        suffixes.add(selected[key][family].removeprefix(family + "_"))
                    for suffix in suffixes:
                        keep.update(
                            family + "_" + suffix
                            for family in (
                                "humming",
                                "humming_direct_b",
                                "humming_direct_all",
                            )
                        )
                if reference_selected:
                    keep.update(reference_selected[key].values())
                candidates = {name: v for name, v in candidates.items() if name in keep}
            valid, eligible = {}, {}
            for name, (family, fn) in candidates.items():
                try:
                    strict = True
                    for kind in ("zero", "one", "random0", "random1", "random2"):
                        if kind == "zero":
                            cpu_x = torch.zeros((m, k), dtype=torch.float16)
                        elif kind == "one":
                            cpu_x = torch.ones((m, k), dtype=torch.float16)
                        else:
                            torch.manual_seed(int(kind[-1]))
                            cpu_x = torch.randn((m, k), dtype=torch.float16)
                        x.copy_(cpu_x)
                        ref = (cpu_x.float() @ reference_cpu).half().cuda()
                        fn(x, w, s, out)
                        error = check(out, ref)
                        strict &= error["original_elementwise_tolerance_passed"]
                        checks.append(dict(case=key, impl=name, input=kind, **error))
                    valid[name] = (family, fn)
                    eligible[name] = strict
                except Exception as exc:
                    failures.append(dict(case=key, impl=name, error=repr(exc)))
                    save()
                    if family == "custom":
                        raise
            families = ("custom", "humming") if m <= args.max_custom_m else ("humming",)
            if direct is not None:
                families += ("humming_direct_b", "humming_direct_all")
            for family in families:
                assert any(
                    f == family and eligible[name] for name, (f, _) in valid.items()
                ), key
            if selected:
                assert all(
                    name in valid and eligible[name] for name in selected[key].values()
                ), key
            if args.phase == "profile":
                assert selected
                for name in selected[key].values():
                    fn = valid[name][1]
                    for _ in range(10):
                        fn(x, w, s, out)
                    torch.accelerator.synchronize()
                    print(f"PROFILE {key} {name}", flush=True)
                    torch.cuda.profiler.start()
                    fn(x, w, s, out)
                    torch.accelerator.synchronize()
                    torch.cuda.profiler.stop()
            if args.phase not in ("check", "profile"):
                medians = {}
                for cold in (True,) if args.phase == "tune" else (True, False):
                    rounds = {name: [] for name in valid}
                    order = list(valid)
                    for sequence in (order, list(reversed(order))):
                        for name in sequence:
                            fn = valid[name][1]
                            rounds[name].append(
                                module.measure(fn, (x, w, s, out), cold, args.repeats)
                            )
                    for name, r in rounds.items():
                        us = float(np.median(r[0] + r[1]))
                        results.append(
                            dict(
                                layer=layer,
                                M=m,
                                K=k,
                                N=n,
                                cache="cold" if cold else "warm",
                                family=valid[name][0],
                                impl=name,
                                median_us=us,
                                round0_us=float(np.median(r[0])),
                                round1_us=float(np.median(r[1])),
                                effective_gbps=(
                                    k * n * (0.5 + 2 / 128) + 2 * m * (k + n)
                                )
                                / us
                                / 1000,
                            )
                        )
                        samples.append(dict(case=key, impl=name, cold=cold, rounds=r))
                        if cold:
                            medians[name] = us
                selections[key] = (
                    selected[key]
                    if selected
                    else {
                        family: min(
                            (
                                name
                                for name, (f, _) in valid.items()
                                if f == family and eligible[name]
                            ),
                            key=medians.get,
                        )
                        for family in families
                    }
                )
            else:
                selections[key] = selected[key]
            choice = selections[key]
            buffers[m] = (x, out, valid[choice["custom" if m <= 8 else "humming"]][1])
            assert fingerprint == (w.data_ptr(), s.data_ptr(), w._version, s._version)
            print(f"DONE {key}: {choice}", flush=True)
            save()
        sequence = [m for m in (1, 16, 2, 32, 4, 16, 8, 32) if m in buffers]
        for m in sequence:
            x, out, fn = buffers[m]
            cpu_x = torch.ones((m, k), dtype=torch.float16)
            x.copy_(cpu_x)
            fn(x, w, s, out)
            error = check(out, (cpu_x.float() @ reference_cpu).half().cuda())
            assert error["original_elementwise_tolerance_passed"]
            assert fingerprint == (w.data_ptr(), s.data_ptr(), w._version, s._version)
            switches.append(
                dict(
                    layer=layer,
                    M=m,
                    weight_ptr=w.data_ptr(),
                    scale_ptr=s.data_ptr(),
                    **error,
                )
            )
        del candidates, valid, buffers
        save()
    print("SHARED_LAYOUT_EXPERIMENT_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cupti-root", type=Path, required=True)
    parser.add_argument("--humming-root", type=Path, required=True)
    parser.add_argument("--humming-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--reference-selection", type=Path)
    parser.add_argument("--humming-direct", action="store_true")
    parser.add_argument("--custom-selection", type=Path)
    parser.add_argument("--max-custom-m", type=int, choices=(8, 32, 64), default=8)
    parser.add_argument(
        "--phase", choices=("tune", "compare", "check", "profile"), default="tune"
    )
    parser.add_argument("--layers", nargs="+", choices=LAYERS, default=list(LAYERS))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--variants", nargs="+")
    main(parser.parse_args())
