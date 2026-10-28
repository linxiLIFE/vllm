# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build an isolated SM86 FP16/INT4 Humming direct-load experiment."""

import argparse
import shutil
from pathlib import Path


def prepare_package(source, destination):
    """Copy the pinned package and apply the accompanying CUDA experiment patch."""
    import subprocess

    assert not destination.exists(), destination
    shutil.copytree(source / "humming", destination / "humming")
    patch = Path(__file__).with_suffix(".patch").resolve()
    subprocess.run(
        ["git", "apply", "--unsafe-paths", str(patch)], cwd=destination, check=True
    )
    shutil.copy2(patch, destination / "direct-load.patch")


class DirectExperiment:
    """Prepare independent Humming kernels and return fixed launcher closures."""

    def __init__(self):
        import humming
        from humming.kernel.humming import HummingKernel

        applied = Path(humming.__file__).resolve().parent.parent / "direct-load.patch"
        assert applied.read_bytes() == Path(__file__).with_suffix(".patch").read_bytes()
        self.kernel_class = HummingKernel
        self.mode = 0
        self.compact = 0
        self.caches = {}
        original = HummingKernel.prepare

        def prepare(kernel):
            kernel.code = (
                f"#define HUMMING_DIRECT_MODE {self.mode}\n"
                f"#define HUMMING_DIRECT_COMPACT {self.compact}\n" + kernel.code
            )
            return original(kernel)

        HummingKernel.prepare = prepare

    def prepare(self, layer, compute, tuning, mode, compact, locks):
        from humming.ops.launcher import launch_kernel

        self.mode, self.compact = mode, compact
        cache = self.caches.setdefault((mode, compact), ({}, {}))
        self.kernel_class._instances = cache[0]
        self.kernel_class._str2kernel_cache = cache[1]
        configs = self.kernel_class.prepare_kernels(
            layer.to_str(), compute, tuning, device=0
        )
        kernel = self.kernel_class._id2kernel[int(configs[2])]

        def launch(a, b, c, d):
            return launch_kernel(
                configs=configs,
                inputs=a,
                weight=b,
                weight_scale=c,
                outputs=d,
                locks=locks,
            )

        return launch, kernel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    prepare_package(args.source, args.destination)
