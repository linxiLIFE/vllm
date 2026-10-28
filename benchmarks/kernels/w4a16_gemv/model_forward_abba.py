# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Alternate kernels within one loaded model to isolate GPU forward differences."""

import argparse
import json
from pathlib import Path

from model_dispatch import install_dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from vllm.benchmarks.forward import main as forward
    from vllm.model_executor.layers.quantization.humming import HummingLinearMethod
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    original = HummingLinearMethod.apply
    counts = install_dispatch(args.output_dir / "build", True, False)
    hybrid = HummingLinearMethod.apply
    original_run_case = forward.run_case
    records = []

    def run_case(runner, case, options):
        result = None
        for repeat in range(2):
            for backend in ("humming", "hybrid", "hybrid", "humming"):
                HummingLinearMethod.apply = original if backend == "humming" else hybrid
                result = original_run_case(runner, case, options)
                records.append(
                    dict(case=case.name, repeat=repeat, backend=backend, **result)
                )
                (args.output_dir / "results.json").write_text(
                    json.dumps(records, indent=2)
                )
                print(
                    case.name,
                    repeat,
                    backend,
                    result["latency_us"]["median"],
                    flush=True,
                )
        return result

    forward.run_case = run_case
    forward_parser = FlexibleArgumentParser()
    forward.add_cli_args(forward_parser)
    forward.main(forward_parser.parse_args(remaining))
    (args.output_dir / "dispatch.json").write_text(
        json.dumps({str(key): value for key, value in counts.items()}, indent=2)
    )


if __name__ == "__main__":
    main()
