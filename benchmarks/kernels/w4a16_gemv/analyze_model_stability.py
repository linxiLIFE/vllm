# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize the isolated serving diagnostics without rerunning inference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    clocks = json.loads((args.input_dir / "gpu-telemetry.json").read_text())
    rows = []
    for request_path in sorted(args.input_dir.glob("arm*-requests.json")):
        prefix = request_path.name.removesuffix("-requests.json")
        steps = json.loads((args.input_dir / (prefix + "-steps.json")).read_text())
        for case in json.loads(request_path.read_text()):
            selected = [
                step for step in steps if case["start"] <= step["start"] <= case["end"]
            ]
            decode = [
                step
                for step in selected
                if step.get("prefill_requests") == 0 and step["requests"] == 4
            ]
            prefill = [step for step in selected if step.get("prefill_requests", 0)]
            assert len(decode) > 900 and all(step["full_graph"] for step in decode)
            median = lambda key, samples=decode: float(
                np.median([step[key] for step in samples])
            )
            row = dict(
                arm=prefix,
                protocol=case["protocol"],
                repeat=case["repeat"],
                steps=len(selected),
                prefill_steps=len(prefill),
                pure_decode4_steps=len(decode),
                output_tokens_per_s=16 * 256 / (case["end"] - case["start"]),
                middle_itl_ms=float(
                    np.median(
                        [
                            value * 1000
                            for request in case["requests"]
                            for value in request["itl"][32:224]
                        ]
                    )
                ),
                ttft_ms=float(
                    np.median([request["ttft"] * 1000 for request in case["requests"]])
                ),
                tpot_ms=float(
                    np.median(
                        [
                            (request["latency"] - request["ttft"]) * 1000 / 255
                            for request in case["requests"]
                        ]
                    )
                ),
            )
            for key in (
                "forward_stream_ms",
                "sample_stream_ms",
                "execute_stream_ms",
                "execute_host_ms",
                "sample_host_ms",
            ):
                row[key] = median(key)
            for phase, selected_steps in (("prefill", prefill),):
                for key in ("forward_stream_ms", "forward_host_ms", "execute_host_ms"):
                    row[f"{phase}_{key}"] = float(
                        np.median([step[key] for step in selected_steps])
                    )
            intervals = []
            host_gaps = []
            for previous, following in zip(selected, selected[1:]):
                if all(
                    step.get("prefill_requests") == 0 and step["requests"] == 4
                    for step in (previous, following)
                ):
                    intervals.append((following["start"] - previous["start"]) * 1000)
                    host_gaps.append(
                        (following["start"] - previous["sample_end"]) * 1000
                    )
            row["step_interval_ms"] = float(np.median(intervals))
            row["host_between_calls_ms"] = float(np.median(host_gaps))
            selected_clocks = [
                clock
                for clock in clocks
                if case["start"] <= clock["time"] <= case["end"]
            ]
            for key in ("sm_mhz", "memory_mhz", "power_mw", "gpu_util"):
                row[key] = float(np.median([clock[key] for clock in selected_clocks]))
                row[key + "_min"] = min(clock[key] for clock in selected_clocks)
                row[key + "_max"] = max(clock[key] for clock in selected_clocks)
            rows.append(row)
    with (args.input_dir / "step-summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (args.input_dir / "step-summary.json").write_text(json.dumps(rows, indent=2))
    for arm in dict.fromkeys(row["arm"] for row in rows):
        selected = [row for row in rows if row["arm"] == arm]
        print(
            arm,
            {
                key: round(float(np.median([row[key] for row in selected])), 3)
                for key in (
                    "forward_stream_ms",
                    "sample_stream_ms",
                    "step_interval_ms",
                    "middle_itl_ms",
                    "output_tokens_per_s",
                    "sm_mhz",
                )
            },
        )


if __name__ == "__main__":
    main()
