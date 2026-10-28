# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker hook for serving the experimental shared-layout dispatch."""

import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class Worker(GPUWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self.dispatch_counts = None
        if os.environ["W4A16_SERVING_BACKEND"] != "humming":
            from model_dispatch import install_dispatch

            self.dispatch_counts = install_dispatch(
                Path(os.environ["W4A16_SERVING_BUILD"]),
                True,
                False,
                custom_enabled=os.environ["W4A16_SERVING_BACKEND"] == "hybrid",
            )
        super().load_model(load_dummy_weights=load_dummy_weights)
        self.step_diagnostics = None
        if os.environ.get("W4A16_STEP_DIAGNOSTICS"):
            from serving_step_diagnostics import StepDiagnostics

            self.step_diagnostics = StepDiagnostics(self.model_runner)
        print(
            "W4A16_SERVING_BACKEND=" + os.environ["W4A16_SERVING_BACKEND"], flush=True
        )

    def shutdown(self) -> None:
        if self.step_diagnostics is not None:
            self.step_diagnostics.save(Path(os.environ["W4A16_STEP_DIAGNOSTICS"]))
        if self.dispatch_counts is not None:
            Path(os.environ["W4A16_SERVING_STATS"]).write_text(
                json.dumps(
                    {str(key): value for key, value in self.dispatch_counts.items()},
                    indent=2,
                )
            )
        super().shutdown()
