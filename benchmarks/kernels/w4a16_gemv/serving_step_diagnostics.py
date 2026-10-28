# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional asynchronous event measurements for the standalone serving experiment."""

import json
import time

import torch


class StepDiagnostics:
    def __init__(self, runner):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

        self.records = []
        self.current = None
        self.last = None
        original_execute = runner.execute_model
        original_prepare = runner.prepare_inputs
        original_forward = runner.model.forward
        original_graph = ModelCudaGraphManager.run_fullgraph
        original_sample = runner.sample_tokens

        def timed(record, name, function, *args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            begin = time.perf_counter()
            result = function(*args, **kwargs)
            record[name + "_host_ms"] = (time.perf_counter() - begin) * 1000
            end.record()
            record["events"][name] = (start, end)
            return result

        def execute(scheduler_output, *args, **kwargs):
            if (
                kwargs.get("dummy_run")
                or not scheduler_output.total_num_scheduled_tokens
            ):
                return original_execute(scheduler_output, *args, **kwargs)
            record = dict(
                start=time.perf_counter(),
                requests=len(scheduler_output.num_scheduled_tokens),
                tokens=scheduler_output.total_num_scheduled_tokens,
                new_requests=len(scheduler_output.scheduled_new_reqs),
                full_graph=False,
                events={},
            )
            self.current = record
            try:
                result = timed(
                    record,
                    "execute",
                    original_execute,
                    scheduler_output,
                    *args,
                    **kwargs,
                )
            finally:
                self.current = None
            record["execute_end"] = time.perf_counter()
            self.records.append(record)
            self.last = record
            return result

        def prepare(*args, **kwargs):
            batch = original_prepare(*args, **kwargs)
            if self.current is not None:
                self.current["prefill_requests"] = int(batch.is_prefilling_np.sum())
                self.current["padded_tokens"] = int(batch.num_tokens_after_padding)
            return batch

        def forward(*args, **kwargs):
            if self.current is None:
                return original_forward(*args, **kwargs)
            return timed(self.current, "forward", original_forward, *args, **kwargs)

        def graph(manager, descriptor):
            if self.current is None:
                return original_graph(manager, descriptor)
            self.current["full_graph"] = True
            return timed(self.current, "forward", original_graph, manager, descriptor)

        def sample(*args, **kwargs):
            if self.last is None:
                return original_sample(*args, **kwargs)
            result = timed(self.last, "sample", original_sample, *args, **kwargs)
            self.last["sample_end"] = time.perf_counter()
            self.last = None
            return result

        runner.execute_model = execute
        runner.prepare_inputs = prepare
        runner.model.forward = forward
        runner.sample_tokens = sample
        ModelCudaGraphManager.run_fullgraph = graph

    def save(self, path):
        torch.accelerator.synchronize()
        output = []
        previous = None
        for record in self.records:
            values = {key: value for key, value in record.items() if key != "events"}
            for name, (start, end) in record["events"].items():
                values[name + "_stream_ms"] = start.elapsed_time(end)
            if previous is not None and "sample" in previous["events"]:
                values["between_steps_stream_ms"] = previous["events"]["sample"][
                    1
                ].elapsed_time(record["events"]["execute"][0])
            output.append(values)
            previous = record
        path.write_text(json.dumps(output, indent=2))
