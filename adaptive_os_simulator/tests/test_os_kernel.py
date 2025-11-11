from __future__ import annotations

from adaptive_os_simulator.backend.utils import Process
from adaptive_os_simulator.backend.os_kernel import OSKernel, KernelConfig


def test_kernel_runs_simple():
    # create a few simple processes
    procs = [
        Process(pid='p1', burst_time=1.0, priority=1, arrival_time=0.0),
        Process(pid='p2', burst_time=2.0, priority=2, arrival_time=0.5),
        Process(pid='p3', burst_time=0.5, priority=3, arrival_time=1.0),
    ]

    kernel = OSKernel(KernelConfig(policy='RR', time_quantum=1.0, context_switch_time=0.0, adaptive_window=2.0))
    result = kernel.run(procs)

    # All processes should complete
    assert all(p.completion_time is not None for p in result.processes)
    # Total time should be at least sum of bursts
    assert result.total_time >= sum(p.burst_time for p in procs)
