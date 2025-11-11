from adaptive_os_simulator.backend.utils import Process
from adaptive_os_simulator.backend.simulator import simulate, Scheduler


def test_fixed_rr_completes():
    procs = [
        Process("P1", 3, priority=1, arrival_time=0),
        Process("P2", 4, priority=2, arrival_time=0),
    ]
    result = simulate(procs, policy=Scheduler.RR, time_quantum=1.0)
    assert all(p.completion_time is not None for p in result.processes)
    assert result.total_time >= 7.0


def test_sjf_vs_srtf_ordering():
    procs = [
        Process("A", 8, priority=1, arrival_time=0),
        Process("B", 2, priority=1, arrival_time=1),
    ]
    r1 = simulate([Process(p.pid, p.burst_time, p.priority, p.arrival_time) for p in procs], policy=Scheduler.SJF)
    r2 = simulate([Process(p.pid, p.burst_time, p.priority, p.arrival_time) for p in procs], policy=Scheduler.SRTF)
    # SRTF should preempt and complete B earlier
    b_sjf = next(p for p in r1.processes if p.pid == "B").completion_time
    b_srtf = next(p for p in r2.processes if p.pid == "B").completion_time
    assert b_srtf <= b_sjf


