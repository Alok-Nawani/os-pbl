from __future__ import annotations

import os
import sys

# ensure project root on sys.path when running from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.utils import Process
from backend.simulator import simulate, Scheduler
from backend.adaptive_controller import rule_based_decider


def make_workload():
    # Designed to trigger different decisions across windows:
    # - start with many jobs (>=6) so rule picks RR
    # - include several very short jobs so next window may pick SJF
    # - leave a small set of long jobs later to favor FCFS
    procs = [
        Process(pid="P1", burst_time=8.0, arrival_time=0.0),
        Process(pid="P2", burst_time=7.0, arrival_time=0.1),
        Process(pid="P3", burst_time=6.0, arrival_time=0.2),
        Process(pid="P4", burst_time=0.5, arrival_time=0.3),
        Process(pid="P5", burst_time=0.5, arrival_time=0.4),
        Process(pid="P6", burst_time=0.6, arrival_time=0.5),
        # short jobs arriving later to tip towards SJF around t=3
        Process(pid="P7", burst_time=0.4, arrival_time=3.0),
        Process(pid="P8", burst_time=0.4, arrival_time=3.2),
    ]
    return procs


if __name__ == '__main__':
    procs = make_workload()
    # Start with FCFS so we can observe a switch to RR (many jobs), then to SJF, then to FCFS as long jobs remain
    result = simulate(procs, policy=Scheduler.FCFS, time_quantum=2.0, adaptive_policy_fn=rule_based_decider, window=2.0)

    print("--- Simulation Summary ---")
    print(f"Total time: {result.total_time:.2f}")
    print(f"Avg waiting: {result.avg_waiting_time:.3f}, Avg turnaround: {result.avg_turnaround_time:.3f}, Throughput: {result.throughput:.3f}")

    print("\n--- Policy switches ---")
    for s in result.logger.policy_switches:
        print(f"t={s['time']:.2f}: {s['from']} -> {s['to']}  ({s.get('reason')})")

    print("\n--- Timeline (slices) ---")
    for t in result.logger.timeline:
        pid = t['pid'] or '<idle/cs>'
        print(f"{t['start']:.2f} - {t['end']:.2f} : {pid} [{t['policy']}] {t.get('reason','')}")

    print('\n--- Per-process completion times ---')
    for p in result.processes:
        print(f"{p.pid}: arrival={p.arrival_time}, burst={p.burst_time}, start={p.start_time}, completion={p.completion_time}")
