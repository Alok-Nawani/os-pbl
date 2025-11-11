from __future__ import annotations

import argparse
from typing import List
import os
import sys

# Ensure project root is on sys.path when running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.utils import Process
from backend.simulator import simulate, Scheduler
from backend.adaptive_controller import rule_based_decider, ml_driven_decider, get_model_decider
from backend.visualizer import plot_gantt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive OS Scheduler Simulator")
    p.add_argument("--mode", choices=["fixed", "rule", "adaptive"], default="fixed")
    p.add_argument("--policy", choices=[Scheduler.RR, Scheduler.SJF, Scheduler.SRTF, Scheduler.PRIORITY, Scheduler.FCFS], default=Scheduler.RR)
    p.add_argument("--n", type=int, default=10, help="Number of synthetic processes")
    p.add_argument("--quantum", type=float, default=2.0)
    p.add_argument("--window", type=float, default=2.0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--model", type=str, default=None, help="Path to trained model .joblib for --mode adaptive")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def generate_workload(n: int, seed: int) -> List[Process]:
    import random
    random.seed(seed)
    procs: List[Process] = []
    time = 0.0
    for i in range(n):
        burst = max(1.0, random.expovariate(1/4))
        priority = random.randint(0, 4)
        inter_arrival = random.expovariate(1/1.5)
        time += inter_arrival
        procs.append(Process(pid=f"P{i+1}", burst_time=burst, priority=priority, arrival_time=round(time, 2)))
    return procs


def main() -> None:
    args = parse_args()
    procs = generate_workload(args.n, args.seed)
    if args.mode == "fixed":
        result = simulate(procs, policy=args.policy, time_quantum=args.quantum)
    elif args.mode == "rule":
        result = simulate(procs, policy=args.policy, time_quantum=args.quantum, adaptive_policy_fn=rule_based_decider, window=args.window)
    else:
        decider = ml_driven_decider if not args.model else get_model_decider(args.model)
        result = simulate(procs, policy=args.policy, time_quantum=args.quantum, adaptive_policy_fn=decider, window=args.window)

    print(f"Avg waiting: {result.avg_waiting_time:.3f}, Avg turnaround: {result.avg_turnaround_time:.3f}, Throughput: {result.throughput:.3f}")
    if args.out:
        plot_gantt(procs, result.logger, args.out)
        print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()


