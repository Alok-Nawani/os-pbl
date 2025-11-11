from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List

# Ensure repo root is on sys.path so this script can be executed directly
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from adaptive_os_simulator.backend.utils import Process
from adaptive_os_simulator.backend.os_kernel import OSKernel, KernelConfig


def load_processes_from_dataset(path: str, limit: int | None = 50) -> List[Process]:
    procs: List[Process] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            # Map dataset columns to Process fields. We create synthetic burst & arrival times.
            avg_burst = float(row.get('avg_burst_time', 5))
            arrival_rate = float(row.get('arrival_rate', 5))
            priority_variance = float(row.get('priority_variance', 1))
            queue_length = int(row.get('queue_length', 5))

            pid = f"p{i}"
            # Use avg_burst directly as burst_time, and stagger arrivals by arrival_rate
            burst_time = max(0.1, avg_burst)
            arrival_time = float(i) * (1.0 / max(1.0, arrival_rate))
            # Derive priority from priority_variance heuristically
            priority = int(max(0, min(10, round(priority_variance) )))

            procs.append(Process(pid=pid, burst_time=burst_time, priority=priority, arrival_time=arrival_time))
    return procs


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run kernel demo from dataset')
    parser.add_argument('--limit', type=int, default=50, help='Max number of dataset rows to load (default: 50)')
    parser.add_argument('--outdir', type=str, default='kernel_run_logs', help='Output directory for logs')
    parser.add_argument('--policy', type=str, default='RR', help='Initial scheduling policy (default RR)')
    parser.add_argument('--time-quantum', type=float, default=2.0, help='Time quantum for RR')
    parser.add_argument('--window', type=float, default=2.0, help='Adaptive decision window')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset = repo_root / 'adaptive_scheduler_dataset_10k (4).csv'
    if not dataset.exists():
        print(f"Dataset not found at {dataset}")
        sys.exit(1)

    procs = load_processes_from_dataset(str(dataset), limit=args.limit)
    kernel = OSKernel(KernelConfig(policy=args.policy, time_quantum=args.time_quantum, context_switch_time=0.0, adaptive_window=args.window))
    result = kernel.run(procs)

    print(f"Total time: {result.total_time}")
    print(f"Avg waiting: {result.avg_waiting_time:.3f}")
    print(f"Avg turnaround: {result.avg_turnaround_time:.3f}")
    print(f"Throughput: {result.throughput:.3f}")

    out = Path.cwd() / args.outdir
    out.mkdir(exist_ok=True)
    # write timestamped files to avoid overwriting
    base = out / f'kernel_run_{args.limit}'
    result.logger.export_json(str(base.with_suffix('.json')))
    result.logger.export_csv(str(base))
    print(f"Logs written to {out} (base: {base})")


if __name__ == '__main__':
    main()
