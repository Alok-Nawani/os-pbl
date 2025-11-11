from __future__ import annotations

import shlex
from typing import List, Optional
from colorama import Fore, Style, init as colorama_init

from .utils import Process
from .simulator import simulate, Scheduler
from .adaptive_controller import rule_based_decider, ml_driven_decider
from .visualizer import plot_gantt


class ManualTerminal:
    def __init__(self) -> None:
        colorama_init(autoreset=True)
        self.processes: List[Process] = []
        self.last_result = None

    def prompt(self) -> None:
        print(Fore.CYAN + "Mini OS Terminal. Type 'help' for commands.")
        while True:
            try:
                raw = input(Fore.GREEN + "> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw.strip():
                continue
            self.handle_command(raw)

    def handle_command(self, raw: str) -> None:
        try:
            parts = shlex.split(raw)
        except ValueError as e:
            print(Fore.RED + f"Parse error: {e}")
            return
        if not parts:
            return
        cmd, *args = parts
        cmd = cmd.lower()
        if cmd == "help":
            self._help()
        elif cmd == "add":
            self._add(args)
        elif cmd == "list":
            self._list()
        elif cmd == "run":
            self._run(args)
        elif cmd == "stats":
            self._stats()
        elif cmd == "exit" or cmd == "quit":
            raise SystemExit(0)
        else:
            print(Fore.YELLOW + "Unknown command. Type 'help'.")

    def _help(self) -> None:
        print("Commands:")
        print("  add <pid> <burst> <priority> [arrival=0]")
        print("  list")
        print("  run [--mode fixed|rule|adaptive] [--policy RR|SJF|SRTF|PRIORITY|FCFS] [--quantum Q] [--window W] [--out path]")
        print("  stats")
        print("  exit")

    def _add(self, args: List[str]) -> None:
        if len(args) < 3:
            print(Fore.RED + "Usage: add <pid> <burst> <priority> [arrival]")
            return
        pid = args[0]
        try:
            burst = float(args[1])
            priority = int(args[2])
            arrival = float(args[3]) if len(args) >= 4 else 0.0
        except ValueError:
            print(Fore.RED + "Invalid numeric values")
            return
        self.processes.append(Process(pid=pid, burst_time=burst, priority=priority, arrival_time=arrival))
        print(Fore.CYAN + f"Process {pid} added: burst={burst}, priority={priority}, arrival={arrival}")

    def _list(self) -> None:
        if not self.processes:
            print("No processes yet")
            return
        for p in self.processes:
            print(f"{p.pid}: burst={p.burst_time}, priority={p.priority}, arrival={p.arrival_time}")

    def _run(self, args: List[str]) -> None:
        mode = "rule"
        policy = Scheduler.RR
        quantum = 2.0
        window = 2.0
        out_path: Optional[str] = None
        # Parse simple flags
        it = iter(args)
        for token in it:
            if token == "--mode":
                mode = next(it, mode)
            elif token == "--policy":
                policy = next(it, policy)
            elif token == "--quantum":
                try:
                    quantum = float(next(it))
                except (TypeError, ValueError):
                    pass
            elif token == "--window":
                try:
                    window = float(next(it))
                except (TypeError, ValueError):
                    pass
            elif token == "--out":
                out_path = next(it, None)

        procs = [Process(pid=p.pid, burst_time=p.burst_time, priority=p.priority, arrival_time=p.arrival_time) for p in self.processes]

        if mode == "fixed":
            result = simulate(procs, policy=policy, time_quantum=quantum)
        elif mode == "adaptive":
            result = simulate(procs, policy=policy, time_quantum=quantum, adaptive_policy_fn=ml_driven_decider, window=window)
        elif mode == "rule":
            result = simulate(procs, policy=policy, time_quantum=quantum, adaptive_policy_fn=rule_based_decider, window=window)
        else:
            print(Fore.RED + "Unknown mode")
            return

        self.last_result = result
        print(Style.BRIGHT + f"Simulation finished. Avg waiting: {result.avg_waiting_time:.2f}, Avg turnaround: {result.avg_turnaround_time:.2f}, Throughput: {result.throughput:.3f}")
        if out_path:
            plot_gantt(procs, result.logger, out_path)
            print(Fore.CYAN + f"Saved plot to {out_path}")

    def _stats(self) -> None:
        if not self.last_result:
            print("No simulation yet")
            return
        r = self.last_result
        print(f"Avg waiting time: {r.avg_waiting_time:.3f}")
        print(f"Avg turnaround time: {r.avg_turnaround_time:.3f}")
        print(f"Throughput: {r.throughput:.3f} jobs/s")


def main() -> None:
    ManualTerminal().prompt()


if __name__ == "__main__":
    main()


