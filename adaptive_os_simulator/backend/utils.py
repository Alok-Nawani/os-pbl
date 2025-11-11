from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json
import csv
import math
import random


@dataclass
class Process:
    pid: str
    burst_time: float
    priority: int = 0
    arrival_time: float = 0.0
    remaining_time: float = field(init=False)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    color: Optional[str] = None

    def __post_init__(self) -> None:
        self.remaining_time = float(self.burst_time)
        if self.color is None:
            # Generate a stable color from pid
            random.seed(hash(self.pid) & 0xFFFFFFFF)
            r = random.randint(50, 220)
            g = random.randint(50, 220)
            b = random.randint(50, 220)
            self.color = f"#{r:02x}{g:02x}{b:02x}"


class EventLogger:
    def __init__(self) -> None:
        self.policy_switches: List[Dict[str, Any]] = []
        self.process_events: List[Dict[str, Any]] = []
        self.timeline: List[Dict[str, Any]] = []

    def log_policy_switch(self, time_s: float, from_policy: str, to_policy: str, reason: str) -> None:
        self.policy_switches.append({
            "time": time_s,
            "from": from_policy,
            "to": to_policy,
            "reason": reason,
        })

    def log_process_event(self, time_s: float, pid: str, event: str) -> None:
        self.process_events.append({
            "time": time_s,
            "pid": pid,
            "event": event,
        })

    def log_timeline_slice(self, start: float, end: float, pid: Optional[str], policy: str, reason: Optional[str] = None) -> None:
        self.timeline.append({
            "start": start,
            "end": end,
            "pid": pid,
            "policy": policy,
            "reason": reason,
        })

    def export_json(self, path: str) -> None:
        data = {
            "policy_switches": self.policy_switches,
            "process_events": self.process_events,
            "timeline": self.timeline,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, base_path_no_ext: str) -> None:
        with open(f"{base_path_no_ext}_switches.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "from", "to", "reason"])
            writer.writeheader()
            for row in self.policy_switches:
                writer.writerow(row)
        with open(f"{base_path_no_ext}_events.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "pid", "event"])
            writer.writeheader()
            for row in self.process_events:
                writer.writerow(row)
        with open(f"{base_path_no_ext}_timeline.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["start", "end", "pid", "policy", "reason"])
            writer.writeheader()
            for row in self.timeline:
                writer.writerow(row)


def compute_waiting_times(processes: List[Process]) -> Dict[str, float]:
    waiting: Dict[str, float] = {}
    for p in processes:
        if p.completion_time is None:
            continue
        turnaround = p.completion_time - p.arrival_time
        waiting[p.pid] = max(0.0, turnaround - p.burst_time)
    return waiting


def compute_turnaround_times(processes: List[Process]) -> Dict[str, float]:
    tat: Dict[str, float] = {}
    for p in processes:
        if p.completion_time is None:
            continue
        tat[p.pid] = max(0.0, p.completion_time - p.arrival_time)
    return tat


def compute_avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_throughput(processes: List[Process], total_time: float) -> float:
    if total_time <= 0:
        return 0.0
    completed = len([p for p in processes if p.completion_time is not None])
    return completed / total_time


