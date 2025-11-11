from __future__ import annotations

from typing import List, Dict, Any
import statistics

from .core import PCB


class Scheduler:
    """Scheduler type constants."""
    RR = "Round Robin"
    FCFS = "FCFS"
    SJF = "SJF"
    SRTF = "SRTF"
    PRIORITY = "Priority"


def rule_based_decider(current_time: float, procs: List[PCB]) -> Dict[str, Any]:
    """Rule-based scheduler decision maker that works with PCB objects."""
    if not procs:
        return {"policy": Scheduler.RR, "reason": "no processes"}

    remaining_times = [p.remaining_time for p in procs]
    priorities = [p.priority for p in procs]
    mean_rt = sum(remaining_times) / len(remaining_times)
    short_jobs = len([t for t in remaining_times if t <= max(1.0, 0.5 * mean_rt)])
    prio_var = statistics.pvariance(priorities) if len(priorities) > 1 else 0.0

    if prio_var > 2.0:
        return {"policy": Scheduler.PRIORITY, "reason": "high priority variance"}
    if short_jobs >= max(2, len(procs) // 2):
        return {"policy": Scheduler.SJF, "reason": "many short jobs"}
    # Small queue, low variance and longer jobs -> FCFS is fine
    if len(procs) <= 3 and mean_rt >= 3.0 and prio_var < 0.5:
        return {"policy": Scheduler.FCFS, "reason": "small queue with similar priorities"}
    if len(procs) >= 6:
        return {"policy": Scheduler.RR, "reason": "many jobs, fair sharing"}
    # default to SRTF for responsiveness on mixed loads
    return {"policy": Scheduler.SRTF, "reason": "default responsiveness"}
