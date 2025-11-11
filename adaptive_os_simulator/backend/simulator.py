from __future__ import annotations

from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass

from .utils import Process, EventLogger, compute_waiting_times, compute_turnaround_times, compute_avg, compute_throughput


@dataclass
class SimulationResult:
    processes: List[Process]
    total_time: float
    waiting_times: Dict[str, float]
    turnaround_times: Dict[str, float]
    avg_waiting_time: float
    avg_turnaround_time: float
    throughput: float
    logger: EventLogger


class Scheduler:
    RR = "RR"
    SJF = "SJF"           # non-preemptive
    SRTF = "SRTF"         # preemptive SJF
    PRIORITY = "PRIORITY" # non-preemptive (lower number = higher priority)
    FCFS = "FCFS"         # non-preemptive First-Come, First-Served


def simulate(
    processes: List[Process],
    policy: str = Scheduler.RR,
    time_quantum: float = 2.0,
    context_switch_time: float = 0.0,
    adaptive_policy_fn: Optional[Callable[[float, List[Process]], Dict[str, Any]]] = None,
    window: float = 2.0,
) -> SimulationResult:
    time_now: float = 0.0
    ready: List[Process] = []
    completed: List[Process] = []
    pending = sorted(processes, key=lambda p: p.arrival_time)
    logger = EventLogger()
    current_policy = policy

    def arrive_new():
        nonlocal pending, ready, time_now
        while pending and pending[0].arrival_time <= time_now:
            ready.append(pending.pop(0))

    # initial arrivals
    arrive_new()

    next_window_time = window if adaptive_policy_fn else float("inf")
    last_slice_start = time_now

    def pick_next(proc_list: List[Process], pol: str) -> Optional[Process]:
        if not proc_list:
            return None
        if pol == Scheduler.RR:
            return proc_list[0]
        if pol == Scheduler.FCFS:
            # earliest arrival among ready
            return min(proc_list, key=lambda p: (p.arrival_time, p.pid))
        if pol == Scheduler.SJF:
            return min(proc_list, key=lambda p: p.remaining_time)
        if pol == Scheduler.SRTF:
            return min(proc_list, key=lambda p: p.remaining_time)
        if pol == Scheduler.PRIORITY:
            return min(proc_list, key=lambda p: (p.priority, p.arrival_time))
        return proc_list[0]

    rr_slice_remaining: float = time_quantum
    last_executing: Optional[Process] = None

    while len(completed) < len(processes):
        # Advance to next arrival if idle
        if not ready:
            if pending:
                # jump to next arrival
                time_now = max(time_now, pending[0].arrival_time)
                arrive_new()
                last_slice_start = time_now
                rr_slice_remaining = time_quantum
            else:
                break

        # adaptive window check
        if adaptive_policy_fn and time_now >= next_window_time:
            decision = adaptive_policy_fn(time_now, [p for p in processes if p.completion_time is None])
            new_policy = decision.get("policy", current_policy)
            reason = decision.get("reason", "adaptive switch")
            if new_policy != current_policy:
                logger.log_policy_switch(time_now, current_policy, new_policy, reason)
                current_policy = new_policy
            next_window_time += window

        arrive_new()
        current = pick_next(ready, current_policy)
        if current is None:
            # No ready proc, jump to next arrival
            if pending:
                time_now = max(time_now, pending[0].arrival_time)
                arrive_new()
                continue
            else:
                break

        if current.start_time is None:
            current.start_time = time_now
            logger.log_process_event(time_now, current.pid, "start")

        # Determine time slice to run
        run_for = 0.0
        preemption_reason: Optional[str] = None
        if current_policy == Scheduler.RR:
            run_for = min(current.remaining_time, rr_slice_remaining)
        elif current_policy in (Scheduler.SJF, Scheduler.FCFS):
            run_for = current.remaining_time
        elif current_policy == Scheduler.SRTF:
            # run until next arrival that could preempt, or finish
            # find time to next arrival
            next_arrival = pending[0].arrival_time if pending else float("inf")
            run_for = min(current.remaining_time, max(0.0, next_arrival - time_now))
            if run_for == 0 and pending:
                # immediate arrival: advance infinitesimally to trigger arrival
                time_now = next_arrival
                arrive_new()
                continue
        elif current_policy == Scheduler.PRIORITY:
            run_for = current.remaining_time

        # Execute
        time_slice_end = time_now + run_for
        logger.log_timeline_slice(time_now, time_slice_end, current.pid, current_policy)
        time_now = time_slice_end
        current.remaining_time -= run_for

        # arrivals during execution (only relevant for SRTF)
        arrive_new()

        # Decide what to do after slice
        finished = current.remaining_time <= 1e-9
        if finished:
            current.completion_time = time_now
            logger.log_process_event(time_now, current.pid, "complete")
            completed.append(current)
            ready.remove(current)
            rr_slice_remaining = time_quantum
            last_executing = None
            if context_switch_time > 0:
                logger.log_timeline_slice(time_now, time_now + context_switch_time, None, current_policy, reason="context_switch")
                time_now += context_switch_time
            continue

        # Not finished
        if current_policy == Scheduler.RR:
            rr_slice_remaining -= run_for
            if rr_slice_remaining <= 1e-9:
                # rotate
                ready.append(ready.pop(0))
                rr_slice_remaining = time_quantum
                last_executing = None
                if context_switch_time > 0:
                    logger.log_timeline_slice(time_now, time_now + context_switch_time, None, current_policy, reason="context_switch")
                    time_now += context_switch_time
        elif current_policy == Scheduler.SRTF:
            # Preempt if a new process has shorter remaining time
            best = min(ready, key=lambda p: p.remaining_time)
            if best is not current:
                # move current to end to mimic queue behavior
                # ensure current is first in ready
                # keep current in place but do nothing; next loop will pick best
                if context_switch_time > 0:
                    logger.log_timeline_slice(time_now, time_now + context_switch_time, None, current_policy, reason="preempt")
                    time_now += context_switch_time
            # else continue in next loop
        else:
            # For non-preemptive, continue same until finish; but we already consumed all remaining time
            # So put back to head for next iteration
            pass

    waiting_times = compute_waiting_times(processes)
    turnaround_times = compute_turnaround_times(processes)
    avg_wait = compute_avg(list(waiting_times.values()))
    avg_tat = compute_avg(list(turnaround_times.values()))
    throughput = compute_throughput(processes, time_now if time_now > 0 else 1.0)

    return SimulationResult(
        processes=processes,
        total_time=time_now,
        waiting_times=waiting_times,
        turnaround_times=turnaround_times,
        avg_waiting_time=avg_wait,
        avg_turnaround_time=avg_tat,
        throughput=throughput,
        logger=logger,
    )


