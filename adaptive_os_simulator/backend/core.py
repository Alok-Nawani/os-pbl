"""
Core data structures for the adaptive OS simulator.
Includes Process, PCB, and Queue implementations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict
import time
import heapq


class ProcessState(Enum):
    """Process states in the system."""
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    TERMINATED = "TERMINATED"


@dataclass
class ProcessStats:
    """Statistics tracked for each process."""
    waiting_time: float = 0.0
    turnaround_time: float = 0.0
    response_time: Optional[float] = None
    completion_time: Optional[float] = None
    first_run_time: Optional[float] = None
    context_switches: int = 0


@dataclass
class PCB:
    """Process Control Block - maintains all process information."""
    pid: int
    arrival_time: float
    burst_time: float
    priority: int = 0
    remaining_time: float = None
    state: ProcessState = ProcessState.NEW
    stats: ProcessStats = None

    def __post_init__(self):
        """Initialize derived attributes."""
        self.remaining_time = self.burst_time if self.remaining_time is None else self.remaining_time
        self.stats = ProcessStats() if self.stats is None else self.stats

    def __lt__(self, other):
        """Compare PCBs based on priority and remaining time."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.remaining_time < other.remaining_time  # Then shorter remaining time


class ReadyQueue:
    """Priority queue implementation for ready processes."""
    def __init__(self):
        # Simple list-backed queue. We keep a map for quick lookup/removal.
        self._items: List[PCB] = []
        self._pid_map: Dict[int, PCB] = {}
        self._entry_count = 0

    def push(self, pcb: PCB) -> None:
        """Add a process to the ready queue.

        We maintain a list of PCBs and support different pop/peek modes
        (fcfs, sjf, priority). This avoids a single fixed heap-order
        that would force one scheduling discipline on all schedulers.
        """
        if pcb.pid in self._pid_map:
            self.remove(pcb.pid)

        self._items.append(pcb)
        self._pid_map[pcb.pid] = pcb
        self._entry_count += 1

    def _select_index(self, mode: str = 'fcfs') -> Optional[int]:
        """Return index in _items for the next process according to mode."""
        if not self._items:
            return None

        if mode == 'fcfs':
            # earliest arrival_time first, tie-breaker by pid
            best_idx = min(range(len(self._items)), key=lambda i: (self._items[i].arrival_time, self._items[i].pid))
            return best_idx
        elif mode == 'sjf':
            # shortest remaining time first, tie-breaker by arrival_time
            best_idx = min(range(len(self._items)), key=lambda i: (self._items[i].remaining_time, self._items[i].arrival_time))
            return best_idx
        elif mode == 'priority':
            # highest priority number first, tie-breaker by arrival_time
            best_idx = max(range(len(self._items)), key=lambda i: (self._items[i].priority, -self._items[i].arrival_time))
            return best_idx
        else:
            # default to FCFS
            return self._select_index('fcfs')

    def pop(self, mode: str = 'priority') -> Optional[PCB]:
        """Remove and return the next process according to the given mode."""
        idx = self._select_index(mode)
        if idx is None:
            return None
        pcb = self._items.pop(idx)
        if pcb.pid in self._pid_map:
            del self._pid_map[pcb.pid]
        return pcb

    def peek(self, mode: str = 'priority') -> Optional[PCB]:
        """View the next process according to mode without removing it."""
        idx = self._select_index(mode)
        if idx is None:
            return None
        return self._items[idx]

    def remove(self, pid: int) -> Optional[PCB]:
        """Remove a specific process by PID."""
        if pid not in self._pid_map:
            return None
        pcb = self._pid_map[pid]
        self._items = [p for p in self._items if p.pid != pid]
        del self._pid_map[pid]
        return pcb

    def update_priority(self, pid: int, new_priority: int) -> bool:
        """Update a process's priority and keep it in the queue."""
        pcb = self._pid_map.get(pid)
        if not pcb:
            return False
        pcb.priority = new_priority
        return True

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def __len__(self) -> int:
        return len(self._items)

    def get_all_processes(self) -> List[PCB]:
        return list(self._items)


class SystemMetrics:
    """System-wide metrics tracking."""
    
    def __init__(self):
        self.total_processes = 0
        self.completed_processes = 0
        self.total_waiting_time = 0.0
        self.total_turnaround_time = 0.0
        self.total_response_time = 0.0
        self.total_context_switches = 0
        # Simulation clock tracking
        self.sim_start_time: Optional[float] = None
        self.sim_last_time: Optional[float] = None
        self.sim_elapsed: float = 0.0
        self.cpu_busy_time = 0.0
        self.idle_time = 0.0
        self.process_history: Dict[int, PCB] = {}

    def add_process(self, pcb: PCB, sim_time: Optional[float] = None) -> None:
        """Record a new process."""
        self.total_processes += 1
        self.process_history[pcb.pid] = pcb
        # Initialize simulation clock using provided simulated time if available
        if sim_time is not None:
            if self.sim_start_time is None:
                self.sim_start_time = sim_time
            if self.sim_last_time is None:
                self.sim_last_time = sim_time

    def update_sim_clock(self, now: float, cpu_was_busy: bool) -> None:
        """Advance the simulation clock and attribute elapsed time."""
        if self.sim_start_time is None:
            self.sim_start_time = now
        if self.sim_last_time is None:
            self.sim_last_time = now

        if now < self.sim_last_time:
            # Ignore out-of-order timestamps
            return

        delta = now - self.sim_last_time
        if cpu_was_busy:
            self.cpu_busy_time += delta
        else:
            self.idle_time += delta

        self.sim_last_time = now
        self.sim_elapsed = max(self.sim_elapsed, self.sim_last_time - self.sim_start_time)

    def process_completed(self, pcb: PCB, completion_time: float) -> None:
        """Record metrics for a completed process."""
        self.completed_processes += 1
        self.total_waiting_time += pcb.stats.waiting_time
        self.total_turnaround_time += pcb.stats.turnaround_time
        self.total_response_time += pcb.stats.response_time or 0.0
        self.total_context_switches += pcb.stats.context_switches
        self.update_sim_clock(completion_time, cpu_was_busy=False)

    def get_avg_waiting_time(self) -> float:
        """Calculate average waiting time."""
        # Prefer computing average from per-process records to avoid
        # any mismatch from incremental totals (more robust and debuggable).
        completed_waits = [p.stats.waiting_time for p in self.process_history.values() if p.stats.completion_time is not None]
        if not completed_waits:
            return 0.0
        return sum(completed_waits) / len(completed_waits)

    def get_avg_turnaround_time(self) -> float:
        """Calculate average turnaround time."""
        completed_tats = [p.stats.turnaround_time for p in self.process_history.values() if p.stats.completion_time is not None]
        if not completed_tats:
            return 0.0
        return sum(completed_tats) / len(completed_tats)

    def get_avg_response_time(self) -> float:
        """Calculate average response time."""
        if self.completed_processes == 0:
            return 0.0
        return self.total_response_time / self.completed_processes

    def get_throughput(self) -> float:
        """Calculate system throughput (processes per unit time)."""
        elapsed_time = max(1e-9, self.sim_elapsed)
        return self.completed_processes / elapsed_time

    def get_cpu_utilization(self) -> float:
        """Calculate CPU utilization percentage."""
        elapsed_time = max(1e-9, self.sim_elapsed)
        return (self.cpu_busy_time / elapsed_time) * 100
