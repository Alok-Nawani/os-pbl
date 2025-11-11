"""
Base scheduler implementations including FCFS, SJF, and Round Robin.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import time

from .core import PCB, ReadyQueue, SystemMetrics, ProcessState


class BaseScheduler(ABC):
    """Abstract base class for all schedulers."""
    
    def __init__(self):
        self.ready_queue = ReadyQueue()
        self.metrics = SystemMetrics()
        self.current_process: Optional[PCB] = None
        self.time_quantum: float = float('inf')  # Default to no time slicing
        self.last_update_time: Optional[float] = None

    @staticmethod
    def _resolve_time(sim_time: Optional[float]) -> float:
        """Return the appropriate clock reference."""
        return sim_time if sim_time is not None else time.time()

    @abstractmethod
    def schedule(self, sim_time: Optional[float] = None) -> Optional[PCB]:
        """Select the next process to run. Accepts simulated time for metrics."""
        pass

    def add_process(self, pcb: PCB, sim_time: Optional[float] = None) -> None:
        """Add a new process to the ready queue."""
        pcb.state = ProcessState.READY
        self.ready_queue.push(pcb)
        self.metrics.add_process(pcb, sim_time=sim_time)

    def context_switch(self, new_process: PCB, sim_time: Optional[float] = None) -> None:
        """Perform a context switch using simulated time."""
        now = self._resolve_time(sim_time)
        
        # Handle current process
        if self.current_process:
            # Account for elapsed run time before switching away
            if self.last_update_time is not None and now >= self.last_update_time:
                self.metrics.update_sim_clock(now, cpu_was_busy=self.current_process.remaining_time > 0)
            self.last_update_time = now

            if self.current_process.remaining_time > 0:
                self.current_process.state = ProcessState.READY
                self.ready_queue.push(self.current_process)
            else:
                self.current_process.state = ProcessState.TERMINATED
                # Complete using simulated time 'now'
                self.complete_process(self.current_process, sim_time=now)
            
            self.current_process.stats.context_switches += 1
        
        # Switch to new process
        self.current_process = new_process
        self.current_process.state = ProcessState.RUNNING
        
        if self.current_process.stats.first_run_time is None:
            self.current_process.stats.first_run_time = now
            self.current_process.stats.response_time = now - self.current_process.arrival_time
        self.last_update_time = now
        self.metrics.total_context_switches += 1

    def complete_process(self, pcb: PCB, sim_time: Optional[float] = None) -> None:
        """Handle process completion using simulated time."""
        now = self._resolve_time(sim_time)
        pcb.stats.completion_time = now
        pcb.stats.turnaround_time = now - pcb.arrival_time
        pcb.stats.waiting_time = max(0.0, pcb.stats.turnaround_time - pcb.burst_time)
        self.metrics.process_completed(pcb, completion_time=now)

    def update_metrics(self, sim_time: Optional[float] = None) -> None:
        """Update system-wide metrics using simulated time."""
        now = self._resolve_time(sim_time)

        if self.last_update_time is None:
            self.last_update_time = now
            self.metrics.update_sim_clock(now, cpu_was_busy=bool(self.current_process))
            return

        if now < self.last_update_time:
            return

        delta = now - self.last_update_time
        if delta <= 0:
            return

        cpu_busy = bool(self.current_process and self.current_process.remaining_time > 0)
        self.metrics.update_sim_clock(now, cpu_was_busy=cpu_busy)

        # Update waiting time for all ready processes
        for pcb in self.ready_queue.get_all_processes():
            pcb.stats.waiting_time += delta

        self.last_update_time = now


class FCFSScheduler(BaseScheduler):
    """First Come First Serve scheduler implementation."""
    
    def schedule(self, sim_time: Optional[float] = None) -> Optional[PCB]:
        """Select next process based on arrival time."""
        if self.current_process and self.current_process.remaining_time > 0:
            return self.current_process
        
        # FCFS: pop the process with earliest arrival
        next_process = self.ready_queue.pop(mode='fcfs')
        if next_process:
            self.context_switch(next_process, sim_time=sim_time)
        return next_process


class SJFScheduler(BaseScheduler):
    """Shortest Job First scheduler implementation."""
    
    def schedule(self, sim_time: Optional[float] = None) -> Optional[PCB]:
        """Select next process based on burst time."""
        if self.current_process and self.current_process.remaining_time > 0:
            # Non-preemptive SJF
            return self.current_process
        
        # SJF: pick the process with shortest remaining time
        next_process = self.ready_queue.pop(mode='sjf')
        if next_process:
            self.context_switch(next_process, sim_time=sim_time)
        return next_process


class SRTFScheduler(SJFScheduler):
    """Shortest Remaining Time First (preemptive SJF) scheduler."""
    
    def schedule(self, sim_time: Optional[float] = None) -> Optional[PCB]:
        """Preemptively schedule shortest remaining time process."""
        if not self.current_process:
            next_process = self.ready_queue.pop(mode='sjf')
            if next_process:
                self.context_switch(next_process, sim_time=sim_time)
            return next_process
        
        # Check if there's a process with shorter remaining time
        next_process = self.ready_queue.peek(mode='sjf')
        if (next_process and 
            next_process.remaining_time < self.current_process.remaining_time):
            next_process = self.ready_queue.pop(mode='sjf')
            self.context_switch(next_process, sim_time=sim_time)
            return next_process
        
        return self.current_process


class RoundRobinScheduler(BaseScheduler):
    """Round Robin scheduler implementation."""
    
    def __init__(self, time_quantum: float = 2.0):
        super().__init__()
        self.time_quantum = time_quantum
        self.current_quantum_start: float = 0.0

    def schedule(self, sim_time: Optional[float] = None) -> Optional[PCB]:
        """Schedule processes with time quantum. Uses simulated time when provided."""
        now = self._resolve_time(sim_time)
        
        # Check if current process has exceeded its time quantum
        if (self.current_process and 
            now - self.current_quantum_start >= self.time_quantum):
            # Time quantum expired, switch to next process
            next_process = self.ready_queue.pop(mode='fcfs')
            if next_process:
                self.context_switch(next_process, sim_time=sim_time)
                self.current_quantum_start = now
                return next_process
        
        # If no current process or it's completed
        if (not self.current_process or 
            self.current_process.remaining_time <= 0):
            next_process = self.ready_queue.pop(mode='fcfs')
            if next_process:
                self.context_switch(next_process, sim_time=sim_time)
                self.current_quantum_start = now
                return next_process
        
        return self.current_process


class PriorityScheduler(BaseScheduler):
    """Priority-based scheduler implementation."""
    
    def __init__(self, preemptive: bool = True):
        super().__init__()
        self.preemptive = preemptive

    def schedule(self, sim_time: Optional[float] = None) -> Optional[PCB]:
        """Schedule based on priority."""
        if not self.preemptive and self.current_process and self.current_process.remaining_time > 0:
            return self.current_process
        
        # Priority: peek at highest-priority waiting process
        next_process = self.ready_queue.peek(mode='priority')
        if (self.current_process and next_process and 
            next_process.priority > self.current_process.priority):
            # Preempt current process
            next_process = self.ready_queue.pop(mode='priority')
            self.context_switch(next_process, sim_time=sim_time)
            return next_process
        elif not self.current_process or self.current_process.remaining_time <= 0:
            # Current process completed
            next_process = self.ready_queue.pop(mode='priority')
            if next_process:
                self.context_switch(next_process, sim_time=sim_time)
            return next_process
        
        return self.current_process
