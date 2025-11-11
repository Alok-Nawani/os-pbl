"""
Adaptive scheduler implementation with dynamic adjustments.
"""

from typing import Dict, List, Optional
import time
import math
from collections import defaultdict

from .core import PCB, ProcessState
from .schedulers import BaseScheduler


class ProcessGroup:
    """Groups similar processes for prediction."""
    
    def __init__(self):
        self.burst_times: List[float] = []
        self.priorities: List[int] = []
        self.last_prediction: float = 0.0

    def add_process(self, pcb: PCB) -> None:
        """Add process metrics to group."""
        self.burst_times.append(pcb.burst_time)
        self.priorities.append(pcb.priority)
        
        # Update prediction using exponential moving average
        if not self.burst_times:
            self.last_prediction = pcb.burst_time
        else:
            alpha = 0.3  # Smoothing factor
            self.last_prediction = (alpha * pcb.burst_time + 
                                  (1 - alpha) * self.last_prediction)

    def predict_burst_time(self) -> float:
        """Predict burst time for next similar process."""
        if not self.burst_times:
            return 0.0
        return self.last_prediction


class AdaptiveScheduler(BaseScheduler):
    """
    Adaptive scheduler with dynamic adjustments based on:
    1. System load (for time quantum)
    2. Waiting time (for priority aging)
    3. Process history (for burst prediction)
    """
    
    def __init__(self, 
                 base_quantum: float = 2.0,
                 min_quantum: float = 0.5,
                 max_quantum: float = 4.0):
        super().__init__()
        self.base_quantum = base_quantum
        self.min_quantum = min_quantum
        self.max_quantum = max_quantum
        self.time_quantum = base_quantum
        self.current_quantum_start: float = 0.0
        
        # Adaptive components
        self.process_groups: Dict[str, ProcessGroup] = defaultdict(ProcessGroup)
        self.load_history: List[float] = []
        self.load_window = 5  # Number of samples for load average
        
        # Configuration
        self.load_check_interval = 1.0  # Seconds between load checks
        self.last_load_check = time.time()
        self.aging_factor = 1.2  # Priority boost for waiting processes
        self.load_threshold_high = 0.8  # 80% load
        self.load_threshold_low = 0.5   # 50% load

    def get_process_group_key(self, pcb: PCB) -> str:
        """Generate key for grouping similar processes."""
        # Group by burst time range and priority level
        burst_range = math.floor(pcb.burst_time / 2) * 2  # Group in 2-second ranges
        priority_level = math.floor(pcb.priority / 5) * 5  # Group in priority bands of 5
        return f"burst_{burst_range}_prio_{priority_level}"

    def update_load_metrics(self) -> None:
        """Update system load metrics and adjust scheduling parameters."""
        now = time.time()
        if now - self.last_load_check < self.load_check_interval:
            return

        # Calculate current load based on ready queue length
        ready_count = len(self.ready_queue)
        total_processes = ready_count + (1 if self.current_process else 0)
        current_load = min(1.0, float(total_processes) / max(1.0, 10.0))
        
        # Update load history
        self.load_history.append(current_load)
        if len(self.load_history) > self.load_window:
            self.load_history.pop(0)
        
        # Calculate average load
        avg_load = sum(self.load_history) / len(self.load_history)
        
        # Adjust time quantum based on load
        if avg_load > self.load_threshold_high:
            # High load: decrease quantum to improve responsiveness
            self.time_quantum = max(self.min_quantum, self.base_quantum * 0.8)
        elif avg_load < self.load_threshold_low:
            # Low load: increase quantum to reduce context switches
            self.time_quantum = min(self.max_quantum, self.base_quantum * 1.2)
        
        self.last_load_check = now

    def update_priorities(self) -> None:
        """Update process priorities based on waiting time."""
        avg_waiting_time = self.metrics.get_avg_waiting_time()
        # If no historical data, use zero threshold so aging can still apply
        threshold = avg_waiting_time * 1.5 if avg_waiting_time > 0 else 0.0

        for pcb in self.ready_queue.get_all_processes():
            # Boost priority if waiting too long
            if pcb.stats.waiting_time >= threshold:
                new_priority = min(99, math.floor(pcb.priority * self.aging_factor) + 1)
                self.ready_queue.update_priority(pcb.pid, new_priority)

    def predict_burst_time(self, pcb: PCB) -> float:
        """Predict burst time based on similar processes."""
        group_key = self.get_process_group_key(pcb)
        return self.process_groups[group_key].predict_burst_time()

    def add_process(self, pcb: PCB, sim_time: Optional[float] = None) -> None:
        """Add process with burst time prediction."""
        predicted_burst = self.predict_burst_time(pcb)
        if predicted_burst > 0:
            # Use prediction to adjust initial priority
            burst_factor = pcb.burst_time / predicted_burst
            pcb.priority = max(0, min(99, math.floor(
                pcb.priority * (1 / burst_factor)
            )))
        # Call parent implementation to actually enqueue the process
        super().add_process(pcb, sim_time=sim_time)

    def complete_process(self, pcb: PCB, sim_time: float = 0.0) -> None:
        """Update history when process completes (use simulated time)."""
        super().complete_process(pcb, sim_time=sim_time)

        # Add to process group for future prediction
        group_key = self.get_process_group_key(pcb)
        self.process_groups[group_key].add_process(pcb)

    def schedule(self, sim_time: Optional[float] = None) -> Optional[PCB]:
        """Adaptive scheduling decision. Accepts simulated time."""
        now = self._resolve_time(sim_time)

        # Update system metrics
        self.update_load_metrics()
        self.update_priorities()
        
        # Decide selection mode based on queue characteristics
        rq_procs = self.ready_queue.get_all_processes()
        selection_mode = 'fcfs'
        if rq_procs:
            remaining_times = [p.remaining_time for p in rq_procs]
            mean_rt = sum(remaining_times) / len(remaining_times)
            # If many short jobs, prefer SJF/SRTF
            short_jobs = len([t for t in remaining_times if t <= max(1.0, 0.5 * mean_rt)])
            if short_jobs >= max(2, len(rq_procs) // 2):
                selection_mode = 'sjf'
            else:
                # If priorities are heterogeneous, use priority scheduling
                prios = [p.priority for p in rq_procs]
                if max(prios) - min(prios) > 2:
                    selection_mode = 'priority'

        # Check if current process should continue (with possible preemption)
        if self.current_process and self.current_process.remaining_time > 0:
            # SRTF-like preemption: if a waiting process has shorter remaining time
            if selection_mode == 'sjf':
                nxt = self.ready_queue.peek(mode='sjf')
                if nxt and nxt.remaining_time < self.current_process.remaining_time:
                    next_process = self.ready_queue.pop(mode='sjf')
                    self.context_switch(next_process, sim_time=now)
                    self.current_quantum_start = now
                    return next_process

            # Check time quantum expiration
            if now - self.current_quantum_start >= self.time_quantum:
                # Time quantum expired -> schedule next according to chosen mode
                next_process = self.ready_queue.pop(mode=selection_mode)
                if next_process:
                    self.context_switch(next_process, sim_time=now)
                    self.current_quantum_start = now
                    return next_process

            return self.current_process
        
        # Select next process according to selection_mode
        next_process = self.ready_queue.pop(mode=selection_mode)
        if next_process:
            self.context_switch(next_process, sim_time=now)
            self.current_quantum_start = now
            
        return next_process
