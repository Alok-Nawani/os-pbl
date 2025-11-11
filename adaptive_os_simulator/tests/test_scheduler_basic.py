"""
Tests for core functionality of the scheduler.
"""

import pytest
import time
from ..backend.core import PCB, ReadyQueue, SystemMetrics, ProcessState
from ..backend.schedulers import (
    FCFSScheduler, SJFScheduler, RoundRobinScheduler, 
    PriorityScheduler, SRTFScheduler
)
from ..backend.adaptive_scheduler import AdaptiveScheduler


@pytest.fixture
def sample_pcbs():
    """Create a set of test processes."""
    return [
        PCB(pid=1, arrival_time=0, burst_time=4, priority=1),
        PCB(pid=2, arrival_time=1, burst_time=3, priority=2),
        PCB(pid=3, arrival_time=2, burst_time=1, priority=3),
        PCB(pid=4, arrival_time=3, burst_time=2, priority=2),
        PCB(pid=5, arrival_time=4, burst_time=5, priority=1),
    ]


@pytest.fixture
def ready_queue():
    """Create a ready queue for testing."""
    return ReadyQueue()


@pytest.fixture
def metrics():
    """Create system metrics for testing."""
    return SystemMetrics()


class TestReadyQueue:
    """Test the ReadyQueue implementation."""

    def test_push_pop(self, ready_queue, sample_pcbs):
        """Test basic push and pop operations."""
        # Push all processes
        for pcb in sample_pcbs:
            ready_queue.push(pcb)
        
        # Check queue size
        assert len(ready_queue) == len(sample_pcbs)
        
        # Pop and verify order (by priority)
        priorities = []
        while not ready_queue.is_empty():
            pcb = ready_queue.pop()
            priorities.append(pcb.priority)
        
        # Higher priority should come first
        assert priorities == sorted(priorities, reverse=True)

    def test_remove(self, ready_queue, sample_pcbs):
        """Test removing a specific process."""
        for pcb in sample_pcbs:
            ready_queue.push(pcb)
        
        # Remove middle process
        removed = ready_queue.remove(sample_pcbs[2].pid)
        assert removed.pid == sample_pcbs[2].pid
        assert len(ready_queue) == len(sample_pcbs) - 1

    def test_update_priority(self, ready_queue, sample_pcbs):
        """Test priority updates."""
        ready_queue.push(sample_pcbs[0])
        
        # Update priority and verify
        ready_queue.update_priority(sample_pcbs[0].pid, 10)
        pcb = ready_queue.peek()
        assert pcb.priority == 10


class TestFCFS:
    """Test First Come First Serve scheduler."""

    def test_fcfs_order(self, sample_pcbs):
        """Test FCFS scheduling order."""
        scheduler = FCFSScheduler()
        
        # Add processes in arrival order
        for pcb in sorted(sample_pcbs, key=lambda p: p.arrival_time):
            scheduler.add_process(pcb)
        
        # Verify execution order
        pids = []
        while scheduler.ready_queue:
            next_pcb = scheduler.schedule()
            if next_pcb:
                pids.append(next_pcb.pid)
                next_pcb.remaining_time = 0  # Complete process
        
        # Should match arrival order
        expected = [p.pid for p in sorted(sample_pcbs, key=lambda p: p.arrival_time)]
        assert pids == expected


class TestSJF:
    """Test Shortest Job First scheduler."""

    def test_sjf_order(self, sample_pcbs):
        """Test SJF scheduling order."""
        scheduler = SJFScheduler()
        
        # Add all processes at once
        for pcb in sample_pcbs:
            scheduler.add_process(pcb)
        
        # Verify execution order
        burst_times = []
        while scheduler.ready_queue:
            next_pcb = scheduler.schedule()
            if next_pcb:
                burst_times.append(next_pcb.burst_time)
                next_pcb.remaining_time = 0  # Complete process
        
        # Should be in ascending burst time order
        assert burst_times == sorted(burst_times)


class TestRoundRobin:
    """Test Round Robin scheduler."""

    def test_round_robin_switching(self):
        """Test time quantum based switching."""
        scheduler = RoundRobinScheduler(time_quantum=2.0)
        
        # Create two long processes
        pcbs = [
            PCB(pid=1, arrival_time=0, burst_time=5, priority=1),
            PCB(pid=2, arrival_time=0, burst_time=5, priority=1),
        ]
        
        for pcb in pcbs:
            scheduler.add_process(pcb)
        
        # Run first time slice
        p1 = scheduler.schedule()
        assert p1.pid == 1
        time.sleep(2.1)  # Slightly over quantum
        
        # Should switch to second process
        p2 = scheduler.schedule()
        assert p2.pid == 2


class TestAdaptiveScheduler:
    """Test Adaptive scheduler features."""

    def test_load_based_quantum(self):
        """Test quantum adjustment based on load."""
        scheduler = AdaptiveScheduler(base_quantum=2.0)
        
        # Add many processes to simulate high load
        for i in range(10):
            pcb = PCB(pid=i, arrival_time=0, burst_time=5, priority=1)
            scheduler.add_process(pcb)
        
        # Force a load update
        scheduler.last_load_check = 0
        scheduler.update_load_metrics()
        
        # Quantum should decrease under high load
        assert scheduler.time_quantum < scheduler.base_quantum

    def test_priority_aging(self):
        """Test priority aging for waiting processes."""
        scheduler = AdaptiveScheduler()
        
        # Add a low priority process
        pcb = PCB(pid=1, arrival_time=0, burst_time=3, priority=1)
        scheduler.add_process(pcb)
        
        # Simulate long wait
        pcb.stats.waiting_time = scheduler.metrics.get_avg_waiting_time() * 2
        scheduler.update_priorities()
        
        # Priority should have increased
        updated_pcb = scheduler.ready_queue.peek()
        assert updated_pcb.priority > 1


def test_process_completion_metrics(sample_pcbs):
    """Test metric collection for completed processes."""
    metrics = SystemMetrics()
    
    for pcb in sample_pcbs:
        metrics.add_process(pcb, sim_time=0.0)
        # Simulate some execution time
        time.sleep(0.1)
        pcb.stats.waiting_time = 0.1
        pcb.stats.turnaround_time = 0.2
        pcb.stats.response_time = 0.05
        metrics.process_completed(pcb, completion_time=0.2)
    
    # Verify metrics
    assert metrics.total_processes == len(sample_pcbs)
    assert metrics.completed_processes == len(sample_pcbs)
    assert metrics.get_avg_waiting_time() > 0
    assert metrics.get_avg_turnaround_time() > 0
    assert metrics.get_avg_response_time() > 0
    assert metrics.get_throughput() > 0