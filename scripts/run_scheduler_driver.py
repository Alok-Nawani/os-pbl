from __future__ import annotations
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from adaptive_os_simulator.backend.schedulers import FCFSScheduler, SJFScheduler, SRTFScheduler, RoundRobinScheduler
from adaptive_os_simulator.backend.core import PCB


def make_pcbs():
    return [
        PCB(pid=1, arrival_time=0.0, burst_time=8.0, priority=5),
        PCB(pid=2, arrival_time=0.1, burst_time=7.0, priority=5),
        PCB(pid=3, arrival_time=0.2, burst_time=6.0, priority=5),
        PCB(pid=4, arrival_time=0.3, burst_time=0.5, priority=5),
        PCB(pid=5, arrival_time=0.4, burst_time=0.5, priority=5),
        PCB(pid=6, arrival_time=0.5, burst_time=0.6, priority=5),
        PCB(pid=7, arrival_time=3.0, burst_time=0.4, priority=5),
        PCB(pid=8, arrival_time=3.2, burst_time=0.4, priority=5),
    ]


def run():
    tick = 0.1
    t = 0.0
    scheduler = FCFSScheduler()
    procs = make_pcbs()
    for p in procs:
        # simulate arrival by adding when arrival_time <= t
        pass

    remaining = list(procs)
    while True:
        # add arrivals
        for p in list(remaining):
            if p.arrival_time <= t:
                scheduler.add_process(p, sim_time=t)
                remaining.remove(p)
        # schedule
        prev = scheduler.current_process
        cur = scheduler.schedule(sim_time=t)
        # advance time
        scheduler.update_metrics(sim_time=t+tick)
        # if running process, decrement
        if scheduler.current_process:
            scheduler.current_process.remaining_time = max(0.0, scheduler.current_process.remaining_time - tick)
            # if completed, call complete
            if scheduler.current_process.remaining_time <= 1e-9:
                scheduler.complete_process(scheduler.current_process, sim_time=t+tick)
                scheduler.current_process = None
        t += tick
        # stop when all processes completed
        if len(scheduler.metrics.process_history) > 0 and all(p.stats.completion_time is not None for p in scheduler.metrics.process_history.values()):
            break
        if t > 100:
            break
    # print per process stats
    print('Completed:', scheduler.metrics.completed_processes)
    for pid, pcb in sorted(scheduler.metrics.process_history.items()):
        print(f'PID {pid}: waiting={pcb.stats.waiting_time:.2f}, turnaround={pcb.stats.turnaround_time:.2f}, completion={pcb.stats.completion_time:.2f}')
    print('avg waiting:', scheduler.metrics.get_avg_waiting_time())
    print('avg turnaround:', scheduler.metrics.get_avg_turnaround_time())

if __name__ == '__main__':
    run()
