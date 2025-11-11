#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt6.QtWidgets import QApplication
from adaptive_os_simulator.gui.main_window import MainWindow
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


def run_headless():
    app = QApplication([])
    win = MainWindow()
    # Do not show window to keep test headless; we still use its logic

    # Inject processes
    win.reset_simulation()
    procs = make_pcbs()
    win.processes = procs
    for p in procs:
        win.process_table.update_process(p)
    win.gantt_chart.processes = procs

    # Enable auto scheduler mode
    win.toggle_auto_scheduler_mode(True)

    # Step simulation until completion
    max_steps = 20000
    steps = 0
    while not (all(p.state == p.state.TERMINATED for p in win.processes) and len(win.processes) > 0):
        win.simulation_step()
        steps += 1
        if steps >= max_steps:
            print('Timeout after', steps, 'steps')
            break

    # Read metrics from scheduler.metrics
    metrics = win.scheduler.metrics
    print('Completed processes (metric counter):', metrics.completed_processes)
    print('\nPer-process final stats:')
    for p in win.processes:
        print(f'PID {p.pid}: state={p.state.name}, waiting={p.stats.waiting_time:.2f}, turnaround={p.stats.turnaround_time:.2f}, completion={p.stats.completion_time}')
    # Debug metrics.process_history
    print('\nMetrics.process_history keys:', sorted(list(metrics.process_history.keys())))
    completed_waits = [p.stats.waiting_time for p in metrics.process_history.values() if p.stats.completion_time is not None]
    print('Completed waits (from metrics.process_history):', completed_waits)
    print('Avg waiting (metrics):', metrics.get_avg_waiting_time())
    print('Avg turnaround (metrics):', metrics.get_avg_turnaround_time())
    print('Avg response (metrics):', metrics.get_avg_response_time())

    # Also read values as GUI displays (StatCard labels)
    def parse_label(lbl):
        try:
            return float(lbl.value_label.text().split()[0])
        except Exception:
            return lbl.value_label.text()

    for key, card in win.metrics_panel.cards.items():
        print(key, '->', card.value_label.text(), card.detail_label.text())

    # Clean up
    app.quit()


if __name__ == '__main__':
    run_headless()
