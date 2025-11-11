# Adaptive CPU Scheduling Simulator

A sophisticated CPU scheduling simulator that implements both traditional scheduling algorithms and an adaptive scheduling approach. The simulator demonstrates how dynamic adjustments to scheduling parameters can improve system performance under varying loads.

## Features

- Multiple scheduler implementations:
  - First Come First Serve (FCFS)
  - Shortest Job First (SJF)
  - Shortest Remaining Time First (SRTF)
  - Round Robin with configurable quantum
  - Priority-based scheduling
  - Adaptive scheduling with dynamic adjustments

- Adaptive scheduling features:
  - Dynamic time quantum adjustment based on system load
  - Priority aging to prevent starvation
  - Predictive burst time estimation using process history
  - Load-based scheduling parameter optimization

- Comprehensive metrics:
  - Process waiting time
  - Turnaround time
  - Response time
  - CPU utilization
  - Throughput
  - Fairness index

- Visualization tools:
  - Gantt charts for process execution
  - Metric distribution plots
  - Scheduler comparison graphs
  - Real-time performance monitoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Alok-Nawani/Adaptive-CPU-Scheduler.git
cd Adaptive-CPU-Scheduler
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation

1. Run a simple simulation comparing different schedulers:
```bash
python scripts/run_simulation.py
```

2. Use the manual terminal for interactive process creation:
```bash
python scripts/run_manual_terminal.py
```

3. Train the adaptive model on historical data:
```bash
python scripts/train_model.py
```

### Example Code

```python
from adaptive_os_simulator.backend.core import PCB
from adaptive_os_simulator.backend.adaptive_scheduler import AdaptiveScheduler

# Create scheduler
scheduler = AdaptiveScheduler(base_quantum=2.0)

# Add some processes
processes = [
    PCB(pid=1, arrival_time=0, burst_time=4, priority=1),
    PCB(pid=2, arrival_time=1, burst_time=3, priority=2),
    PCB(pid=3, arrival_time=2, burst_time=1, priority=3)
]

for process in processes:
    scheduler.add_process(process)

# Run simulation
while not scheduler.ready_queue.is_empty():
    process = scheduler.schedule()
    # Simulate process execution...
```

## Project Structure

```
adaptive_os_simulator/
├── backend/              # Core implementation
│   ├── core.py          # Basic data structures
│   ├── schedulers.py    # Traditional schedulers
│   ├── adaptive_scheduler.py  # Adaptive scheduler
│   └── visualizer.py    # Plotting and visualization
├── scripts/             # Runner scripts
│   ├── run_simulation.py
│   ├── run_manual_terminal.py
│   └── train_model.py
└── tests/               # Unit tests
```

## Running Tests

Run the test suite:
```bash
pytest tests/
```

Generate coverage report:
```bash
pytest --cov=adaptive_os_simulator tests/
```

## OS Integration Concepts

To integrate this scheduler into a real OS kernel (e.g., Linux):

1. Kernel modifications would be needed in:
   - `/kernel/sched/core.c`
   - `/kernel/sched/fair.c`
   - `/include/linux/sched.h`

2. Key data structures to modify:
   - `struct task_struct`
   - `struct sched_entity`
   - CFS run queue

3. Testing recommendations:
   - Start with isolated CPU testing
   - Use QEMU/KVM for initial validation
   - Gradually expand to multi-core scenarios
   - Profile with `perf` tools

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Academic Usage

When using this simulator for academic projects, consider the following report structure:

1. Abstract
   - Brief overview
   - Key objectives
   - Main results

2. Introduction
   - Background on CPU scheduling
   - Problem statement
   - Objectives

3. Methodology
   - Algorithm descriptions
   - Implementation details
   - Testing approach

4. Results
   - Performance metrics
   - Comparative analysis
   - Visualizations

5. Discussion
   - Findings interpretation
   - Limitations
   - Future improvements

6. Conclusion
   - Summary of results
   - Practical implications
   - Future work

## Acknowledgments

- Referenced scheduling concepts from Operating System Concepts by Silberschatz et al.
- Visualization inspired by Python plotting best practices
- Testing framework based on modern Python testing patterns