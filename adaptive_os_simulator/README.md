Adaptive OS Scheduler Simulator
================================

Features
--------
- Mini OS terminal (add/list/run/stats/exit)
- Policies: RR, SJF (non-preemptive), SRTF (preemptive), Priority, FCFS
- Rule-based and ML-like adaptive policy switching 
- Event logging (policy switches, process events)
- Gantt chart visualization with policy annotations

Quickstart
----------
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fixed policy
python scripts/run_simulation.py --mode fixed --policy FCFS --n 20 --out plots/fixed_fcfs.png

# Rule-based adaptive
python scripts/run_simulation.py --mode rule --n 25 --window 2 --out plots/rule.png

# Manual terminal
python scripts/run_manual_terminal.py
```

Train ML Model
--------------
Assumes a CSV with columns:
`num_procs, mean_remaining, var_remaining, min_remaining, max_remaining, mean_priority, var_priority, min_priority, max_priority, short_jobs_frac, policy`

```bash
python scripts/train_model.py --csv adaptive_scheduler_dataset.csv --out models/policy_model.joblib

# Use trained model in adaptive mode
python scripts/run_simulation.py --mode adaptive --model models/policy_model.joblib --n 25 --window 2 --out plots/ml_adaptive.png
```

Manual Terminal Example
-----------------------
```text
> list
No processes yet
> add P1 5 1
Process P1 added: burst=5, priority=1, arrival=0.0
> add P2 3 2
Process P2 added: burst=3, priority=2, arrival=0.0
> run --mode rule --window 2 --out plots/demo.png
Simulation finished. Avg waiting: 4.00, Avg turnaround: 6.50, Throughput: 0.150
> stats
Avg waiting time: 4.000
Avg turnaround time: 6.500
Throughput: 0.150 jobs/s
> exit
```

Project Structure
-----------------
See repository layout under `adaptive_os_simulator/`.

Testing
-------
```bash
pytest -q
```


Kernel demo
-----------
A small demo runner was added that loads the provided `adaptive_scheduler_dataset_10k (4).csv` and runs a lightweight OS kernel using the adaptive scheduler. Run from the repo root:

```bash
# activate your venv first (optional: create one at repo root)
source .venv/bin/activate
python adaptive_os_simulator/scripts/run_kernel.py
```

This will run a short 50-row demo, print summary metrics, and write logs to `kernel_run_logs/`.


