from __future__ import annotations

from typing import List, Dict, Any
import math
import statistics
import joblib

from .utils import Process


FEATURE_NAMES = [
    "num_procs",
    "mean_remaining",
    "var_remaining",
    "min_remaining",
    "max_remaining",
    "mean_priority",
    "var_priority",
    "min_priority",
    "max_priority",
    "short_jobs_frac",
]


def extract_features_from_processes(procs: List[Process]) -> Dict[str, float]:
    if not procs:
        return {name: 0.0 for name in FEATURE_NAMES}
    remaining = [float(p.remaining_time) for p in procs]
    prios = [int(p.priority) for p in procs]
    mean_r = sum(remaining) / len(remaining)
    short_jobs = len([x for x in remaining if x <= max(1.0, 0.5 * mean_r)])
    feats: Dict[str, float] = {
        "num_procs": float(len(procs)),
        "mean_remaining": mean_r,
        "var_remaining": statistics.pvariance(remaining) if len(remaining) > 1 else 0.0,
        "min_remaining": min(remaining),
        "max_remaining": max(remaining),
        "mean_priority": sum(prios) / len(prios),
        "var_priority": statistics.pvariance(prios) if len(prios) > 1 else 0.0,
        "min_priority": min(prios),
        "max_priority": max(prios),
        "short_jobs_frac": short_jobs / float(len(procs)),
    }
    return feats


def load_model(path: str):
    return joblib.load(path)


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


