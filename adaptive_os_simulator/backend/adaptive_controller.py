from __future__ import annotations

from typing import List, Dict, Any, Optional
import statistics
import os
import pandas as pd
import numpy as np
from pathlib import Path

from .core import PCB
from .ml_model import load_model


class Scheduler:
    """Scheduler type constants."""
    RR = "Round Robin"
    FCFS = "FCFS"
    SJF = "SJF"
    SRTF = "SRTF"
    PRIORITY = "Priority"


def rule_based_decider(current_time: float, procs: List[PCB]) -> Dict[str, Any]:
    """Rule-based scheduler decision maker that works with PCB objects."""
    if not procs:
        return {"policy": Scheduler.RR, "reason": "no processes"}

    remaining_times = [p.remaining_time for p in procs]
    priorities = [p.priority for p in procs]
    mean_rt = sum(remaining_times) / len(remaining_times)
    short_jobs = len([t for t in remaining_times if t <= max(1.0, 0.5 * mean_rt)])
    prio_var = statistics.pvariance(priorities) if len(priorities) > 1 else 0.0

    if prio_var > 2.0:
        return {"policy": Scheduler.PRIORITY, "reason": "high priority variance"}
    if short_jobs >= max(2, len(procs) // 2):
        return {"policy": Scheduler.SJF, "reason": "many short jobs"}
    # Small queue, low variance and longer jobs -> FCFS is fine
    if len(procs) <= 3 and mean_rt >= 3.0 and prio_var < 0.5:
        return {"policy": Scheduler.FCFS, "reason": "small queue with similar priorities"}
    if len(procs) >= 6:
        return {"policy": Scheduler.RR, "reason": "many jobs, fair sharing"}
    # default to SRTF for responsiveness on mixed loads
    return {"policy": Scheduler.SRTF, "reason": "default responsiveness"}


def _load_best_model():
    """Automatically load the best performing model based on results CSV."""
    try:
        # Try to find the results CSV
        current_dir = Path(__file__).parent.parent
        results_path = current_dir / "plots" / "optimized_results.csv"
        
        if results_path.exists():
            # Read results and find best model
            df = pd.read_csv(results_path, index_col=0)
            best_model_name = df['accuracy'].idxmax()
            
            # Convert model name to filename
            model_filename = f"optimized_model_{best_model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
            model_path = current_dir.parent / "models" / model_filename
            
            if model_path.exists():
                print(f"✓ Loaded best model: {best_model_name} ({df.loc[best_model_name, 'accuracy']*100:.2f}% accuracy)")
                return load_model(str(model_path)), best_model_name
            else:
                print(f"⚠ Best model file not found: {model_path}")
        
        # Fallback: try gradient boosting (current best)
        fallback_path = current_dir.parent / "models" / "optimized_model_gradient_boosting.joblib"
        if fallback_path.exists():
            print(f"✓ Loaded Gradient Boosting model (fallback)")
            return load_model(str(fallback_path)), "Gradient Boosting"
        
        print("⚠ No trained model found. Using rule-based decider.")
        return None, None
        
    except Exception as e:
        print(f"⚠ Error loading model: {e}. Using rule-based decider.")
        return None, None


# Global model cache
_cached_model = None
_cached_model_name = None


def _get_model():
    """Get cached model or load it."""
    global _cached_model, _cached_model_name
    if _cached_model is None:
        _cached_model, _cached_model_name = _load_best_model()
    return _cached_model, _cached_model_name


def _extract_features(current_time: float, procs: List[PCB]) -> pd.DataFrame:
    """Extract features from current scheduler state for ML prediction."""
    if not procs:
        # Return default features when no processes
        return pd.DataFrame([{
            'avg_burst_time': 1.0,
            'arrival_rate': 1.0,
            'cpu_io_ratio': 1.0,
            'priority_variance': 0.0,
            'queue_length': 0,
            'throughput_req_high': 0,
            'throughput_req_low': 0,
            'throughput_req_medium': 1
        }])
    
    # Calculate features
    burst_times = [p.burst_time for p in procs]
    remaining_times = [p.remaining_time for p in procs]
    priorities = [p.priority for p in procs]
    arrival_times = [p.arrival_time for p in procs]
    
    avg_burst_time = np.mean(burst_times)
    
    # Estimate arrival rate
    if len(arrival_times) > 1:
        time_span = max(arrival_times) - min(arrival_times)
        arrival_rate = len(procs) / max(time_span, 1.0) if time_span > 0 else len(procs)
    else:
        arrival_rate = 1.0
    
    # CPU/IO ratio (simplified: ratio of remaining to original burst)
    cpu_io_ratio = np.mean([r / max(b, 0.1) for r, b in zip(remaining_times, burst_times)])
    
    # Priority variance
    priority_variance = np.var(priorities) if len(priorities) > 1 else 0.0
    
    # Queue length
    queue_length = len(procs)
    
    # Throughput requirement (simplified heuristic)
    avg_remaining = np.mean(remaining_times)
    if avg_remaining < 2.0:
        throughput_req = 'high'
    elif avg_remaining < 5.0:
        throughput_req = 'medium'
    else:
        throughput_req = 'low'
    
    # One-hot encode throughput_req
    features = {
        'avg_burst_time': avg_burst_time,
        'arrival_rate': arrival_rate,
        'cpu_io_ratio': cpu_io_ratio,
        'priority_variance': priority_variance,
        'queue_length': queue_length,
        'throughput_req_high': 1 if throughput_req == 'high' else 0,
        'throughput_req_low': 1 if throughput_req == 'low' else 0,
        'throughput_req_medium': 1 if throughput_req == 'medium' else 0
    }
    
    return pd.DataFrame([features])


def ml_driven_decider(current_time: float, procs: List[PCB]) -> Dict[str, Any]:
    """ML-driven scheduler decision maker using the best trained model."""
    model, model_name = _get_model()
    
    # Fallback to rule-based if model not available
    if model is None:
        return rule_based_decider(current_time, procs)
    
    try:
        # Extract features
        features = _extract_features(current_time, procs)
        
        # Predict best scheduler
        prediction = model.predict(features)[0]
        
        # Map prediction to scheduler constant
        scheduler_map = {
            'FCFS': Scheduler.FCFS,
            'SJF': Scheduler.SJF,
            'RR': Scheduler.RR,
            'Priority': Scheduler.PRIORITY,
            'SRTF': Scheduler.SRTF
        }
        
        policy = scheduler_map.get(prediction, Scheduler.RR)
        reason = f"ML prediction by {model_name}"
        
        return {"policy": policy, "reason": reason}
        
    except Exception as e:
        print(f"⚠ ML prediction error: {e}. Falling back to rule-based.")
        return rule_based_decider(current_time, procs)


def get_model_decider(model_path: str):
    """Create a decider function using a specific model file."""
    try:
        model = load_model(model_path)
        model_name = os.path.basename(model_path).replace('optimized_model_', '').replace('.joblib', '').replace('_', ' ').title()
        print(f"✓ Loaded model from: {model_path}")
        
        def custom_ml_decider(current_time: float, procs: List[PCB]) -> Dict[str, Any]:
            try:
                features = _extract_features(current_time, procs)
                prediction = model.predict(features)[0]
                
                scheduler_map = {
                    'FCFS': Scheduler.FCFS,
                    'SJF': Scheduler.SJF,
                    'RR': Scheduler.RR,
                    'Priority': Scheduler.PRIORITY,
                    'SRTF': Scheduler.SRTF
                }
                
                policy = scheduler_map.get(prediction, Scheduler.RR)
                return {"policy": policy, "reason": f"ML: {model_name}"}
            except Exception as e:
                print(f"⚠ Prediction error: {e}")
                return rule_based_decider(current_time, procs)
        
        return custom_ml_decider
        
    except Exception as e:
        print(f"⚠ Error loading model from {model_path}: {e}")
        return ml_driven_decider


def heuristic_ml_decider(current_time: float, procs: List[PCB]) -> Dict[str, Any]:
    """Hybrid decider: uses ML when available, falls back to rules."""
    return ml_driven_decider(current_time, procs)
