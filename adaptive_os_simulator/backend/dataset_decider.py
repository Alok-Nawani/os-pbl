"""
Dataset-driven scheduler decision maker.
"""

from typing import Optional, Dict, Any, List
from .core import PCB
from .adaptive_controller import ml_driven_decider


def get_dataset_decider():
    """Get dataset decider - returns ML-driven decider wrapper."""
    
    class MLDeciderWrapper:
        """Wrapper to provide .decide() method interface."""
        
        def decide(self, current_time: float, procs: List[PCB]) -> Dict[str, Any]:
            """Make scheduling decision using ML model."""
            return ml_driven_decider(current_time, procs)
    
    return MLDeciderWrapper()
