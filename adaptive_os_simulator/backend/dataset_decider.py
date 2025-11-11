"""
Dataset-driven scheduler decision maker.
"""

from typing import Optional
from .core import PCB


def get_dataset_decider():
    """Get dataset decider if available, otherwise return None."""
    # For now, return None - can be extended with ML model loading
    return None
