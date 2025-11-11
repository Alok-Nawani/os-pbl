from __future__ import annotations

from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass

from .utils import Process, EventLogger
from .simulator import simulate, Scheduler
from .adaptive_controller import heuristic_ml_decider


@dataclass
class KernelConfig:
    policy: str = Scheduler.RR
    time_quantum: float = 2.0
    context_switch_time: float = 0.0
    adaptive_window: float = 2.0


class OSKernel:
    """Lightweight user-space OS kernel simulator wrapper.

    This wraps the existing `simulate` function so we can plug in adaptive
    decision functions and load processes from arbitrary sources.
    """

    def __init__(self, config: KernelConfig | None = None):
        self.config = config or KernelConfig()

    def run(self, processes: List[Process], adaptive: Optional[Callable[[float, List[Process]], Dict[str, Any]]] = None):
        """Run the simulation using the provided adaptive decision function.

        Returns the SimulationResult from `simulate`.
        """
        # Use heuristic decider if none provided
        decider = adaptive or heuristic_ml_decider
        result = simulate(
            processes=processes,
            policy=self.config.policy,
            time_quantum=self.config.time_quantum,
            context_switch_time=self.config.context_switch_time,
            adaptive_policy_fn=decider,
            window=self.config.adaptive_window,
        )
        return result

