
from __future__ import annotations

from typing import List, Optional, Dict, Any
import os
import matplotlib.pyplot as plt

from .utils import Process, EventLogger


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_gantt(processes: List[Process], logger: EventLogger, out_path: Optional[str] = None) -> None:
    # Prepare figure
    fig, ax = plt.subplots(figsize=(12, 3 + 0.2 * max(1, len(processes))))

    # Map pid to color
    pid_to_color = {p.pid: p.color for p in processes}
    pids_order = sorted({seg.get("pid") for seg in logger.timeline if seg.get("pid")}, key=lambda x: x)
    y_positions: Dict[str, int] = {pid: i for i, pid in enumerate(pids_order)}

    # Draw execution bars
    for seg in logger.timeline:
        pid = seg.get("pid")
        if not pid:
            # policy windows or context switches in background
            continue
        start = seg["start"]
        end = seg["end"]
        policy = seg.get("policy", "")
        ax.barh(y_positions[pid], end - start, left=start, color=pid_to_color.get(pid, "#777777"), edgecolor="black", alpha=0.9)

    # Annotate policy switches with explicit phrasing
    for sw in logger.policy_switches:
        t = sw["time"]
        reason = sw.get("reason", "")
        from_p = sw.get("from", "?")
        to_p = sw.get("to", "?")
        ax.axvline(t, color="#444444", linestyle="--", alpha=0.7)
        label = f"switch {from_p} -> {to_p} because {reason}"
        ax.text(
            t,
            len(y_positions) + 0.6,
            label,
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999", alpha=0.8),
        )

    ax.set_yticks([y_positions[pid] for pid in pids_order])
    ax.set_yticklabels(pids_order)
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart with Policy Switches")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()

