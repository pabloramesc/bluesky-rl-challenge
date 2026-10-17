"""Persistent run metadata, raw episode histories, and static performance plots."""

import csv
import json
from time import perf_counter

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeHistory(BaseCallback):
    """Record raw Monitor episode returns and produce a performance plot.

    Attributes:
        directory: Existing output directory for episode CSV and plot files.
        rows: Completed episode records collected during this training segment.
    """

    def __init__(self, directory):
        """Initialize episode tracking and write the CSV header.

        Args:
            directory: Existing pathlib.Path for run outputs.
        """
        super().__init__()
        self.directory = directory
        self.rows = []
        self.started = perf_counter()
        self.fields = [
            "episode",
            "timesteps",
            "env_index",
            "return",
            "length",
            "elapsed_seconds",
        ]
        with (self.directory / "episodes.csv").open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.fields).writeheader()

    def _on_step(self):
        """Append completed episodes from the current SB3 callback infos.

        Returns:
            True to continue training.
        """
        for index, info in enumerate(self.locals["infos"]):
            if "episode" not in info:
                continue
            episode = info["episode"]
            row = {
                "episode": len(self.rows) + 1,
                "timesteps": self.num_timesteps,
                "env_index": index,
                "return": float(episode["r"]),
                "length": int(episode["l"]),
                "elapsed_seconds": perf_counter() - self.started,
            }
            with (self.directory / "episodes.csv").open("a", newline="") as stream:
                csv.DictWriter(stream, fieldnames=self.fields).writerow(row)
            self.rows.append(row)
        return True

    def plot(self):
        """Save return, rolling-mean, and episode-length curves to performance.png."""
        figure = Figure(figsize=(10, 7), layout="constrained")
        FigureCanvasAgg(figure)
        reward_axis, length_axis = figure.subplots(2, 1, sharex=True)
        if self.rows:
            episodes = np.arange(1, len(self.rows) + 1)
            returns = np.array([row["return"] for row in self.rows])
            reward_axis.plot(episodes, returns, alpha=0.35, label="Raw episode return")
            window = min(20, len(returns))
            rolling = np.convolve(returns, np.ones(window) / window, mode="valid")
            reward_axis.plot(
                episodes[window - 1 :], rolling, label=f"{window}-episode mean"
            )
            reward_axis.legend()
            length_axis.plot(episodes, [row["length"] for row in self.rows])
        else:
            reward_axis.text(0.5, 0.5, "No completed training episodes", ha="center")
        reward_axis.set_ylabel("Return (raw reward)")
        length_axis.set_ylabel("Episode length (steps)")
        length_axis.set_xlabel("Completed episode (per agent for shared policies)")
        figure.savefig(self.directory / "performance.png", dpi=150)


def write_metadata(directory, metadata):
    """Atomically replace run.json so interrupted writes retain valid JSON.

    Args:
        directory: Existing pathlib.Path for run outputs.
        metadata: Report mapping; unsupported JSON values are converted to strings.
    """
    temporary = directory / "run.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    temporary.replace(directory / "run.json")
