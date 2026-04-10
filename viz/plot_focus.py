from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_focusing_map(
    focusing_map: np.ndarray,
    azimuths_deg: np.ndarray,
    elevations_deg: np.ndarray,
    true_positions: Sequence[tuple[float, float]] | None = None,
    predicted_positions: Sequence[tuple[float, float]] | None = None,
    title: str | None = None,
):
    figure, axis = plt.subplots(figsize=(10, 4.5))
    image = axis.imshow(
        focusing_map,
        origin="lower",
        aspect="auto",
        extent=[float(azimuths_deg[0]), float(azimuths_deg[-1]), float(elevations_deg[0]), float(elevations_deg[-1])],
        cmap="magma",
    )
    axis.set_xlabel("Azimuth (deg)")
    axis.set_ylabel("Elevation (deg)")
    axis.set_title(title or "Time-Reversal Focusing Map")
    plt.colorbar(image, ax=axis, label="Normalized energy")

    if true_positions:
        axis.scatter(
            [position[0] for position in true_positions],
            [position[1] for position in true_positions],
            c="cyan",
            marker="o",
            s=70,
            label="Ground Truth",
        )
    if predicted_positions:
        axis.scatter(
            [position[0] for position in predicted_positions],
            [position[1] for position in predicted_positions],
            c="lime",
            marker="x",
            s=80,
            label="Prediction",
        )
    if true_positions or predicted_positions:
        axis.legend(loc="upper right")

    figure.tight_layout()
    return figure, axis
