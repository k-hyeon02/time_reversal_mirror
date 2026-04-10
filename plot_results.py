from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training history from summary.json.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    history = summary.get("history", [])
    if not history:
        raise ValueError(f"No training history found in {summary_path}.")

    epochs = np.array([record["epoch"] for record in history], dtype=np.int32)
    train_loss = np.array([record["train_loss"] for record in history], dtype=np.float32)
    val_loss = np.array([record["val_loss"] for record in history], dtype=np.float32)
    train_mae = np.array([record["train_mae_deg"] for record in history], dtype=np.float32)
    val_mae = np.array([record["val_mae_deg"] for record in history], dtype=np.float32)
    train_acc10 = np.array([record.get("train_acc_10deg", 0.0) for record in history], dtype=np.float32)
    val_acc10 = np.array([record.get("val_acc_10deg", 0.0) for record in history], dtype=np.float32)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, train_loss, marker="o", label="Train")
    axes[0].plot(epochs, val_loss, marker="o", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_mae, marker="o", label="Train")
    axes[1].plot(epochs, val_mae, marker="o", label="Val")
    axes[1].set_title("Angular MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Degrees")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, train_acc10, marker="o", label="Train")
    axes[2].plot(epochs, val_acc10, marker="o", label="Val")
    axes[2].set_title("ACC@10deg")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.suptitle(summary_path.parent.name)
    figure.tight_layout()

    output_path = Path(args.output) if args.output else summary_path.parent / "training_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
