from __future__ import annotations

import argparse
import json
from dataclasses import replace

from config import load_experiment_config
from engine import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained time-reversal DOA checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    if args.profile is not None:
        config = replace(config, dataset=replace(config.dataset, profile=args.profile))
    if args.num_samples is not None:
        if args.split == "train":
            config = replace(config, dataset=replace(config.dataset, train_samples=args.num_samples))
        else:
            config = replace(config, dataset=replace(config.dataset, val_samples=args.num_samples))

    metrics = evaluate_checkpoint(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        split=args.split,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
