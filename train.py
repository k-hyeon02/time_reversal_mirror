from __future__ import annotations

import argparse
import json
from dataclasses import replace

from config import load_experiment_config
from engine import train_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the time-reversal DOA baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--segment-seconds", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    if args.profile is not None:
        config = replace(config, dataset=replace(config.dataset, profile=args.profile))
    if args.epochs is not None:
        config = replace(config, optim=replace(config.optim, epochs=args.epochs))
    if args.train_samples is not None:
        config = replace(config, dataset=replace(config.dataset, train_samples=args.train_samples))
    if args.val_samples is not None:
        config = replace(config, dataset=replace(config.dataset, val_samples=args.val_samples))
    if args.batch_size is not None:
        config = replace(config, dataset=replace(config.dataset, batch_size=args.batch_size))
    if args.num_workers is not None:
        config = replace(config, dataset=replace(config.dataset, num_workers=args.num_workers))
    if args.max_speakers is not None:
        config = replace(config, simulation=replace(config.simulation, max_speakers=args.max_speakers))
        config = replace(config, label=replace(config.label, max_sources=args.max_speakers))
    if args.segment_seconds is not None:
        config = replace(config, simulation=replace(config.simulation, segment_seconds=args.segment_seconds))
    if args.output_dir is not None:
        config = replace(config, experiment=replace(config.experiment, output_dir=args.output_dir))

    summary = train_experiment(config=config, device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
