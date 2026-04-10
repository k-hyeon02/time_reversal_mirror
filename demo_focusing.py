from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from config import load_experiment_config, resolve_dataset_root
from data import SyntheticDOADataset
from data.simulate import SimulationConfig
from features import TimeReversalFeatureConfig, TimeReversalFeatureExtractor
from labels import decode_slot_predictions, extract_ground_truth_positions
from models import IterativeTRDOANet
from viz import plot_focusing_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one focusing map example.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/iterative_tr_doa/demo_focus.png")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    single_speaker_config = replace(config, simulation=replace(config.simulation, max_speakers=1))
    librispeech_root = resolve_dataset_root(single_speaker_config.dataset.librispeech_root, "librispeech")
    ms_snsd_root = resolve_dataset_root(single_speaker_config.dataset.ms_snsd_root, "ms_snsd")

    dataset = SyntheticDOADataset(
        librispeech_root=str(librispeech_root),
        ms_snsd_root=str(ms_snsd_root),
        num_samples=max(args.index + 1, 1),
        profile=single_speaker_config.dataset.profile,
        batch_size=1,
        seed=single_speaker_config.dataset.seed,
        simulation_config=SimulationConfig(**single_speaker_config.simulation.__dict__),
        rotate_arrays=single_speaker_config.dataset.rotate_arrays,
    )

    sample = dataset[args.index]
    extractor = TimeReversalFeatureExtractor(TimeReversalFeatureConfig(**single_speaker_config.feature.__dict__)).to(device)

    feature_outputs = extractor(
        sample["input_audio"].unsqueeze(0).to(device),
        sample["mic_coordinate"].unsqueeze(0).to(device),
        sample["vad"].unsqueeze(0).to(device),
        return_sequence=True,
    )
    focusing_map = feature_outputs["final_map"]
    predicted_positions = None

    if args.checkpoint:
        model = IterativeTRDOANet(
            feature_config=TimeReversalFeatureConfig(**single_speaker_config.feature.__dict__),
            max_sources=single_speaker_config.label.max_sources,
            base_channels=single_speaker_config.model.base_channels,
            cnn_blocks=single_speaker_config.model.cnn_blocks,
            gru_hidden_dim=single_speaker_config.model.gru_hidden_dim,
            gru_layers=single_speaker_config.model.gru_layers,
            dropout=single_speaker_config.model.dropout,
            coarse_head_channels=single_speaker_config.model.coarse_head_channels,
        ).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        with torch.no_grad():
            outputs = model(
                audio=sample["input_audio"].unsqueeze(0).to(device),
                mic_coordinates=sample["mic_coordinate"].unsqueeze(0).to(device),
                vad=sample["vad"].unsqueeze(0).to(device),
            )
        predicted_positions = [
            (prediction[0], prediction[1])
            for prediction in decode_slot_predictions(
                coarse_logits=outputs["coarse_logits"].cpu(),
                offset_maps=outputs["offset_maps"].cpu(),
                activity_logits=outputs["activity_logits"].cpu(),
                azimuths_deg=extractor.azimuths_deg.cpu(),
                elevations_deg=extractor.elevations_deg.cpu(),
                activity_threshold=single_speaker_config.evaluation.activity_threshold,
                max_predictions=single_speaker_config.label.max_sources,
            )[0]
        ]

    true_positions = extract_ground_truth_positions(
        sample["polar_position"].unsqueeze(0),
        sample["n_spk"].unsqueeze(0),
    )[0]

    figure, _ = plot_focusing_map(
        focusing_map=focusing_map.squeeze(0).mean(dim=0).cpu().numpy(),
        azimuths_deg=extractor.azimuths_deg.cpu().numpy(),
        elevations_deg=extractor.elevations_deg.cpu().numpy(),
        true_positions=true_positions,
        predicted_positions=predicted_positions,
        title="Time-Reversal Focusing Demo",
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
