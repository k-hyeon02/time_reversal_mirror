from __future__ import annotations

import itertools
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from config import ExperimentConfig, config_to_dict, resolve_dataset_root, write_config_snapshot
from data import SyntheticDOADataset, build_dataloader
from data.simulate import SimulationConfig
from engine.metrics import compute_doa_metrics, count_parameters, estimate_model_flops, measure_inference_latency
from features import TimeReversalFeatureConfig
from labels import build_grid_targets, coarse_cell_to_angles
from models import IterativeTRDOANet


def _build_feature_config(config: ExperimentConfig) -> TimeReversalFeatureConfig:
    return TimeReversalFeatureConfig(**config_to_dict(config)["feature"])


def _build_simulation_config(config: ExperimentConfig) -> SimulationConfig:
    return SimulationConfig(**config_to_dict(config)["simulation"])


def _build_datasets(
    config: ExperimentConfig,
    project_root: str | Path | None = None,
) -> tuple[SyntheticDOADataset, SyntheticDOADataset]:
    librispeech_root = resolve_dataset_root(
        config.dataset.librispeech_root,
        dataset_name="librispeech",
        project_root=project_root,
    )
    ms_snsd_root = resolve_dataset_root(
        config.dataset.ms_snsd_root,
        dataset_name="ms_snsd",
        project_root=project_root,
    )

    train_dataset = SyntheticDOADataset(
        librispeech_root=str(librispeech_root),
        ms_snsd_root=str(ms_snsd_root),
        num_samples=config.dataset.train_samples,
        profile=config.dataset.profile,
        batch_size=config.dataset.batch_size,
        seed=config.dataset.seed,
        simulation_config=_build_simulation_config(config),
        rotate_arrays=config.dataset.rotate_arrays,
    )
    val_dataset = SyntheticDOADataset(
        librispeech_root=str(librispeech_root),
        ms_snsd_root=str(ms_snsd_root),
        num_samples=config.dataset.val_samples,
        profile=config.dataset.profile,
        batch_size=config.dataset.batch_size,
        seed=config.dataset.seed + 10_000,
        simulation_config=_build_simulation_config(config),
        rotate_arrays=config.dataset.rotate_arrays,
    )
    return train_dataset, val_dataset


def _build_model(config: ExperimentConfig) -> IterativeTRDOANet:
    return IterativeTRDOANet(
        feature_config=_build_feature_config(config),
        max_sources=config.label.max_sources,
        base_channels=config.model.base_channels,
        cnn_blocks=config.model.cnn_blocks,
        gru_hidden_dim=config.model.gru_hidden_dim,
        gru_layers=config.model.gru_layers,
        dropout=config.model.dropout,
        coarse_head_channels=config.model.coarse_head_channels,
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _angular_distance_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_az = torch.deg2rad(predicted[..., 0])
    pred_el = torch.deg2rad(predicted[..., 1])
    target_az = torch.deg2rad(target[..., 0])
    target_el = torch.deg2rad(target[..., 1])

    pred_vec = torch.stack(
        [
            torch.sin(pred_el) * torch.cos(pred_az),
            torch.sin(pred_el) * torch.sin(pred_az),
            torch.cos(pred_el),
        ],
        dim=-1,
    )
    target_vec = torch.stack(
        [
            torch.sin(target_el) * torch.cos(target_az),
            torch.sin(target_el) * torch.sin(target_az),
            torch.cos(target_el),
        ],
        dim=-1,
    )
    dot = (pred_vec * target_vec).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.rad2deg(torch.acos(dot)) / 180.0


def _slot_assignments(num_slots: int, num_targets: int, pit_enabled: bool) -> list[tuple[int, ...]]:
    if num_targets <= 0:
        return [tuple()]
    if not pit_enabled:
        return [tuple(range(min(num_slots, num_targets)))]
    return list(itertools.permutations(range(num_slots), num_targets))


def _compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: ExperimentConfig,
    azimuths_deg: torch.Tensor,
    elevations_deg: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = build_grid_targets(
        polar_positions=batch["polar_position"],
        n_speakers=batch["n_spk"],
        azimuths_deg=azimuths_deg,
        elevations_deg=elevations_deg,
        offset_clamp=config.label.offset_clamp,
    )
    coarse_logits = outputs["coarse_logits"]
    offset_maps = outputs["offset_maps"]
    activity_logits = outputs["activity_logits"]
    num_slots = int(activity_logits.shape[1])

    total_losses: list[torch.Tensor] = []
    classification_losses: list[torch.Tensor] = []
    angular_losses: list[torch.Tensor] = []
    activity_losses: list[torch.Tensor] = []
    offset_losses: list[torch.Tensor] = []

    for batch_idx in range(coarse_logits.shape[0]):
        target_count = int(batch["n_spk"][batch_idx].item())
        best: dict[str, torch.Tensor] | None = None

        for assignment in _slot_assignments(num_slots, target_count, config.label.pit_enabled):
            activity_target = torch.zeros(num_slots, device=activity_logits.device)
            if assignment:
                activity_target[torch.tensor(assignment, device=activity_logits.device)] = 1.0
            activity_loss = F.binary_cross_entropy_with_logits(
                activity_logits[batch_idx],
                activity_target,
            )

            classification_loss = torch.zeros((), device=coarse_logits.device)
            angular_loss = torch.zeros((), device=coarse_logits.device)
            offset_loss = torch.zeros((), device=coarse_logits.device)

            for target_idx, slot_idx in enumerate(assignment):
                target_cell = targets["coarse_indices"][batch_idx, target_idx]
                logits = coarse_logits[batch_idx, slot_idx].flatten().unsqueeze(0)
                classification_loss = classification_loss + F.cross_entropy(
                    logits,
                    target_cell.unsqueeze(0),
                )

                az_idx = targets["azimuth_indices"][batch_idx, target_idx]
                el_idx = targets["elevation_indices"][batch_idx, target_idx]
                predicted_offset = offset_maps[batch_idx, slot_idx, :, el_idx, az_idx]
                target_offset = targets["offsets"][batch_idx, target_idx]
                offset_loss = offset_loss + F.smooth_l1_loss(predicted_offset, target_offset)

                predicted_angles = coarse_cell_to_angles(
                    azimuth_indices=az_idx.unsqueeze(0),
                    elevation_indices=el_idx.unsqueeze(0),
                    offsets=predicted_offset.unsqueeze(0),
                    azimuths_deg=azimuths_deg,
                    elevations_deg=elevations_deg,
                )
                target_angles = targets["target_angles"][batch_idx, target_idx].unsqueeze(0)
                angular_loss = angular_loss + _angular_distance_loss(predicted_angles, target_angles).mean()

            normalizer = max(1, target_count)
            classification_loss = classification_loss / normalizer
            angular_loss = angular_loss / normalizer
            offset_loss = offset_loss / normalizer

            total = (
                config.loss.classification_weight * classification_loss
                + config.loss.angular_weight * angular_loss
                + config.loss.activity_weight * activity_loss
                + config.loss.offset_weight * offset_loss
            )
            candidate = {
                "total": total,
                "classification": classification_loss,
                "angular": angular_loss,
                "activity": activity_loss,
                "offset": offset_loss,
            }
            if best is None or float(candidate["total"].detach().item()) < float(best["total"].detach().item()):
                best = candidate

        assert best is not None
        total_losses.append(best["total"])
        classification_losses.append(best["classification"])
        angular_losses.append(best["angular"])
        activity_losses.append(best["activity"])
        offset_losses.append(best["offset"])

    total_loss = torch.stack(total_losses).mean()
    components = {
        "classification_loss": float(torch.stack(classification_losses).mean().detach().item()),
        "angular_loss": float(torch.stack(angular_losses).mean().detach().item()),
        "activity_loss": float(torch.stack(activity_losses).mean().detach().item()),
        "offset_loss": float(torch.stack(offset_losses).mean().detach().item()),
    }
    return total_loss, components


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def _run_epoch(
    *,
    dataloader,
    model: IterativeTRDOANet,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    autocast_enabled = bool(config.optim.amp and device.type == "cuda")
    total_batches = 0
    total_loss = 0.0
    total_components = {
        "classification_loss": 0.0,
        "angular_loss": 0.0,
        "activity_loss": 0.0,
        "offset_loss": 0.0,
    }
    aggregate_metrics: dict[str, float] = {}

    for batch in dataloader:
        batch = _move_batch(batch, device)

        with (
            torch.amp.autocast(device_type=device.type, enabled=autocast_enabled)
            if autocast_enabled
            else nullcontext()
        ):
            outputs = model(
                audio=batch["input_audio"],
                mic_coordinates=batch["mic_coordinate"],
                vad=batch.get("vad"),
            )
            loss, components = _compute_loss(
                outputs=outputs,
                batch=batch,
                config=config,
                azimuths_deg=model.azimuths_deg,
                elevations_deg=model.elevations_deg,
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        metrics = compute_doa_metrics(
            outputs=outputs,
            polar_positions=batch["polar_position"],
            n_speakers=batch["n_spk"],
            azimuths_deg=model.azimuths_deg,
            elevations_deg=model.elevations_deg,
            rt60=batch.get("rt60"),
            rt60_bins_s=config.evaluation.rt60_bins_s,
            activity_threshold=config.evaluation.activity_threshold,
            match_threshold_deg=config.evaluation.match_threshold_deg,
            acc_thresholds_deg=config.evaluation.acc_thresholds_deg,
            ospa_cutoff_deg=config.evaluation.ospa_cutoff_deg,
            ospa_order=config.evaluation.ospa_order,
        )

        total_batches += 1
        total_loss += float(loss.detach().item())
        for key, value in components.items():
            total_components[key] += float(value)
        for key, value in _scalar_metrics(metrics).items():
            aggregate_metrics[key] = aggregate_metrics.get(key, 0.0) + float(value)

    if total_batches == 0:
        return {"loss": 0.0}

    return {
        "loss": total_loss / total_batches,
        **{key: value / total_batches for key, value in total_components.items()},
        **{key: value / total_batches for key, value in aggregate_metrics.items()},
    }


def _save_checkpoint(
    output_dir: Path,
    filename: str,
    epoch: int,
    model: IterativeTRDOANet,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
    config: ExperimentConfig,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config_to_dict(config),
        },
        checkpoint_path,
    )
    return checkpoint_path


def train_experiment(
    config: ExperimentConfig,
    device: str | torch.device | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    run_dir = (root / config.experiment.output_dir).resolve()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(config.dataset.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.dataset.seed)

    train_dataset, val_dataset = _build_datasets(config, project_root=root)
    model = _build_model(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=bool(config.optim.amp and device.type == "cuda"))

    best_val_mae = float("inf")
    history: list[dict[str, Any]] = []
    write_config_snapshot(config, run_dir / "config_snapshot.json")

    for epoch in range(config.optim.epochs):
        train_dataset.set_epoch(epoch)
        train_loader = build_dataloader(
            dataset=train_dataset,
            batch_size=config.dataset.batch_size,
            num_workers=config.dataset.num_workers,
            shuffle=True,
        )
        val_loader = build_dataloader(
            dataset=val_dataset,
            batch_size=config.dataset.batch_size,
            num_workers=config.dataset.num_workers,
            shuffle=False,
        )

        train_stats = _run_epoch(
            dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            device=device,
        )
        with torch.no_grad():
            val_stats = _run_epoch(
                dataloader=val_loader,
                model=model,
                optimizer=None,
                scaler=scaler,
                config=config,
                device=device,
            )

        epoch_record = {
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_stats.items()},
        }
        history.append(epoch_record)

        _save_checkpoint(
            output_dir=run_dir,
            filename="last.pt",
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            metrics=epoch_record,
            config=config,
        )
        val_mae = float(val_stats.get("mae_deg", float("inf")))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            _save_checkpoint(
                output_dir=run_dir,
                filename="best.pt",
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                metrics=epoch_record,
                config=config,
            )

    summary = {
        "device": str(device),
        "output_dir": str(run_dir),
        "history": history,
        "best_val_mae_deg": best_val_mae,
        "parameter_count": count_parameters(model),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    split: str = "val",
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset, val_dataset = _build_datasets(config, project_root=root)
    dataset = train_dataset if split == "train" else val_dataset
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=False,
    )

    model = _build_model(config).to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    total_batches = 0
    loss_sum = 0.0
    component_sum = {
        "classification_loss": 0.0,
        "angular_loss": 0.0,
        "activity_loss": 0.0,
        "offset_loss": 0.0,
    }
    scalar_metric_sum: dict[str, float] = {}
    rt60_bin_sum: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch(batch, device)
            outputs = model(
                audio=batch["input_audio"],
                mic_coordinates=batch["mic_coordinate"],
                vad=batch.get("vad"),
            )
            loss, components = _compute_loss(
                outputs=outputs,
                batch=batch,
                config=config,
                azimuths_deg=model.azimuths_deg,
                elevations_deg=model.elevations_deg,
            )
            batch_metrics = compute_doa_metrics(
                outputs=outputs,
                polar_positions=batch["polar_position"],
                n_speakers=batch["n_spk"],
                azimuths_deg=model.azimuths_deg,
                elevations_deg=model.elevations_deg,
                rt60=batch.get("rt60"),
                rt60_bins_s=config.evaluation.rt60_bins_s,
                activity_threshold=config.evaluation.activity_threshold,
                match_threshold_deg=config.evaluation.match_threshold_deg,
                acc_thresholds_deg=config.evaluation.acc_thresholds_deg,
                ospa_cutoff_deg=config.evaluation.ospa_cutoff_deg,
                ospa_order=config.evaluation.ospa_order,
            )

            total_batches += 1
            loss_sum += float(loss.detach().item())
            for key, value in components.items():
                component_sum[key] += float(value)
            for key, value in _scalar_metrics(batch_metrics).items():
                scalar_metric_sum[key] = scalar_metric_sum.get(key, 0.0) + float(value)
            for key, value in batch_metrics.get("rt60_binned", {}).items():
                group = rt60_bin_sum.setdefault(
                    key,
                    {"tp": 0.0, "fp": 0.0, "fn": 0.0, "weighted_mae": 0.0, "target_count": 0.0},
                )
                group["tp"] += float(value.get("tp", 0.0))
                group["fp"] += float(value.get("fp", 0.0))
                group["fn"] += float(value.get("fn", 0.0))
                target_count = float(value.get("target_count", 0.0))
                group["target_count"] += target_count
                group["weighted_mae"] += float(value.get("mae_deg", 0.0)) * target_count

    metrics = {
        "loss": loss_sum / max(1, total_batches),
        **{key: value / max(1, total_batches) for key, value in component_sum.items()},
        **{key: value / max(1, total_batches) for key, value in scalar_metric_sum.items()},
    }
    if rt60_bin_sum:
        metrics["rt60_binned"] = {}
        for key, value in rt60_bin_sum.items():
            precision = value["tp"] / max(1.0, value["tp"] + value["fp"])
            recall = value["tp"] / max(1.0, value["tp"] + value["fn"])
            f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
            metrics["rt60_binned"][key] = {
                "mae_deg": value["weighted_mae"] / max(1.0, value["target_count"]),
                "f1_10deg": f1,
                "target_count": value["target_count"],
            }

    first_batch = next(iter(dataloader))
    first_batch = _move_batch(first_batch, device)
    forward_kwargs = {
        "audio": first_batch["input_audio"],
        "mic_coordinates": first_batch["mic_coordinate"],
        "vad": first_batch.get("vad"),
    }
    metrics["parameter_count"] = count_parameters(model)
    metrics["flops"] = estimate_model_flops(model, forward_kwargs=forward_kwargs)
    metrics["latency_ms_per_sample"] = measure_inference_latency(
        model,
        forward_kwargs=forward_kwargs,
        warmup_steps=config.evaluation.latency_warmup_steps,
        measure_steps=config.evaluation.latency_measure_steps,
    )
    metrics["device"] = str(device)
    metrics["split"] = split
    return metrics
