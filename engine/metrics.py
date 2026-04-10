from __future__ import annotations

import itertools
import time
from typing import Any

import torch
from torch import nn

from labels import angular_distance_deg, decode_slot_predictions, extract_ground_truth_positions


def _all_pairs(
    predictions: list[tuple[float, float, float, int]],
    targets: list[tuple[float, float]],
) -> list[tuple[int, int, float]]:
    return [
        (
            pred_idx,
            target_idx,
            angular_distance_deg(prediction[0], prediction[1], target[0], target[1]),
        )
        for pred_idx, prediction in enumerate(predictions)
        for target_idx, target in enumerate(targets)
    ]


def _best_matching(
    predictions: list[tuple[float, float, float, int]],
    targets: list[tuple[float, float]],
    threshold_deg: float | None = None,
    capped_cost_deg: float | None = None,
) -> list[tuple[int, int, float]]:
    if not predictions or not targets:
        return []

    pred_indices = range(len(predictions))
    target_indices = range(len(targets))
    max_pairs = min(len(predictions), len(targets))

    best_pairs: list[tuple[int, int, float]] = []
    best_pair_count = -1
    best_cost = float("inf")

    for pair_count in range(max_pairs, -1, -1):
        for pred_perm in itertools.permutations(pred_indices, pair_count):
            for target_perm in itertools.permutations(target_indices, pair_count):
                pairs: list[tuple[int, int, float]] = []
                feasible = True
                total_cost = 0.0
                for pred_idx, target_idx in zip(pred_perm, target_perm):
                    error = angular_distance_deg(
                        predictions[pred_idx][0],
                        predictions[pred_idx][1],
                        targets[target_idx][0],
                        targets[target_idx][1],
                    )
                    if threshold_deg is not None and error > threshold_deg:
                        feasible = False
                        break
                    cost = min(error, capped_cost_deg) if capped_cost_deg is not None else error
                    total_cost += cost
                    pairs.append((pred_idx, target_idx, error))
                if not feasible:
                    continue
                if pair_count > best_pair_count or (pair_count == best_pair_count and total_cost < best_cost):
                    best_pair_count = pair_count
                    best_cost = total_cost
                    best_pairs = pairs
        if best_pair_count == pair_count and pair_count >= 0:
            break
    return best_pairs


def _target_errors(
    predictions: list[tuple[float, float, float, int]],
    targets: list[tuple[float, float]],
) -> list[float]:
    if not targets:
        return []
    errors = [180.0 for _ in targets]
    for pred_idx, target_idx, error in _best_matching(predictions, targets):
        del pred_idx
        errors[target_idx] = error
    return errors


def _threshold_f1(
    predictions: list[tuple[float, float, float, int]],
    targets: list[tuple[float, float]],
    threshold_deg: float,
) -> tuple[int, int, int]:
    matches = _best_matching(predictions, targets, threshold_deg=threshold_deg)
    true_positive = len(matches)
    false_positive = max(0, len(predictions) - true_positive)
    false_negative = max(0, len(targets) - true_positive)
    return true_positive, false_positive, false_negative


def _ospa_distance(
    predictions: list[tuple[float, float, float, int]],
    targets: list[tuple[float, float]],
    cutoff_deg: float,
    order: int,
) -> float:
    order = max(1, int(order))
    if not predictions and not targets:
        return 0.0
    if not predictions or not targets:
        return float(cutoff_deg)

    matches = _best_matching(predictions, targets, capped_cost_deg=cutoff_deg)
    matched_cost = sum(min(error, cutoff_deg) ** order for _, _, error in matches)
    cardinality_penalty = (cutoff_deg**order) * abs(len(predictions) - len(targets))
    normalizer = max(len(predictions), len(targets))
    return float(((matched_cost + cardinality_penalty) / normalizer) ** (1.0 / order))


def _rt60_bin_key(rt60: float, bin_edges: tuple[float, ...]) -> str:
    if not bin_edges:
        return "all"
    if rt60 < bin_edges[0]:
        return f"lt_{bin_edges[0]:.1f}"
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if left <= rt60 < right:
            return f"{left:.1f}_{right:.1f}"
    return f"ge_{bin_edges[-1]:.1f}"


def _iteration_recovery(
    iteration_maps: torch.Tensor,
    targets: list[list[tuple[float, float]]],
    azimuths_deg: torch.Tensor,
    elevations_deg: torch.Tensor,
    threshold_deg: float,
) -> dict[str, float]:
    if iteration_maps.numel() == 0:
        return {}

    batch, iterations, _, _, width = iteration_maps.shape
    recovery = [0 for _ in range(iterations)]
    total = max(1, batch)

    for batch_idx in range(batch):
        sample_targets = targets[batch_idx]
        if not sample_targets:
            continue
        for iter_idx in range(iterations):
            dominant_map = iteration_maps[batch_idx, iter_idx].mean(dim=0)
            flat_index = int(torch.argmax(dominant_map).item())
            el_idx = flat_index // width
            az_idx = flat_index % width
            pred = (
                float(azimuths_deg[az_idx].item()),
                float(elevations_deg[el_idx].item()),
            )
            error = min(
                angular_distance_deg(pred[0], pred[1], target[0], target[1]) for target in sample_targets
            )
            if error <= threshold_deg:
                recovery[iter_idx] += 1

    return {
        f"iter_recovery_{iter_idx + 1}": recovery_count / total
        for iter_idx, recovery_count in enumerate(recovery)
    }


def _residual_energy_metrics(residual_energies: torch.Tensor) -> dict[str, float]:
    if residual_energies.numel() == 0:
        return {}
    baseline = residual_energies[:, :1].clamp_min(1e-8)
    reductions = 1.0 - residual_energies[:, 1:] / baseline
    return {
        f"residual_reduction_{iter_idx + 1}": float(reductions[:, iter_idx].mean().item())
        for iter_idx in range(reductions.shape[1])
    }


def compute_doa_metrics(
    outputs: dict[str, torch.Tensor],
    polar_positions: torch.Tensor,
    n_speakers: torch.Tensor,
    azimuths_deg: torch.Tensor,
    elevations_deg: torch.Tensor,
    rt60: torch.Tensor | None,
    rt60_bins_s: tuple[float, ...],
    activity_threshold: float,
    match_threshold_deg: float,
    acc_thresholds_deg: tuple[float, ...],
    ospa_cutoff_deg: float,
    ospa_order: int,
) -> dict[str, Any]:
    predictions = decode_slot_predictions(
        coarse_logits=outputs["coarse_logits"].detach().cpu(),
        offset_maps=outputs["offset_maps"].detach().cpu(),
        activity_logits=outputs["activity_logits"].detach().cpu(),
        azimuths_deg=azimuths_deg.detach().cpu(),
        elevations_deg=elevations_deg.detach().cpu(),
        activity_threshold=activity_threshold,
        max_predictions=outputs["activity_logits"].shape[1],
    )
    targets = extract_ground_truth_positions(polar_positions.detach().cpu(), n_speakers.detach().cpu())
    rt60_values = rt60.detach().cpu().tolist() if rt60 is not None else [None] * len(targets)

    all_errors: list[float] = []
    total_targets = 0
    total_predictions = 0
    hits = {float(threshold): 0 for threshold in acc_thresholds_deg}
    tp = 0
    fp = 0
    fn = 0
    count_accuracy = 0
    ospa_total = 0.0

    for sample_predictions, sample_targets, sample_rt60 in zip(predictions, targets, rt60_values):
        errors = _target_errors(sample_predictions, sample_targets)
        total_targets += len(sample_targets)
        total_predictions += len(sample_predictions)
        all_errors.extend(errors)

        for threshold in acc_thresholds_deg:
            hits[float(threshold)] += sum(error <= float(threshold) for error in errors)

        matched_tp, matched_fp, matched_fn = _threshold_f1(
            sample_predictions,
            sample_targets,
            threshold_deg=match_threshold_deg,
        )
        tp += matched_tp
        fp += matched_fp
        fn += matched_fn
        count_accuracy += int(len(sample_predictions) == len(sample_targets))
        ospa_total += _ospa_distance(
            sample_predictions,
            sample_targets,
            cutoff_deg=ospa_cutoff_deg,
            order=ospa_order,
        )

    rt60_binned: dict[str, dict[str, float]] = {}
    if rt60 is not None:
        for sample_predictions, sample_targets, sample_rt60 in zip(predictions, targets, rt60_values):
            key = _rt60_bin_key(float(sample_rt60), rt60_bins_s)
            group = rt60_binned.setdefault(key, {"errors": [], "tp": 0, "fp": 0, "fn": 0})
            sample_errors = _target_errors(sample_predictions, sample_targets)
            sample_tp, sample_fp, sample_fn = _threshold_f1(
                sample_predictions,
                sample_targets,
                threshold_deg=match_threshold_deg,
            )
            group["errors"].extend(sample_errors)
            group["tp"] += sample_tp
            group["fp"] += sample_fp
            group["fn"] += sample_fn

        for key, group in rt60_binned.items():
            errors = group.pop("errors")
            precision = group["tp"] / max(1, group["tp"] + group["fp"])
            recall = group["tp"] / max(1, group["tp"] + group["fn"])
            f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
            group["mae_deg"] = float(sum(errors) / max(1, len(errors)))
            group["f1_10deg"] = float(f1)
            group["target_count"] = len(errors)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)

    metrics: dict[str, Any] = {
        "mae_deg": float(sum(all_errors) / max(1, len(all_errors))),
        "median_ae_deg": float(torch.tensor(all_errors or [180.0]).median().item()),
        "f1_10deg": float(f1),
        "source_count_acc": float(count_accuracy / max(1, len(targets))),
        "ospa": float(ospa_total / max(1, len(targets))),
        "mean_predicted_sources": float(total_predictions / max(1, len(targets))),
        "mean_target_sources": float(total_targets / max(1, len(targets))),
    }
    for threshold, hit_count in hits.items():
        metrics[f"acc_{int(threshold)}deg"] = float(hit_count / max(1, total_targets))

    if "iteration_maps" in outputs:
        metrics.update(
            _iteration_recovery(
                iteration_maps=outputs["iteration_maps"].detach().cpu(),
                targets=targets,
                azimuths_deg=azimuths_deg.detach().cpu(),
                elevations_deg=elevations_deg.detach().cpu(),
                threshold_deg=match_threshold_deg,
            )
        )
    if "residual_energies" in outputs:
        metrics.update(_residual_energy_metrics(outputs["residual_energies"].detach().cpu()))
    if rt60_binned:
        metrics["rt60_binned"] = rt60_binned

    return metrics


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_model_flops(model: nn.Module, forward_kwargs: dict[str, torch.Tensor]) -> int:
    flops = 0
    handles = []

    def conv1d_hook(module: nn.Conv1d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        batch = int(output.shape[0])
        out_channels = int(output.shape[1])
        out_length = int(output.shape[2])
        kernel = int(module.kernel_size[0])
        in_channels = int(module.in_channels / module.groups)
        flops += 2 * batch * out_channels * out_length * kernel * in_channels

    def conv2d_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        batch = int(output.shape[0])
        out_channels = int(output.shape[1])
        out_height = int(output.shape[2])
        out_width = int(output.shape[3])
        kernel = int(module.kernel_size[0] * module.kernel_size[1])
        in_channels = int(module.in_channels / module.groups)
        flops += 2 * batch * out_channels * out_height * out_width * kernel * in_channels

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        batch = int(output.shape[0]) if output.ndim > 1 else 1
        flops += 2 * batch * int(module.in_features) * int(module.out_features)

    def gru_hook(module: nn.GRU, inputs: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
        nonlocal flops
        sequence = inputs[0]
        batch = int(sequence.shape[0])
        steps = int(sequence.shape[1])
        hidden = int(module.hidden_size)
        directions = 2 if module.bidirectional else 1
        input_size = int(sequence.shape[2])
        total = 0
        for layer_idx in range(int(module.num_layers)):
            layer_input = input_size if layer_idx == 0 else hidden * directions
            total += 2 * batch * steps * directions * 3 * hidden * (layer_input + hidden)
        flops += total

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(conv1d_hook))
        elif isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.GRU):
            handles.append(module.register_forward_hook(gru_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(**forward_kwargs)
    if was_training:
        model.train()

    for handle in handles:
        handle.remove()
    return int(flops)


def measure_inference_latency(
    model: nn.Module,
    forward_kwargs: dict[str, torch.Tensor],
    warmup_steps: int,
    measure_steps: int,
) -> float:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(max(0, int(warmup_steps))):
            model(**forward_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(max(1, int(measure_steps))):
            model(**forward_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    if was_training:
        model.train()
    batch_size = max(1, int(forward_kwargs["audio"].shape[0]))
    return 1000.0 * elapsed / (max(1, int(measure_steps)) * batch_size)
