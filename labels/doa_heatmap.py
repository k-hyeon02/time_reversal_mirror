from __future__ import annotations

import math

import torch


def circular_distance_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    delta = torch.abs(a - b)
    return torch.minimum(delta, 360.0 - delta)


def wrapped_azimuth_diff_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a - b + 180.0, 360.0) - 180.0


def _mean_step(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 1.0
    steps = values[1:] - values[:-1]
    return float(steps.float().mean().item())


def build_grid_targets(
    polar_positions: torch.Tensor,
    n_speakers: torch.Tensor,
    azimuths_deg: torch.Tensor,
    elevations_deg: torch.Tensor,
    offset_clamp: float = 0.5,
) -> dict[str, torch.Tensor]:
    if polar_positions.ndim == 2:
        polar_positions = polar_positions.unsqueeze(0)
    if n_speakers.ndim == 0:
        n_speakers = n_speakers.unsqueeze(0)

    batch_size, max_speakers, _ = polar_positions.shape
    device = polar_positions.device
    azimuths = azimuths_deg.to(device=device, dtype=polar_positions.dtype)
    elevations = elevations_deg.to(device=device, dtype=polar_positions.dtype)

    speaker_az = polar_positions[..., 0]
    speaker_el = polar_positions[..., 1]

    az_idx = circular_distance_deg(
        azimuths[None, None, :],
        speaker_az[:, :, None],
    ).argmin(dim=-1)
    el_idx = torch.abs(
        elevations[None, None, :] - speaker_el[:, :, None]
    ).argmin(dim=-1)

    az_centers = azimuths[az_idx]
    el_centers = elevations[el_idx]
    az_step = _mean_step(azimuths)
    el_step = _mean_step(elevations)

    az_offset = wrapped_azimuth_diff_deg(speaker_az, az_centers) / max(az_step, 1e-6)
    el_offset = (speaker_el - el_centers) / max(el_step, 1e-6)
    offsets = torch.stack([az_offset, el_offset], dim=-1).clamp(-offset_clamp, offset_clamp)

    activity_mask = (
        torch.arange(max_speakers, device=device)[None, :] < n_speakers[:, None]
    )
    coarse_indices = el_idx * azimuths.numel() + az_idx

    return {
        "coarse_indices": coarse_indices.long(),
        "azimuth_indices": az_idx.long(),
        "elevation_indices": el_idx.long(),
        "offsets": offsets.to(torch.float32),
        "activity_mask": activity_mask.to(torch.float32),
        "target_angles": polar_positions[..., :2].to(torch.float32),
        "target_distances": polar_positions[..., 2].to(torch.float32),
    }


def coarse_cell_to_angles(
    azimuth_indices: torch.Tensor,
    elevation_indices: torch.Tensor,
    offsets: torch.Tensor,
    azimuths_deg: torch.Tensor,
    elevations_deg: torch.Tensor,
) -> torch.Tensor:
    azimuths = azimuths_deg.to(device=offsets.device, dtype=offsets.dtype)
    elevations = elevations_deg.to(device=offsets.device, dtype=offsets.dtype)
    az_step = _mean_step(azimuths)
    el_step = _mean_step(elevations)

    az_center = azimuths[azimuth_indices]
    el_center = elevations[elevation_indices]

    azimuth = torch.remainder(az_center + offsets[..., 0] * az_step, 360.0)
    elevation = torch.clamp(
        el_center + offsets[..., 1] * el_step,
        min=float(elevations[0].item()),
        max=float(elevations[-1].item()),
    )
    return torch.stack([azimuth, elevation], dim=-1)


def decode_slot_predictions(
    coarse_logits: torch.Tensor,
    offset_maps: torch.Tensor,
    activity_logits: torch.Tensor,
    azimuths_deg: torch.Tensor,
    elevations_deg: torch.Tensor,
    activity_threshold: float = 0.5,
    max_predictions: int | None = None,
) -> list[list[tuple[float, float, float, int]]]:
    if coarse_logits.ndim != 4:
        raise ValueError(f"Expected coarse logits with shape (B, K, H, W), got {tuple(coarse_logits.shape)}.")

    batch_size, num_slots, _, width = coarse_logits.shape
    flat_logits = coarse_logits.flatten(start_dim=2)
    activity_probs = torch.sigmoid(activity_logits)
    az_step = _mean_step(azimuths_deg)
    el_min = float(elevations_deg[0].item())
    el_max = float(elevations_deg[-1].item())
    el_step = _mean_step(elevations_deg)

    predictions: list[list[tuple[float, float, float, int]]] = []
    for batch_idx in range(batch_size):
        sample_predictions: list[tuple[float, float, float, int]] = []
        for slot_idx in range(num_slots):
            coarse_score = torch.softmax(flat_logits[batch_idx, slot_idx], dim=0)
            cell_index = int(torch.argmax(coarse_score).item())
            el_idx = cell_index // width
            az_idx = cell_index % width
            offsets = offset_maps[batch_idx, slot_idx, :, el_idx, az_idx]
            azimuth = float(
                torch.remainder(azimuths_deg[az_idx] + offsets[0] * az_step, 360.0).item()
            )
            elevation = float(
                torch.clamp(
                    elevations_deg[el_idx] + offsets[1] * el_step,
                    min=el_min,
                    max=el_max,
                ).item()
            )
            score = float((activity_probs[batch_idx, slot_idx] * coarse_score[cell_index]).item())
            if float(activity_probs[batch_idx, slot_idx].item()) >= activity_threshold:
                sample_predictions.append((azimuth, elevation, score, slot_idx))

        if max_predictions is not None and len(sample_predictions) > max_predictions:
            sample_predictions = sorted(sample_predictions, key=lambda item: item[2], reverse=True)[:max_predictions]
        predictions.append(sample_predictions)
    return predictions


def _to_unit_vector(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    return (
        math.sin(elevation) * math.cos(azimuth),
        math.sin(elevation) * math.sin(azimuth),
        math.cos(elevation),
    )


def angular_distance_deg(
    azimuth_a_deg: float,
    elevation_a_deg: float,
    azimuth_b_deg: float,
    elevation_b_deg: float,
) -> float:
    vec_a = _to_unit_vector(azimuth_a_deg, elevation_a_deg)
    vec_b = _to_unit_vector(azimuth_b_deg, elevation_b_deg)
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def extract_ground_truth_positions(
    polar_positions: torch.Tensor,
    n_speakers: torch.Tensor,
) -> list[list[tuple[float, float]]]:
    if polar_positions.ndim == 2:
        polar_positions = polar_positions.unsqueeze(0)
    if n_speakers.ndim == 0:
        n_speakers = n_speakers.unsqueeze(0)

    positions: list[list[tuple[float, float]]] = []
    for batch_idx in range(polar_positions.shape[0]):
        count = int(n_speakers[batch_idx].item())
        positions.append(
            [
                (
                    float(polar_positions[batch_idx, speaker_idx, 0].item()),
                    float(polar_positions[batch_idx, speaker_idx, 1].item()),
                )
                for speaker_idx in range(count)
            ]
        )
    return positions
