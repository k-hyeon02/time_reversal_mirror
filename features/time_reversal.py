from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AngularGrid:
    azimuths_deg: torch.Tensor
    elevations_deg: torch.Tensor
    direction_vectors: torch.Tensor


@dataclass(frozen=True)
class TimeReversalFeatureConfig:
    sample_rate: int = 16_000
    speed_of_sound: float = 343.0
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    window_type: str = "hann"
    freq_min_hz: float = 300.0
    freq_max_hz: float = 3_500.0
    num_frequency_bands: int = 8
    num_azimuth_bins: int = 72
    num_elevation_bins: int = 13
    azimuth_range_deg: tuple[float, float] = (0.0, 360.0)
    elevation_range_deg: tuple[float, float] = (30.0, 150.0)
    num_iterations: int = 3
    apply_time_reverse: bool = True
    apply_phat: bool = True
    use_vad_mask: bool = True
    log_compression: bool = True
    normalize_per_sample: bool = True
    residual_shrink: float = 0.85
    selection_temperature: float = 0.35
    eps: float = 1e-8


def _build_direction_vectors(
    azimuths_deg: torch.Tensor,
    elevations_deg: torch.Tensor,
) -> torch.Tensor:
    azimuths = torch.deg2rad(azimuths_deg)
    elevations = torch.deg2rad(elevations_deg)
    el_grid, az_grid = torch.meshgrid(elevations, azimuths, indexing="ij")
    x = torch.sin(el_grid) * torch.cos(az_grid)
    y = torch.sin(el_grid) * torch.sin(az_grid)
    z = torch.cos(el_grid)
    return torch.stack([x, y, z], dim=-1).to(torch.float32)


def build_angular_grid(config: TimeReversalFeatureConfig) -> AngularGrid:
    azimuths_deg = torch.linspace(
        config.azimuth_range_deg[0],
        config.azimuth_range_deg[1],
        steps=config.num_azimuth_bins + 1,
        dtype=torch.float32,
    )[:-1]
    elevations_deg = torch.linspace(
        config.elevation_range_deg[0],
        config.elevation_range_deg[1],
        steps=config.num_elevation_bins,
        dtype=torch.float32,
    )
    return AngularGrid(
        azimuths_deg=azimuths_deg,
        elevations_deg=elevations_deg,
        direction_vectors=_build_direction_vectors(azimuths_deg, elevations_deg),
    )


def _build_analysis_window(window_type: str, win_length: int) -> torch.Tensor:
    if window_type == "hann":
        return torch.hann_window(win_length, periodic=True)
    if window_type == "hamming":
        return torch.hamming_window(win_length, periodic=True)
    if window_type in {"rect", "boxcar", "ones"}:
        return torch.ones(win_length, dtype=torch.float32)
    raise ValueError(f"Unsupported window type '{window_type}'.")


class ConvSTFT(nn.Module):
    def __init__(
        self,
        win_length: int,
        hop_length: int,
        fft_length: int,
        window_type: str = "hann",
    ) -> None:
        super().__init__()
        if fft_length < win_length:
            raise ValueError("fft_length must be greater than or equal to win_length.")

        window = _build_analysis_window(window_type, win_length).to(torch.float32)
        freq_bins = fft_length // 2 + 1
        time_index = torch.arange(win_length, dtype=torch.float32)
        freq_index = torch.arange(freq_bins, dtype=torch.float32)[:, None]
        phase = 2.0 * math.pi * freq_index * time_index[None, :] / float(fft_length)

        real = torch.cos(phase) * window[None, :]
        imag = -torch.sin(phase) * window[None, :]
        kernels = torch.cat([real, imag], dim=0).unsqueeze(1)

        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.fft_length = int(fft_length)
        self.register_buffer("weight", kernels, persistent=False)
        self.register_buffer(
            "vad_kernel",
            torch.ones(1, 1, win_length, dtype=torch.float32) / float(win_length),
            persistent=False,
        )

    def _pad_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        samples = int(inputs.shape[-1])
        if samples < self.win_length:
            return F.pad(inputs, (0, self.win_length - samples))
        extra = (samples - self.win_length) % self.hop_length
        pad_size = (self.hop_length - extra) % self.hop_length
        return F.pad(inputs, (0, pad_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"Expected input with shape (B, C, T), got {tuple(inputs.shape)}.")

        batch, channels, samples = inputs.shape
        framed = self._pad_inputs(inputs.to(torch.float32)).reshape(batch * channels, 1, -1)
        transformed = F.conv1d(framed, self.weight, stride=self.hop_length)
        real, imag = torch.chunk(transformed, 2, dim=1)
        spectrum = torch.complex(real, imag)
        return spectrum.view(batch, channels, -1, spectrum.shape[-1])

    def frame_vad(self, vad: torch.Tensor) -> torch.Tensor:
        if vad.ndim != 2:
            raise ValueError(f"Expected VAD with shape (B, T), got {tuple(vad.shape)}.")
        framed = self._pad_inputs(vad.to(torch.float32).unsqueeze(1))
        activity = F.conv1d(framed, self.vad_kernel, stride=self.hop_length)
        return activity.squeeze(1).clamp(0.0, 1.0)


class TimeReversalFeatureExtractor(nn.Module):
    def __init__(self, config: TimeReversalFeatureConfig | None = None) -> None:
        super().__init__()
        self.config = config or TimeReversalFeatureConfig()
        grid = build_angular_grid(self.config)

        self.stft = ConvSTFT(
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            fft_length=self.config.n_fft,
            window_type=self.config.window_type,
        )

        freqs = torch.fft.rfftfreq(
            self.config.n_fft,
            d=1.0 / float(self.config.sample_rate),
        ).to(torch.float32)
        band_mask = (freqs >= self.config.freq_min_hz) & (freqs <= self.config.freq_max_hz)
        if not bool(torch.any(band_mask)):
            raise ValueError("Frequency selection is empty. Check n_fft and feature frequency range.")

        self.register_buffer("azimuths_deg", grid.azimuths_deg)
        self.register_buffer("elevations_deg", grid.elevations_deg)
        self.register_buffer("direction_vectors", grid.direction_vectors)
        self.register_buffer("freqs_hz", freqs[band_mask])
        self.register_buffer("frequency_mask", band_mask, persistent=False)

    @property
    def grid(self) -> AngularGrid:
        return AngularGrid(
            azimuths_deg=self.azimuths_deg,
            elevations_deg=self.elevations_deg,
            direction_vectors=self.direction_vectors,
        )

    def forward(
        self,
        audio: torch.Tensor,
        mic_coordinates: torch.Tensor,
        vad: torch.Tensor | None = None,
        return_sequence: bool = False,
        return_debug: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        result = self.extract(audio=audio, mic_coordinates=mic_coordinates, vad=vad)
        if return_debug or return_sequence:
            return result
        return result["final_map"]

    def extract(
        self,
        audio: torch.Tensor,
        mic_coordinates: torch.Tensor,
        vad: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        if mic_coordinates.ndim == 2:
            mic_coordinates = mic_coordinates.unsqueeze(0)

        if audio.ndim != 3:
            raise ValueError(f"Expected audio with shape (B, C, T), got {tuple(audio.shape)}.")
        if mic_coordinates.ndim != 3:
            raise ValueError(
                f"Expected mic_coordinates with shape (B, C, 3), got {tuple(mic_coordinates.shape)}."
            )
        if audio.shape[:2] != mic_coordinates.shape[:2]:
            raise ValueError(
                f"Audio/microphone channel mismatch: {tuple(audio.shape)} vs {tuple(mic_coordinates.shape)}."
            )

        batch_size = audio.shape[0]
        mic_coordinates = mic_coordinates.to(audio.device, dtype=torch.float32)
        centered_mics = mic_coordinates - mic_coordinates.mean(dim=1, keepdim=True)
        spectrum = self.stft(audio.to(torch.float32))[:, :, self.frequency_mask, :]

        if self.config.apply_time_reverse:
            spectrum = torch.conj(spectrum)
        if self.config.apply_phat:
            spectrum = spectrum / spectrum.abs().clamp_min(self.config.eps)

        frame_weights = self._compute_frame_weights(vad=vad, batch_size=batch_size, frames=spectrum.shape[-1], device=audio.device)
        steering = self._build_steering(centered_mics)

        residual = spectrum
        maps_sequence: list[torch.Tensor] = []
        residual_energies: list[torch.Tensor] = [self._residual_energy(residual)]

        for _ in range(max(1, int(self.config.num_iterations))):
            band_maps, dominant_map = self._focus_once(
                spectrum=residual,
                steering=steering,
                frame_weights=frame_weights,
            )
            maps_sequence.append(self._postprocess_map(band_maps))
            residual = self._subtract_dominant_component(
                spectrum=residual,
                steering=steering,
                dominant_map=dominant_map,
            )
            residual_energies.append(self._residual_energy(residual))

        maps = torch.stack(maps_sequence, dim=1)
        return {
            "maps_sequence": maps,
            "final_map": maps[:, -1],
            "residual_energies": torch.stack(residual_energies, dim=1),
        }

    def _compute_frame_weights(
        self,
        vad: torch.Tensor | None,
        batch_size: int,
        frames: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if vad is None or not self.config.use_vad_mask:
            return None
        if vad.ndim == 3:
            vad = vad.max(dim=1).values
        if vad.ndim == 1:
            vad = vad.unsqueeze(0)
        if vad.ndim != 2:
            raise ValueError(f"Expected VAD with shape (B, T) or (B, S, T), got {tuple(vad.shape)}.")

        framed = self.stft.frame_vad(vad.to(device))
        if framed.shape[-1] < frames:
            framed = F.pad(framed, (0, frames - framed.shape[-1]), value=0.0)
        elif framed.shape[-1] > frames:
            framed = framed[:, :frames]
        if framed.shape[0] != batch_size:
            raise ValueError("VAD batch dimension does not match audio batch dimension.")
        return framed.clamp_min(self.config.eps)

    def _build_steering(self, mic_coordinates: torch.Tensor) -> torch.Tensor:
        delays = -torch.einsum("bcd,ead->bcea", mic_coordinates, self.direction_vectors)
        delays = delays / float(self.config.speed_of_sound)
        phase = (
            2.0
            * math.pi
            * self.freqs_hz[None, None, :, None, None]
            * delays[:, :, None, :, :]
        )
        return torch.exp(1j * phase)

    def _focus_once(
        self,
        spectrum: torch.Tensor,
        steering: torch.Tensor,
        frame_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        focused = torch.einsum("bcft,bcfea->bftea", spectrum, steering)
        power = focused.abs().pow(2.0)

        if frame_weights is not None:
            weighted = power * frame_weights[:, None, :, None, None]
            denom = frame_weights.sum(dim=1).clamp_min(self.config.eps)[:, None, None, None]
            pooled = weighted.sum(dim=2) / denom
        else:
            pooled = power.mean(dim=2)

        band_maps = self._aggregate_frequency_bands(pooled)
        dominant_map = band_maps.mean(dim=1)
        return band_maps, dominant_map

    def _aggregate_frequency_bands(self, pooled_power: torch.Tensor) -> torch.Tensor:
        num_bins = pooled_power.shape[1]
        num_bands = max(1, int(self.config.num_frequency_bands))
        if num_bands == 1:
            return pooled_power.mean(dim=1, keepdim=True)

        edges = torch.linspace(
            0,
            num_bins,
            steps=num_bands + 1,
            device=pooled_power.device,
            dtype=torch.float32,
        ).round().to(torch.long)

        band_maps = []
        for band_idx in range(num_bands):
            start = int(edges[band_idx].item())
            end = min(num_bins, max(start + 1, int(edges[band_idx + 1].item())))
            band_maps.append(pooled_power[:, start:end].mean(dim=1))
        return torch.stack(band_maps, dim=1)

    def _postprocess_map(self, band_maps: torch.Tensor) -> torch.Tensor:
        processed = band_maps
        if self.config.log_compression:
            processed = torch.log1p(processed)
        if self.config.normalize_per_sample:
            flat = processed.flatten(start_dim=1)
            min_values = flat.min(dim=1).values[:, None, None, None]
            max_values = flat.max(dim=1).values[:, None, None, None]
            processed = (processed - min_values) / (max_values - min_values + self.config.eps)
        return processed

    def _subtract_dominant_component(
        self,
        spectrum: torch.Tensor,
        steering: torch.Tensor,
        dominant_map: torch.Tensor,
    ) -> torch.Tensor:
        temperature = max(float(self.config.selection_temperature), self.config.eps)
        selection = torch.softmax(dominant_map.flatten(start_dim=1) / temperature, dim=1)
        selection = selection.view_as(dominant_map)
        steering_mix = (steering * selection[:, None, None, :, :]).sum(dim=(-2, -1))
        beamformed = (spectrum * steering_mix.unsqueeze(-1)).mean(dim=1)
        reconstruction = beamformed.unsqueeze(1) * torch.conj(steering_mix).unsqueeze(-1)
        return spectrum - float(self.config.residual_shrink) * reconstruction

    @staticmethod
    def _residual_energy(spectrum: torch.Tensor) -> torch.Tensor:
        return spectrum.abs().pow(2.0).mean(dim=(1, 2, 3))
