from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import fftconvolve
import gpuRIR  # type: ignore

try:
    import webrtcvad  # type: ignore
except ImportError:  # pragma: no cover - runtime-only dependency
    webrtcvad = None


FS = 16_000
AUDIO_LEN = 4 * FS
N_SPK = 2


@dataclass(frozen=True)
class SimulationConfig:
    sample_rate: int = FS
    segment_seconds: float = 4.0
    max_speakers: int = N_SPK
    snr_db: tuple[float, float] = (-5.0, 5.0)
    utterance_sir_db: tuple[float, float] = (-5.0, 15.0)
    noise_sir_db: tuple[float, float] = (0.5, 15.0)
    rt60_s: tuple[float, float] = (0.2, 1.3)
    room_size_min_m: tuple[float, float, float] = (3.0, 3.0, 2.5)
    room_size_max_m: tuple[float, float, float] = (10.0, 8.0, 6.0)
    source_distance_m: tuple[float, float] = (0.5, 10.0)
    azimuth_deg: tuple[float, float] = (0.0, 360.0)
    elevation_deg: tuple[float, float] = (30.0, 150.0)
    min_speaker_gap_deg: float = 10.0
    min_wall_distance_m: float = 0.1
    min_noise_distance_m: float = 2.5
    rir_diffuse_attenuation_db: float = 12.0
    rir_end_attenuation_db: float = 40.0
    vad_frame_ms: int = 20
    vad_aggressiveness: int = 3

    @property
    def segment_samples(self) -> int:
        return int(self.sample_rate * self.segment_seconds)


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64) + 1e-10))


def _trim_or_pad(signal: np.ndarray, length: int) -> np.ndarray:
    signal = signal.astype(np.float32, copy=False)
    if signal.shape[0] >= length:
        return signal[:length].copy()
    return np.pad(signal, (0, length - signal.shape[0])).astype(np.float32)


def _scale_to_db(reference: np.ndarray, signal: np.ndarray, target_db: float) -> float:
    return _rms(reference) / (10.0 ** (target_db / 20.0) * (_rms(signal) + 1e-10))


def _normalize_peak(signal: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_value = np.max(np.abs(signal)) + 1e-10
    return (signal * (peak / max_value)).astype(np.float32)


def compute_vad(
    signal: np.ndarray,
    sample_rate: int = FS,
    frame_ms: int = 20,
    aggressiveness: int = 3,
) -> np.ndarray:
    frame_samples = int(sample_rate * frame_ms / 1000)
    trimmed = _trim_or_pad(signal, len(signal))
    vad = np.zeros_like(trimmed, dtype=np.float32)

    if frame_samples <= 0:
        return vad

    if webrtcvad is not None:
        engine = webrtcvad.Vad(aggressiveness)
        pcm = np.clip(trimmed * 32767.0, -32768.0, 32767.0).astype(np.int16)
        for start in range(0, len(trimmed) - frame_samples + 1, frame_samples):
            frame = pcm[start : start + frame_samples]
            if engine.is_speech(frame.tobytes(), sample_rate):
                vad[start : start + frame_samples] = 1.0
        return vad

    frame_energy = []
    for start in range(0, len(trimmed) - frame_samples + 1, frame_samples):
        frame = trimmed[start : start + frame_samples]
        frame_energy.append(np.mean(frame * frame))
    if not frame_energy:
        return vad

    frame_energy = np.asarray(frame_energy)
    threshold = max(frame_energy.max() * 0.05, 1e-8)
    for idx, energy in enumerate(frame_energy):
        if energy >= threshold:
            start = idx * frame_samples
            vad[start : start + frame_samples] = 1.0
    return vad


def _sample_room_and_rt60(
    rng: np.random.Generator, config: SimulationConfig
) -> tuple[np.ndarray, float]:
    room = np.array(
        [
            rng.uniform(config.room_size_min_m[0], config.room_size_max_m[0]),
            rng.uniform(config.room_size_min_m[1], config.room_size_max_m[1]),
            rng.uniform(config.room_size_min_m[2], config.room_size_max_m[2]),
        ],
        dtype=np.float64,
    )
    rt60 = float(rng.uniform(*config.rt60_s))
    return room, rt60


def _sample_array_center(
    room_size: np.ndarray, rng: np.random.Generator, config: SimulationConfig
) -> np.ndarray:
    z_low = min(max(0.5, config.min_wall_distance_m), room_size[2] - config.min_wall_distance_m)
    z_high = max(z_low, min(room_size[2] - config.min_wall_distance_m, 2.0))
    return np.array(
        [
            rng.uniform(config.min_wall_distance_m, room_size[0] - config.min_wall_distance_m),
            rng.uniform(config.min_wall_distance_m, room_size[1] - config.min_wall_distance_m),
            rng.uniform(z_low, z_high),
        ],
        dtype=np.float64,
    )


def _spherical_to_cartesian(
    distance_m: float, azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.array(
        [
            distance_m * math.sin(elevation) * math.cos(azimuth),
            distance_m * math.sin(elevation) * math.sin(azimuth),
            distance_m * math.cos(elevation),
        ],
        dtype=np.float64,
    )


def _sample_source_position(
    room_size: np.ndarray,
    array_center: np.ndarray,
    used_azimuths: Sequence[float],
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(1024):
        distance = float(rng.uniform(*config.source_distance_m))
        azimuth = float(rng.uniform(*config.azimuth_deg))
        elevation = float(rng.uniform(*config.elevation_deg))

        if any(
            min(abs(azimuth - prev), 360.0 - abs(azimuth - prev))
            < config.min_speaker_gap_deg
            for prev in used_azimuths
        ):
            continue

        relative = _spherical_to_cartesian(distance, azimuth, elevation)
        absolute = array_center + relative
        if np.all(absolute >= config.min_wall_distance_m) and np.all(
            absolute <= room_size - config.min_wall_distance_m
        ):
            polar = np.array([azimuth, elevation, distance], dtype=np.float32)
            return absolute.astype(np.float64), polar

    distance = float(rng.uniform(config.source_distance_m[0], min(2.5, config.source_distance_m[1])))
    azimuth = float(rng.uniform(*config.azimuth_deg))
    elevation = float(rng.uniform(*config.elevation_deg))
    relative = _spherical_to_cartesian(distance, azimuth, elevation)
    absolute = np.clip(
        array_center + relative,
        config.min_wall_distance_m,
        room_size - config.min_wall_distance_m,
    )
    polar = np.array([azimuth, elevation, distance], dtype=np.float32)
    return absolute.astype(np.float64), polar


def _sample_noise_position(
    room_size: np.ndarray,
    array_center: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> np.ndarray:
    for _ in range(1024):
        distance = float(rng.uniform(max(config.min_noise_distance_m, 2.5), config.source_distance_m[1]))
        azimuth = float(rng.uniform(*config.azimuth_deg))
        elevation = float(rng.uniform(*config.elevation_deg))
        absolute = array_center + _spherical_to_cartesian(distance, azimuth, elevation)
        if np.all(absolute >= config.min_wall_distance_m) and np.all(
            absolute <= room_size - config.min_wall_distance_m
        ):
            return absolute.astype(np.float64)

    return np.clip(
        array_center + np.array([config.min_noise_distance_m, 0.0, 0.0], dtype=np.float64),
        config.min_wall_distance_m,
        room_size - config.min_wall_distance_m,
    )


def _render_rir_bank(
    room_size: np.ndarray,
    source_positions: np.ndarray,
    mic_positions: np.ndarray,
    rt60: float,
    config: SimulationConfig,
) -> np.ndarray:
    beta = gpuRIR.beta_SabineEstimation(room_size, rt60)
    t_diff = float(gpuRIR.att2t_SabineEstimator(config.rir_diffuse_attenuation_db, rt60))
    t_max = float(gpuRIR.att2t_SabineEstimator(config.rir_end_attenuation_db, rt60))
    nb_img = gpuRIR.t2n(t_diff, room_size)

    return gpuRIR.simulateRIR(
        room_sz=room_size,
        beta=beta,
        pos_src=source_positions,
        pos_rcv=mic_positions,
        nb_img=nb_img,
        Tmax=t_max,
        fs=config.sample_rate,
        Tdiff=t_diff,
        spkr_pattern="omni",
        mic_pattern="omni",
    )


def _apply_rir_bank(
    signal: np.ndarray, rir_bank: np.ndarray, length: int
) -> np.ndarray:
    return np.stack(
        [
            _trim_or_pad(fftconvolve(signal, rir, mode="full").astype(np.float32), length)
            for rir in rir_bank
        ],
        axis=0,
    )


def simulate_one_sample(
    speeches: Sequence[np.ndarray],
    coherent_noise: np.ndarray,
    mic_coords: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig | None = None,
) -> dict[str, np.ndarray]:
    config = config or SimulationConfig()
    length = config.segment_samples
    num_channels = int(mic_coords.shape[0])
    max_speakers = config.max_speakers
    num_active_speakers = int(rng.integers(1, max_speakers + 1))

    room_size, rt60 = _sample_room_and_rt60(rng, config)
    array_center = _sample_array_center(room_size, rng, config)
    mic_positions = np.clip(
        mic_coords.astype(np.float64) + array_center[None, :],
        config.min_wall_distance_m,
        room_size - config.min_wall_distance_m,
    )

    source_positions = []
    polar_positions = np.zeros((max_speakers, 3), dtype=np.float32)
    used_azimuths: list[float] = []
    for speaker_idx in range(num_active_speakers):
        absolute, polar = _sample_source_position(
            room_size=room_size,
            array_center=array_center,
            used_azimuths=used_azimuths,
            rng=rng,
            config=config,
        )
        source_positions.append(absolute)
        polar_positions[speaker_idx] = polar
        used_azimuths.append(float(polar[0]))

    noise_position = _sample_noise_position(room_size, array_center, rng, config)
    rir_bank = _render_rir_bank(
        room_size=room_size,
        source_positions=np.vstack(source_positions + [noise_position]),
        mic_positions=mic_positions,
        rt60=rt60,
        config=config,
    )

    speech_signals = []
    vads = np.zeros((max_speakers, length), dtype=np.float32)
    for speaker_idx in range(max_speakers):
        if speaker_idx >= num_active_speakers:
            speech_signals.append(np.zeros((num_channels, length), dtype=np.float32))
            continue

        speech = _trim_or_pad(speeches[speaker_idx], length)
        vads[speaker_idx] = compute_vad(
            speech,
            sample_rate=config.sample_rate,
            frame_ms=config.vad_frame_ms,
            aggressiveness=config.vad_aggressiveness,
        )
        speech_signals.append(_apply_rir_bank(speech, rir_bank[speaker_idx], length))

    coherent_noise = _trim_or_pad(coherent_noise, length)
    coherent_noise_mc = _apply_rir_bank(
        coherent_noise, rir_bank[num_active_speakers], length
    )
    white_noise = rng.standard_normal((num_channels, length)).astype(np.float32)

    mixed_speech = speech_signals[0].copy()
    for speaker_idx in range(1, num_active_speakers):
        sir_db = float(rng.uniform(*config.utterance_sir_db))
        scale = _scale_to_db(mixed_speech[0], speech_signals[speaker_idx][0], sir_db)
        mixed_speech += speech_signals[speaker_idx] * scale

    noise_sir_db = float(rng.uniform(*config.noise_sir_db))
    white_scale = _scale_to_db(coherent_noise_mc[0], white_noise[0], noise_sir_db)
    mixed_noise = coherent_noise_mc + white_noise * white_scale

    snr_db = float(rng.uniform(*config.snr_db))
    noise_scale = _scale_to_db(mixed_speech[0], mixed_noise[0], snr_db)
    mixture = _normalize_peak(mixed_speech + mixed_noise * noise_scale)

    return {
        "input_audio": mixture.astype(np.float32),
        "vad": vads.astype(np.float32),
        "mic_coordinate": mic_coords.astype(np.float32),
        "polar_position": polar_positions.astype(np.float32),
        "n_spk": np.int64(num_active_speakers),
        "rt60": np.float32(rt60),
        "room_size": room_size.astype(np.float32),
        "array_center": array_center.astype(np.float32),
    }
