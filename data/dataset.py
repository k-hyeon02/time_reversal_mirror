from __future__ import annotations

import os
from glob import glob
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - runtime-only dependency
    sf = None

try:
    import torchaudio
except ImportError:  # pragma: no cover - runtime-only dependency
    torchaudio = None

from .mic_arrays import get_fixed_array, random_rotate, sample_dynamic_array
from .simulate import AUDIO_LEN, FS, N_SPK, SimulationConfig, simulate_one_sample


PROFILE_SPECS = {
    "stage1": {"array_type": "respeaker", "channel_range": (4, 4)},
    "stage2": {"array_type": "dynamic", "channel_range": (4, 4)},
    "stage3": {"array_type": "dynamic", "channel_range": (4, 12)},
    "respeaker": {"array_type": "respeaker", "channel_range": (4, 4)},
    "nao4": {"array_type": "nao4", "channel_range": (4, 4)},
    "nao12": {"array_type": "nao12", "channel_range": (12, 12)},
    "dynamic4": {"array_type": "dynamic", "channel_range": (4, 4)},
    "dynamic4to12": {"array_type": "dynamic", "channel_range": (4, 12)},
}


def _discover_audio_files(root: str, patterns: Iterable[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob(os.path.join(root, "**", pattern), recursive=True))
    return sorted(set(files))


def _load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    if torchaudio is not None:  # pragma: no cover - runtime-only dependency
        waveform, sample_rate = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0).numpy().astype(np.float32), int(sample_rate)

    if sf is not None:  # pragma: no cover - runtime-only dependency
        waveform, sample_rate = sf.read(path, always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return np.asarray(waveform, dtype=np.float32), int(sample_rate)

    raise ImportError(
        "Neither torchaudio nor soundfile is available. Install one of them to load audio."
    )


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    gcd = np.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd
    return resample_poly(audio, up=up, down=down).astype(np.float32)


def _crop_or_pad(
    audio: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if audio.shape[0] >= length:
        start = int(rng.integers(0, audio.shape[0] - length + 1))
        return audio[start : start + length].astype(np.float32, copy=False)
    return np.pad(audio, (0, length - audio.shape[0])).astype(np.float32)


class SyntheticDOADataset(Dataset):
    def __init__(
        self,
        librispeech_root: str,
        ms_snsd_root: str,
        num_samples: int,
        profile: str = "stage1",
        batch_size: int = 16,
        seed: int = 0,
        simulation_config: SimulationConfig | None = None,
        rotate_arrays: bool = True,
        channel_schedule: Sequence[int] | None = None,
    ) -> None:
        if profile not in PROFILE_SPECS:
            raise ValueError(f"Unknown profile '{profile}'.")

        self.librispeech_root = librispeech_root
        self.ms_snsd_root = ms_snsd_root
        self.num_samples = int(num_samples)
        self.profile = profile
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rotate_arrays = rotate_arrays
        self.simulation_config = simulation_config or SimulationConfig()
        self.channel_schedule = (
            np.asarray(channel_schedule, dtype=np.int32)
            if channel_schedule is not None
            else None
        )
        if self.channel_schedule is not None:
            self.num_samples = int(self.channel_schedule.shape[0])

        self._epoch = 0
        self.speech_files = _discover_audio_files(librispeech_root, ("*.flac", "*.wav"))
        self.noise_files = _discover_audio_files(ms_snsd_root, ("*.wav", "*.flac"))

        if not self.speech_files:
            raise FileNotFoundError(f"No speech files were found under '{librispeech_root}'.")
        if not self.noise_files:
            raise FileNotFoundError(f"No noise files were found under '{ms_snsd_root}'.")

        self.channel_counts = self._assign_channel_counts(self.seed)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self.channel_counts = self._assign_channel_counts(self.seed + self._epoch)

    def set_profile(self, profile: str) -> None:
        if profile not in PROFILE_SPECS:
            raise ValueError(f"Unknown profile '{profile}'.")
        self.profile = profile
        self.channel_counts = self._assign_channel_counts(self.seed + self._epoch)

    def _assign_channel_counts(self, seed: int) -> np.ndarray:
        if self.channel_schedule is not None:
            return self.channel_schedule.copy()

        rng = np.random.default_rng(seed)
        min_channels, max_channels = PROFILE_SPECS[self.profile]["channel_range"]
        if min_channels == max_channels:
            return np.full(self.num_samples, min_channels, dtype=np.int32)

        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        batch_counts = rng.integers(min_channels, max_channels + 1, size=num_batches)
        return np.repeat(batch_counts, self.batch_size)[: self.num_samples].astype(np.int32)

    def _sample_mic_coordinates(
        self, num_channels: int, rng: np.random.Generator
    ) -> np.ndarray:
        array_type = PROFILE_SPECS[self.profile]["array_type"]
        if array_type == "dynamic":
            coords = sample_dynamic_array(num_channels, rng=rng)
        else:
            coords = get_fixed_array(array_type)

        if self.rotate_arrays:
            coords = random_rotate(coords, rng)
        return coords.astype(np.float32)

    def _sample_audio(
        self, file_path: str, rng: np.random.Generator
    ) -> np.ndarray:
        audio, sample_rate = _load_audio_mono(file_path)
        audio = _resample_audio(audio, sample_rate, self.simulation_config.sample_rate)
        return _crop_or_pad(audio, self.simulation_config.segment_samples, rng)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        seed = (self.seed * 1_000_003 + self._epoch * self.num_samples + index) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        num_channels = int(self.channel_counts[index])

        speech_indices = rng.choice(len(self.speech_files), size=N_SPK, replace=False)
        speeches = [self._sample_audio(self.speech_files[i], rng) for i in speech_indices]
        noise_index = int(rng.integers(0, len(self.noise_files)))
        noise = self._sample_audio(self.noise_files[noise_index], rng)
        mic_coordinates = self._sample_mic_coordinates(num_channels, rng)

        sample = simulate_one_sample(
            speeches=speeches,
            coherent_noise=noise,
            mic_coords=mic_coordinates,
            rng=rng,
            config=self.simulation_config,
        )

        return {
            "input_audio": torch.from_numpy(sample["input_audio"]),
            "vad": torch.from_numpy(sample["vad"]),
            "mic_coordinate": torch.from_numpy(sample["mic_coordinate"]),
            "polar_position": torch.from_numpy(sample["polar_position"]),
            "n_spk": torch.tensor(sample["n_spk"], dtype=torch.long),
            "rt60": torch.tensor(sample["rt60"], dtype=torch.float32),
            "room_size": torch.from_numpy(sample["room_size"]),
            "array_center": torch.from_numpy(sample["array_center"]),
        }


class ChannelGroupBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        channel_counts: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        self.channel_counts = np.asarray(channel_counts, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        groups: dict[int, list[int]] = {}
        for index, count in enumerate(self.channel_counts.tolist()):
            groups.setdefault(int(count), []).append(index)

        batches: list[list[int]] = []
        for indices in groups.values():
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._batches.copy()
        if self.shuffle:
            np.random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)


def build_dataloader(
    dataset: SyntheticDOADataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    sampler = ChannelGroupBatchSampler(
        channel_counts=dataset.channel_counts,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
