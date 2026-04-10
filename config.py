from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ExperimentSection:
    name: str = "iterative_tr_doa"
    output_dir: str = "outputs/iterative_tr_doa"


@dataclass(frozen=True)
class DatasetSection:
    librispeech_root: str | None = None
    ms_snsd_root: str | None = None
    train_samples: int = 2048
    val_samples: int = 256
    batch_size: int = 4
    num_workers: int = 0
    seed: int = 1337
    profile: str = "stage1"
    rotate_arrays: bool = True


@dataclass(frozen=True)
class SimulationSection:
    sample_rate: int = 16_000
    segment_seconds: float = 4.0
    max_speakers: int = 2
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


@dataclass(frozen=True)
class FeatureSection:
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


@dataclass(frozen=True)
class LabelSection:
    pit_enabled: bool = True
    max_sources: int = 2
    offset_clamp: float = 0.5


@dataclass(frozen=True)
class ModelSection:
    architecture: str = "iterative_tr_gru"
    base_channels: int = 48
    cnn_blocks: int = 3
    gru_hidden_dim: int = 128
    gru_layers: int = 2
    dropout: float = 0.1
    coarse_head_channels: int = 64


@dataclass(frozen=True)
class LossSection:
    classification_weight: float = 1.0
    angular_weight: float = 1.0
    activity_weight: float = 0.5
    offset_weight: float = 0.5


@dataclass(frozen=True)
class OptimSection:
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    amp: bool = False


@dataclass(frozen=True)
class EvaluationSection:
    activity_threshold: float = 0.5
    match_threshold_deg: float = 10.0
    acc_thresholds_deg: tuple[float, float] = (5.0, 10.0)
    ospa_cutoff_deg: float = 30.0
    ospa_order: int = 1
    rt60_bins_s: tuple[float, float, float] = (0.4, 0.8, 1.2)
    latency_warmup_steps: int = 2
    latency_measure_steps: int = 10


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSection = ExperimentSection()
    dataset: DatasetSection = DatasetSection()
    simulation: SimulationSection = SimulationSection()
    feature: FeatureSection = FeatureSection()
    label: LabelSection = LabelSection()
    model: ModelSection = ModelSection()
    loss: LossSection = LossSection()
    optim: OptimSection = OptimSection()
    evaluation: EvaluationSection = EvaluationSection()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime-only dependency
        raise ImportError(
            "PyYAML is required to load config files. Install it with `pip install pyyaml`."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at '{path}' must contain a mapping at the top level.")
    return data


def _coerce_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _coerce_section(raw: dict[str, Any], section_type: type[T]) -> T:
    kwargs: dict[str, Any] = {}
    defaults = section_type()
    for field in fields(section_type):
        if field.name not in raw:
            continue
        value = raw[field.name]
        default_value = getattr(defaults, field.name)
        if isinstance(default_value, tuple):
            value = _coerce_tuple(value)
        kwargs[field.name] = value
    return section_type(**kwargs)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = _load_yaml(config_path)

    return ExperimentConfig(
        experiment=_coerce_section(raw.get("experiment", {}), ExperimentSection),
        dataset=_coerce_section(raw.get("dataset", {}), DatasetSection),
        simulation=_coerce_section(raw.get("simulation", {}), SimulationSection),
        feature=_coerce_section(raw.get("feature", {}), FeatureSection),
        label=_coerce_section(raw.get("label", {}), LabelSection),
        model=_coerce_section(raw.get("model", {}), ModelSection),
        loss=_coerce_section(raw.get("loss", {}), LossSection),
        optim=_coerce_section(raw.get("optim", {}), OptimSection),
        evaluation=_coerce_section(raw.get("evaluation", {}), EvaluationSection),
    )


def resolve_dataset_root(
    explicit_path: str | None,
    dataset_name: str,
    project_root: str | Path | None = None,
) -> Path:
    if explicit_path:
        resolved = Path(explicit_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Configured {dataset_name} root does not exist: {resolved}")
        return resolved

    root = Path(project_root or Path(__file__).resolve().parent).resolve()
    candidates: list[Path]
    if dataset_name == "librispeech":
        candidates = [
            root / "datasets" / "librispeech",
            root / "datasets" / "LibriSpeech",
            root.parent / "GI-DOAEnet-main" / "datasets" / "librispeech",
            root.parent / "ad2" / "data" / "LibriSpeech",
            root.parent / "ad3" / "data" / "LibriSpeech",
            root.parent / "audio_denoising" / "data" / "LibriSpeech",
            root.parent / "final_project" / "data" / "LibriSpeech",
        ]
    elif dataset_name == "ms_snsd":
        candidates = [
            root / "datasets" / "ms-snsd",
            root / "datasets" / "MS-SNSD",
            root.parent / "GI-DOAEnet-main" / "datasets" / "ms-snsd",
            root.parent / "GI-DOAEnet-main" / "datasets" / "MS-SNSD",
        ]
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not auto-discover {dataset_name}. Set it explicitly in the config file."
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def write_config_snapshot(config: ExperimentConfig, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config_to_dict(config), handle, indent=2)


def with_overrides(config: ExperimentConfig, **overrides: Any) -> ExperimentConfig:
    updated = config
    for section_name, section_overrides in overrides.items():
        if section_overrides:
            current_section = getattr(updated, section_name)
            updated = replace(updated, **{section_name: replace(current_section, **section_overrides)})
    return updated
