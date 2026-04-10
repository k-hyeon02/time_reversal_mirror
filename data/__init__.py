from .mic_arrays import RESPEAKER_4CH, NAO_4CH, NAO_ROBOT_12CH
from .simulate import SimulationConfig, simulate_one_sample

try:
    from .dataset import SyntheticDOADataset, ChannelGroupBatchSampler, build_dataloader
except ImportError:  # pragma: no cover - runtime-only dependency
    SyntheticDOADataset = None
    ChannelGroupBatchSampler = None
    build_dataloader = None

__all__ = ["RESPEAKER_4CH", "NAO_4CH", "NAO_ROBOT_12CH", "SimulationConfig", "simulate_one_sample"]

if SyntheticDOADataset is not None:
    __all__.extend(["SyntheticDOADataset", "ChannelGroupBatchSampler", "build_dataloader"])
