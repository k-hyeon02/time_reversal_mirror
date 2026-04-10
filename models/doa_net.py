from __future__ import annotations

import torch
from torch import nn

from features import TimeReversalFeatureConfig, TimeReversalFeatureExtractor


def _group_count(channels: int) -> int:
    for group in (8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


class ResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, dilation: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.GELU()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // 4), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(4, channels // 4), channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.activation(self.norm1(x)))
        x = self.conv2(self.dropout(self.activation(self.norm2(x))))
        x = x * self.channel_gate(x)
        return self.activation(x + residual)


class SharedSpatialEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.GELU(),
        )
        dilations = [1, 2, 4]
        self.blocks = nn.ModuleList(
            [
                ResidualSpatialBlock(
                    base_channels,
                    dropout=dropout,
                    dilation=dilations[block_idx % len(dilations)],
                )
                for block_idx in range(max(1, int(num_blocks)))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


class IterativeTRDOANet(nn.Module):
    def __init__(
        self,
        feature_config: TimeReversalFeatureConfig | None = None,
        max_sources: int = 2,
        base_channels: int = 48,
        cnn_blocks: int = 3,
        gru_hidden_dim: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.1,
        coarse_head_channels: int = 64,
    ) -> None:
        super().__init__()
        self.feature_config = feature_config or TimeReversalFeatureConfig()
        self.max_sources = int(max_sources)
        self.physics = TimeReversalFeatureExtractor(self.feature_config)
        self.encoder = SharedSpatialEncoder(
            in_channels=self.feature_config.num_frequency_bands,
            base_channels=base_channels,
            num_blocks=cnn_blocks,
            dropout=dropout,
        )
        self.gru = nn.GRU(
            input_size=base_channels,
            hidden_size=gru_hidden_dim,
            num_layers=max(1, int(gru_layers)),
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.iteration_context = nn.Linear(gru_hidden_dim, base_channels)
        self.iteration_weight = nn.Linear(gru_hidden_dim, 1)
        self.global_context = nn.Linear(gru_hidden_dim, base_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.GELU(),
            ResidualSpatialBlock(base_channels, dropout=dropout, dilation=1),
        )
        self.coarse_head = nn.Sequential(
            nn.Conv2d(base_channels, coarse_head_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(coarse_head_channels), coarse_head_channels),
            nn.GELU(),
            nn.Conv2d(coarse_head_channels, self.max_sources, kernel_size=1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(base_channels, coarse_head_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(coarse_head_channels), coarse_head_channels),
            nn.GELU(),
            nn.Conv2d(coarse_head_channels, self.max_sources * 2, kernel_size=1),
            nn.Tanh(),
        )
        self.activity_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, self.max_sources),
        )

    @property
    def azimuths_deg(self) -> torch.Tensor:
        return self.physics.azimuths_deg

    @property
    def elevations_deg(self) -> torch.Tensor:
        return self.physics.elevations_deg

    def forward(
        self,
        audio: torch.Tensor,
        mic_coordinates: torch.Tensor,
        vad: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        physics_outputs = self.physics(
            audio=audio,
            mic_coordinates=mic_coordinates,
            vad=vad,
            return_sequence=True,
        )
        maps_sequence = physics_outputs["maps_sequence"]
        batch, iterations, channels, height, width = maps_sequence.shape

        encoded = self.encoder(maps_sequence.reshape(batch * iterations, channels, height, width))
        encoded = encoded.view(batch, iterations, -1, height, width)
        pooled = encoded.mean(dim=(-1, -2))

        gru_outputs, _ = self.gru(pooled)
        context = self.iteration_context(gru_outputs).unsqueeze(-1).unsqueeze(-1)
        conditioned = encoded + context
        iteration_weights = torch.softmax(self.iteration_weight(gru_outputs).squeeze(-1), dim=1)
        fused = (conditioned * iteration_weights[:, :, None, None, None]).sum(dim=1)

        global_context = self.global_context(gru_outputs[:, -1]).unsqueeze(-1).unsqueeze(-1)
        global_context = global_context.expand(-1, -1, height, width)
        fused = self.fusion(torch.cat([fused, global_context], dim=1))

        coarse_logits = self.coarse_head(fused)
        offset_maps = self.offset_head(fused).view(batch, self.max_sources, 2, height, width)
        activity_logits = self.activity_head(gru_outputs[:, -1])

        return {
            "coarse_logits": coarse_logits,
            "offset_maps": offset_maps,
            "activity_logits": activity_logits,
            "iteration_maps": maps_sequence,
            "residual_energies": physics_outputs["residual_energies"],
            "iteration_weights": iteration_weights,
        }
