from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - runtime-only dependency
    torch = None

if torch is not None:
    from features import TimeReversalFeatureConfig, TimeReversalFeatureExtractor
    from labels import build_grid_targets, decode_slot_predictions
    from models import IterativeTRDOANet


@unittest.skipIf(torch is None, "PyTorch is not installed in the current Python environment.")
class TimeReversalPipelineTests(unittest.TestCase):
    def test_feature_extractor_shape(self) -> None:
        extractor = TimeReversalFeatureExtractor(
            TimeReversalFeatureConfig(
                n_fft=128,
                hop_length=32,
                win_length=128,
                num_frequency_bands=4,
                num_azimuth_bins=36,
                num_elevation_bins=7,
            )
        )

        audio = torch.randn(2, 4, 2048)
        coords = torch.tensor(
            [
                [[-0.03, 0.00, 0.00], [0.00, -0.03, 0.00], [0.03, 0.00, 0.00], [0.00, 0.03, 0.00]],
                [[-0.03, 0.00, 0.00], [0.00, -0.03, 0.00], [0.03, 0.00, 0.00], [0.00, 0.03, 0.00]],
            ],
            dtype=torch.float32,
        )

        result = extractor(audio, coords, return_sequence=True)
        self.assertEqual(tuple(result["maps_sequence"].shape), (2, 3, 4, 7, 36))
        self.assertEqual(tuple(result["final_map"].shape), (2, 4, 7, 36))
        self.assertTrue(torch.isfinite(result["final_map"]).all())

    def test_grid_target_assignment_tracks_cell(self) -> None:
        extractor = TimeReversalFeatureExtractor(
            TimeReversalFeatureConfig(num_azimuth_bins=36, num_elevation_bins=7)
        )
        azimuth = float(extractor.azimuths_deg[5].item())
        elevation = float(extractor.elevations_deg[2].item())

        polar_positions = torch.tensor([[[azimuth, elevation, 2.0]]], dtype=torch.float32)
        n_speakers = torch.tensor([1], dtype=torch.long)
        targets = build_grid_targets(
            polar_positions=polar_positions,
            n_speakers=n_speakers,
            azimuths_deg=extractor.azimuths_deg,
            elevations_deg=extractor.elevations_deg,
        )
        self.assertEqual(int(targets["azimuth_indices"][0, 0].item()), 5)
        self.assertEqual(int(targets["elevation_indices"][0, 0].item()), 2)

    def test_model_output_shape(self) -> None:
        model = IterativeTRDOANet(
            feature_config=TimeReversalFeatureConfig(
                n_fft=128,
                hop_length=32,
                win_length=128,
                num_frequency_bands=4,
                num_azimuth_bins=36,
                num_elevation_bins=7,
                num_iterations=2,
            ),
            max_sources=2,
            base_channels=16,
            cnn_blocks=1,
            gru_hidden_dim=32,
            gru_layers=1,
            dropout=0.0,
            coarse_head_channels=16,
        )
        outputs = model(
            audio=torch.randn(3, 4, 2048),
            mic_coordinates=torch.tensor(
                [
                    [[-0.03, 0.00, 0.00], [0.00, -0.03, 0.00], [0.03, 0.00, 0.00], [0.00, 0.03, 0.00]],
                    [[-0.03, 0.00, 0.00], [0.00, -0.03, 0.00], [0.03, 0.00, 0.00], [0.00, 0.03, 0.00]],
                    [[-0.03, 0.00, 0.00], [0.00, -0.03, 0.00], [0.03, 0.00, 0.00], [0.00, 0.03, 0.00]],
                ],
                dtype=torch.float32,
            ),
            vad=torch.ones(3, 1, 2048),
        )
        self.assertEqual(tuple(outputs["coarse_logits"].shape), (3, 2, 7, 36))
        self.assertEqual(tuple(outputs["offset_maps"].shape), (3, 2, 2, 7, 36))
        self.assertEqual(tuple(outputs["activity_logits"].shape), (3, 2))

    def test_slot_decode_uses_activity_threshold(self) -> None:
        coarse = torch.zeros(1, 1, 2, 3)
        coarse[0, 0, 1, 2] = 10.0
        offsets = torch.zeros(1, 1, 2, 2, 3)
        activity = torch.tensor([[10.0]])
        predictions = decode_slot_predictions(
            coarse_logits=coarse,
            offset_maps=offsets,
            activity_logits=activity,
            azimuths_deg=torch.tensor([0.0, 120.0, 240.0]),
            elevations_deg=torch.tensor([30.0, 90.0]),
            activity_threshold=0.5,
            max_predictions=1,
        )
        self.assertEqual(len(predictions[0]), 1)
        self.assertAlmostEqual(predictions[0][0][0], 240.0, places=4)
        self.assertAlmostEqual(predictions[0][0][1], 90.0, places=4)


if __name__ == "__main__":
    unittest.main()
