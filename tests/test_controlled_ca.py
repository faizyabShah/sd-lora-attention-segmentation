from __future__ import annotations

import unittest

import numpy as np
import torch

from src.segmentation.attention_aggregation import resize_map
from src.segmentation.controlled_ca import (
    build_time_resolution_maps,
    configuration_names,
    resolution_configurations,
    time_windows,
)
from src.segmentation.in_memory_dataset import InMemoryAttentionSample


class ControlledCAConfigurationTests(unittest.TestCase):
    def test_protocol_configuration_counts(self):
        windows = time_windows()
        resolutions = resolution_configurations()
        self.assertEqual(len(windows), 79)
        self.assertEqual(len(resolutions), 15)
        self.assertEqual(len(configuration_names(windows, resolutions)), 79 + 15 + 79 * 15)
        self.assertEqual(windows["all_00_30"], tuple(range(31)))
        self.assertEqual(resolutions["r8_r16_r32_r64"], (8, 16, 32, 64))

    def test_equal_module_call_weighting(self):
        modules = {2: ("ca2",), 4: ("ca4",)}
        step_values = [
            {
                "ca2.target_maps": torch.arange(4, dtype=torch.float32)[None],
                "ca4.target_maps": torch.arange(16, dtype=torch.float32)[None],
            },
            {
                "ca2.target_maps": (torch.arange(4, dtype=torch.float32) + 10)[None],
                "ca4.target_maps": (torch.arange(16, dtype=torch.float32) + 20)[None],
            },
        ]
        sample = InMemoryAttentionSample(
            sample_id="synthetic",
            classname="thing",
            prompt="a thing",
            token_ids=(0, 1, 2),
            attention_mask=(1, 1, 1),
            timesteps=(1, 0),
            modules_by_resolution=modules,
            self_modules_by_resolution={},
            cross_map_key="target_maps",
            target_mask=np.zeros((4, 4), dtype=bool),
            generated_image=None,
            steps=step_values,
        )
        windows = {"all_00_30": (0, 1), "step_00": (0,)}
        resolutions = {"r2": (2,), "r4": (4,), "r2_r4": (2, 4)}
        names, maps = build_time_resolution_maps(
            sample, (1,), windows, resolutions, working_size=4
        )
        by_name = dict(zip(names, maps))

        r2_0 = resize_map(step_values[0]["ca2.target_maps"][0].reshape(2, 2), 4)
        r2_1 = resize_map(step_values[1]["ca2.target_maps"][0].reshape(2, 2), 4)
        r4_0 = step_values[0]["ca4.target_maps"][0].reshape(4, 4)
        r4_1 = step_values[1]["ca4.target_maps"][0].reshape(4, 4)
        torch.testing.assert_close(by_name["time/all_00_30"], (r2_0 + r2_1 + r4_0 + r4_1) / 4)
        torch.testing.assert_close(by_name["time/step_00"], (r2_0 + r4_0) / 2)
        torch.testing.assert_close(by_name["resolution/r2"], (r2_0 + r2_1) / 2)
        torch.testing.assert_close(by_name["resolution/r4"], (r4_0 + r4_1) / 2)
        torch.testing.assert_close(
            by_name["time_resolution/step_00__r2_r4"], (r2_0 + r4_0) / 2
        )


if __name__ == "__main__":
    unittest.main()
