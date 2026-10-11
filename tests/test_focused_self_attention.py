from __future__ import annotations

import unittest

import numpy as np
import torch

from src.segmentation.focused_self_attention import (
    ITERATIONS,
    joint_head_selections,
    joint_timestep_weights,
    propagate_selected_module,
    temporal_weight_schemes,
)
from src.segmentation.in_memory_dataset import InMemoryAttentionSample


MODULE = "up_blocks.3.attentions.2.transformer_blocks.0.attn1"


class FocusedSelfAttentionTests(unittest.TestCase):
    def test_selected_module_aggregation_retains_heads_and_steps(self):
        torch.manual_seed(4)
        steps = []
        queries = []
        keys = []
        for _ in range(2):
            query = torch.randn(1, 2, 4, 3)
            key = torch.randn(1, 2, 4, 3)
            queries.append(query)
            keys.append(key)
            steps.append({f"{MODULE}.q": query, f"{MODULE}.k": key})
        sample = InMemoryAttentionSample(
            sample_id="sample",
            classname="cat",
            prompt="A photograph of a cat",
            token_ids=(1,),
            attention_mask=(1,),
            timesteps=(9, 0),
            modules_by_resolution={},
            self_modules_by_resolution={2: (MODULE,)},
            cross_map_key="ovam_target_map",
            target_mask=np.zeros((4, 4), dtype=bool),
            steps=steps,
        )
        seeds = torch.randn(4, 2)
        actual = propagate_selected_module(
            sample, MODULE, seeds, "cpu", track_seed_column=0, track_step_heads=True
        )

        expected_heads = {iteration: torch.zeros(2, 4) for iteration in ITERATIONS}
        expected_steps = {iteration: torch.zeros(2, 4) for iteration in ITERATIONS}
        expected_all = {iteration: torch.zeros(4, 2) for iteration in ITERATIONS}
        scale = 3**-0.5
        for step_index, (query, key) in enumerate(zip(queries, keys)):
            for head in range(2):
                affinity = torch.softmax(query[0, head] @ key[0, head].T * scale, dim=-1)
                current = seeds
                for iteration in ITERATIONS:
                    current = affinity @ current
                    expected_all[iteration] += current
                    expected_heads[iteration][head] += current[:, 0]
                    expected_steps[iteration][step_index] += current[:, 0]
        for iteration in ITERATIONS:
            expected_all[iteration] /= 4
            expected_heads[iteration] /= 2
            expected_steps[iteration] /= 2
            torch.testing.assert_close(actual.aggregate[iteration], expected_all[iteration])
            torch.testing.assert_close(actual.per_head[iteration], expected_heads[iteration])
            torch.testing.assert_close(actual.per_step[iteration], expected_steps[iteration])
            self.assertEqual(tuple(actual.per_step_head[iteration].shape), (2, 2, 4))
            torch.testing.assert_close(
                actual.per_step_head[iteration].mean(dim=1), expected_steps[iteration]
            )

    def test_temporal_weights_are_normalized_and_late_weighted(self):
        alphas = torch.linspace(0.001, 0.999, 10)
        schemes = temporal_weight_schemes((9, 6, 3, 0), alphas)
        self.assertIn("snr", schemes)
        for weights in schemes.values():
            self.assertEqual(tuple(weights.shape), (4,))
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
            self.assertTrue(torch.all(weights >= 0))
        self.assertGreater(float(schemes["late_linear"][-1]), float(schemes["late_linear"][0]))

    def test_joint_factorial_dimensions(self):
        alphas = torch.linspace(0.001, 0.999, 1000)
        times = joint_timestep_weights(tuple(range(999, 968, -1)), alphas)
        heads = joint_head_selections()
        self.assertEqual(len(times), 15)
        self.assertEqual(len(heads), 13)
        self.assertTrue(all(len(selection) > 0 for selection in heads.values()))
        self.assertTrue(all(abs(float(weights.sum()) - 1.0) < 1e-6 for weights in times.values()))
        expected = len(ITERATIONS) * len(times) * len(heads)
        self.assertEqual(expected * 5 + expected, 4680)


if __name__ == "__main__":
    unittest.main()
