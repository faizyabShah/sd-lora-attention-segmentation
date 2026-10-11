from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from src.attention_capture.memory import CompactAttentionRecorder
from src.segmentation.attention_aggregation import (
    load_step_module_head_maps,
    load_step_module_maps,
)
from src.segmentation.in_memory_dataset import InMemoryAttentionSample
from src.segmentation.ovam_baseline import ovam_attention_map, ovam_mask


MODULE = "down_blocks.0.attentions.0.transformer_blocks.0.attn2"


class AttentionSourceTests(unittest.TestCase):
    def _sample(self, steps):
        return InMemoryAttentionSample(
            sample_id="sample",
            classname="cat",
            prompt="A photograph of a cat",
            token_ids=(49406, 320, 1125, 539, 320, 2368, 49407),
            attention_mask=(1,) * 7,
            timesteps=(999,),
            modules_by_resolution={2: (MODULE,)},
            self_modules_by_resolution={},
            cross_map_key="target_maps",
            target_mask=np.zeros((4, 4), dtype=bool),
            steps=steps,
        )

    def test_compact_cross_maps_match_direct_attention(self):
        torch.manual_seed(7)
        recorder = CompactAttentionRecorder((5,), 7, 5)
        recorder.register_module(MODULE, "cross", 2, 0.5, "test")
        query = torch.randn(2, 2, 4, 3)
        key = torch.randn(2, 2, 7, 3)
        recorder.record(MODULE, "cross", query, key, torch.empty(0), None, 0.5)
        recorder.flush_step(0, 999, torch.empty(0))
        sample = self._sample(recorder.steps)

        expected = (query[1].float() @ key[1].float().transpose(-1, -2) * 0.5).softmax(-1)[:, :, 5]
        heads = load_step_module_head_maps(sample, 0, (5,), (MODULE,))[MODULE][0]
        aggregate = load_step_module_maps(sample, 0, (5,), (MODULE,))[MODULE][0]
        torch.testing.assert_close(heads.flatten(1), expected, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(aggregate.flatten(), expected.mean(0), atol=1e-3, rtol=1e-3)

    def test_ovam_map_uses_unconditional_query_and_prompt_softmax(self):
        torch.manual_seed(11)
        recorder = CompactAttentionRecorder((5,), 7, 5)
        recorder.register_module(MODULE, "cross", 2, 1 / math.sqrt(3), "test")
        query = torch.randn(2, 2, 4, 3)
        key = torch.randn(2, 2, 7, 3)
        scale = 1 / math.sqrt(3)
        recorder.record(MODULE, "cross", query, key, torch.empty(0), None, scale)
        recorder.flush_step(0, 999, torch.empty(0))
        sample = self._sample(recorder.steps)

        expected_heads = (query[0].float() @ key[1].float().transpose(-1, -2) * scale).softmax(-1)[:, :, 5]
        expected = expected_heads.sum(0).reshape(2, 2)
        actual = ovam_attention_map(sample)
        expected_64 = torch.nn.functional.interpolate(
            expected[None, None], size=(64, 64), mode="bilinear", align_corners=False
        )[0, 0]
        torch.testing.assert_close(actual, expected_64, atol=2e-3, rtol=2e-3)
        self.assertEqual(tuple(ovam_mask(sample, (4, 4)).shape), (4, 4))


if __name__ == "__main__":
    unittest.main()
