"""Exact SG-combined-v3 identities; the live smoke repeats the v2 replay."""
import math

import torch

from src.segmentation import sg_combined_v2, sg_combined_v3
from src.segmentation.sg25_sa import _resize
from src.segmentation.sg_combined_v2 import cap
from src.segmentation.sg_combined_v3 import gt_cap_loss


def _inputs(seed=4):
    torch.manual_seed(seed)
    sa = {s: [(torch.randn(8, s * s, 4), torch.randn(8, s * s, 4)) for _ in range(11)]
          for s in (16, 32, 64)}
    return sa, torch.rand(64, 64), torch.rand(16, 16)


def test_r0_r1_replay_v2_and_w2_with_downsampled_seed_equals_r1():
    sa, seed, coarse = _inputs()
    maps, info = sg_combined_v3.aggregate(sa, seed, coarse)
    old, _ = sg_combined_v2.aggregate(sa, seed)
    torch.testing.assert_close(maps["R0"], old["V0"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(maps["R1"], old["V3"], atol=1e-6, rtol=1e-5)
    c16 = _resize(seed.reshape(-1), 64, 16).reshape(16, 16)
    same, info_same = sg_combined_v3.aggregate(sa, seed, c16)
    torch.testing.assert_close(same["W2"], maps["R1"], atol=1e-6, rtol=1e-5)
    assert abs(info_same["seed"][0] - 1) < 1e-5 and info_same["seed"][1] < 1e-7
    assert info["cap"].shape == (4, 3) and 0 < info["cap"][0, 0] < 1


def test_uncapped_cells_equal_the_plain_hierarchy():
    sa, seed, coarse = _inputs(5)
    uncapped, _ = sg_combined_v3.aggregate(sa, seed, coarse, cap_scale=math.inf)
    v1, _ = sg_combined_v2.aggregate(sa, seed)
    for name in ("R1", "W1", "W3"):
        torch.testing.assert_close(uncapped[name], v1["V1"], atol=1e-6, rtol=1e-5)


def test_headwise_cap_removes_at_least_as_much_and_equals_when_heads_agree():
    torch.manual_seed(6)
    parent = torch.rand(16 * 16)
    heads = torch.rand(8, 32 * 32) * 2
    mean_cap, _ = cap(parent, heads.mean(0), 16)
    head_cap = torch.stack([cap(parent, h, 16)[0] for h in heads]).mean(0)
    group = lambda x: x.reshape(16, 2, 16, 2).sum((1, 3))
    assert bool((group(head_cap) <= group(mean_cap) + 1e-5).all())
    assert float(head_cap.sum()) < float(mean_cap.sum())
    agree = heads[:1].expand(8, -1)  # identical heads lie on the same side of every parent
    torch.testing.assert_close(torch.stack([cap(parent, h, 16)[0] for h in agree]).mean(0),
                               cap(parent, agree.mean(0), 16)[0])


def test_temporal_cap_leaves_first_call_and_only_shrinks():
    sa, seed, coarse = _inputs(7)
    maps, info = sg_combined_v3.aggregate(sa, seed, coarse)
    assert bool((maps["W3"] <= maps["R1"] + 1e-6).all())
    assert 0 <= info["cap"][3, 0] <= 1


def test_gt_cap_loss_definition():
    parent = torch.tensor([[1., 0.]]).expand(2, 2).reshape(-1)  # left groups free, right bind
    q = torch.full((16,), .5)
    z, _ = cap(parent, q, 2)
    gt = torch.zeros(4, 4); gt[:, 2:] = 1  # GT exactly on the binding (right) groups
    gamma, l_fg, l_bg = gt_cap_loss(parent, q, z, 2, gt)
    assert abs(gamma - 1) < 1e-6 and abs(l_fg - 1) < 1e-6 and abs(l_bg) < 1e-6
