"""Exact SG-combined-v2 operator checks; the live GPU smoke repeats the identities."""
import math

import torch

from src.segmentation import sg_combined_v1, sg_combined_v2
from src.segmentation.sg_combined_v2 import bayes_completion, cap, route_scores


def test_cap_is_one_sided_and_never_redistributes():
    torch.manual_seed(2)
    parent = torch.rand(4, 4)
    proposal = torch.rand(64) * 2
    z, (binding, removed, _) = cap(parent, proposal, 4)
    means = z.reshape(4, 2, 4, 2).mean((1, 3))
    assert bool((means <= parent + 1e-6).all())
    assert bool((z <= proposal + 1e-7).all())
    q = proposal.reshape(4, 2, 4, 2)
    free = q.mean((1, 3)) <= parent
    torch.testing.assert_close(z.reshape(4, 2, 4, 2)[free[:, None, :, None].expand_as(q)],
                               q[free[:, None, :, None].expand_as(q)])
    assert 0 < binding < 1
    torch.testing.assert_close(torch.tensor(removed), (proposal.sum() - z.sum()) / proposal.sum())


def test_uncapped_and_nonbinding_equal_proposal():
    proposal = torch.rand(64)
    z, stats = cap(torch.zeros(4, 4), proposal, 4, scale=math.inf)
    torch.testing.assert_close(z, proposal)
    z, stats = cap(torch.full((4, 4), 10.), proposal, 4)
    torch.testing.assert_close(z, proposal)
    assert stats[0] == 0 and stats[1] == 0


def test_bayes_completion_admits_whole_residual_only_where_p1_ge_b():
    torch.manual_seed(3)
    core = torch.rand(4096)
    graph = torch.softmax(torch.randn(4096, 4096), -1)
    background = torch.rand(4096)
    y = bayes_completion(core, graph, background)
    p1 = sg_combined_v1.score(core)
    delta = (sg_combined_v1.score(graph @ p1) - p1).clamp_min(0)
    torch.testing.assert_close(y, torch.where(p1 >= background, p1 + delta, p1))


def test_route_score_definition():
    fine = torch.tensor([1., .5, 0., .2])
    hier = torch.tensor([.5, .5, .5, 0.])
    bg = torch.tensor([1., 0., 0., .5])
    r, e, d = route_scores(fine, hier, bg)
    assert abs(r - (1 * .5 + .5 * .2) / .7) < 1e-6
    assert abs(e - .7 / 4) < 1e-7 and abs(d - 1.2 / 4) < 1e-7


def test_aggregate_replays_v1_controls_and_uncapped_identity():
    torch.manual_seed(4)
    sa = {s: [(torch.randn(8, s * s, 4), torch.randn(8, s * s, 4)) for _ in range(11)]
          for s in (16, 32, 64)}
    seed = torch.rand(64, 64)
    maps, transient = sg_combined_v2.aggregate(sa, seed)
    old, _ = sg_combined_v1.aggregate(sa, seed)
    for mine, theirs in (("V0", "C0"), ("V1", "C1"), ("V2", "C6")):
        torch.testing.assert_close(maps[mine], old[theirs], atol=1e-6, rtol=1e-5)
    uncapped, _ = sg_combined_v2.aggregate(sa, seed, cap_scale=math.inf)
    for name in ("V3", "V4"):
        torch.testing.assert_close(uncapped[name], maps["V1"], atol=1e-6, rtol=1e-5)
    assert transient["cap"].shape == (2, 3)
    assert 0 < transient["cap"][0, 0] < 1
