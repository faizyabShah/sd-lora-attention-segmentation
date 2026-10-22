"""Small exact operator checks; the live GPU smoke tests the 64x64 path."""
import torch

from src.segmentation.consensus_graph import locked_sg2_maps
from src.segmentation.observation_reliability import reliability_maps
from src.segmentation.sg22_sg23 import aggregate


def fixtures():
    torch.manual_seed(22)
    calls = [(torch.randn(8, 9, 4), torch.randn(8, 9, 4)) for _ in range(31)]
    seed = torch.rand(9)
    return calls, seed


def test_depth_is_per_observation_and_window_isolated():
    calls, seed = fixtures()
    maps = aggregate(calls, seed, 'sg22', expected_pixels=9)
    expected = {w:[[] for _ in range(3)] for w in ('ALL','MID')}
    for t,(q,k) in enumerate(calls):
        a = torch.softmax(q @ k.transpose(-1,-2) / 2, -1)
        z = seed.expand(8,-1)
        for depth in range(3):
            z = (a @ z[...,None])[...,0]
            expected['ALL'][depth].append(z)
            if 10 <= t <= 20: expected['MID'][depth].append(z)
    for w in expected:
        for depth in range(3):
            torch.testing.assert_close(maps[f'{w}-P{depth+1}'],torch.cat(expected[w][depth]).mean(0))
    changed = list(calls)
    changed[0] = (changed[0][0]*3,changed[0][1]*3)
    new = aggregate(changed, seed, 'sg22', expected_pixels=9)
    for depth in (1,2,3):
        torch.testing.assert_close(new[f'MID-P{depth}'],maps[f'MID-P{depth}'])


def test_reliability_matches_locked_sg2_sg11_sg12():
    calls, seed = fixtures()
    maps = aggregate(calls, seed, 'sg23', expected_pixels=9)
    spec = {'sg11':{'aggregators':['trim20'],'rhos':[.8]},
            'sg12':{'fractions':[.5]},'diagnostic':{'near_seed_epsilon':.01}}
    for w, subset in (('ALL',calls),('MID',calls[10:21])):
        sg2 = locked_sg2_maps(subset,seed)
        torch.testing.assert_close(maps[f'{w}-R0'],sg2[0])
        torch.testing.assert_close(maps[f'{w}-SG2'],sg2[2])
        obs=[]
        for q,k in subset:
            a=torch.softmax(q @ k.transpose(-1,-2)/2,-1)
            obs.append(a @ seed)
        trusted,_=reliability_maps(torch.cat(obs),seed,spec)
        torch.testing.assert_close(maps[f'{w}-SG11'],trusted[4])
        torch.testing.assert_close(maps[f'{w}-SG12'],trusted[5])
    torch.testing.assert_close(maps['ALL-P2'],locked_sg2_maps(calls,seed)[1])
