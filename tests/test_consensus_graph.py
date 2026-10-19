"""Check locked SG2 against independent equations and Claude's complete grid."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import math
from types import SimpleNamespace
import torch
from src.segmentation.consensus_graph import locked_sg2_maps
from scripts.narmyn.attention.sg2_consensus_edges import compute, CONFIGURATIONS


def test_selected_sg2_matches_grid_and_direct_equations():
    generator = torch.Generator().manual_seed(87)
    calls = [(torch.randn(8, 6, 4, generator=generator).half(), torch.randn(8, 6, 4, generator=generator).half()) for _ in range(31)]
    seed = torch.rand(6, generator=generator)
    maps = locked_sg2_maps(calls, seed)
    a = torch.cat([(q.float() @ k.float().transpose(-1, -2)/math.sqrt(4)).softmax(-1) for q, k in calls])
    first = a @ seed
    expected_two = (a @ first[..., None])[..., 0].mean(0)
    mean = a.mean(0)
    rel = mean/(a*a).mean(0).sqrt()
    graph = mean*rel.square()
    graph /= graph.sum(-1, keepdim=True)
    torch.testing.assert_close(maps, torch.stack((first.mean(0), expected_two, graph @ seed)))
    original, _ = compute(SimpleNamespace(sa_qk={'test': calls}), 'test', torch.stack((seed, seed)), torch.tensor([1,1,1,0,0,0]), 'cpu')
    for i, name in enumerate(('W/REF', 'W/P1', 'W/G_cv_g20_n1')):
        torch.testing.assert_close(maps[i], original[CONFIGURATIONS.index(name)])
    print('Locked SG2 matches the full SG2 implementation and independent equations')


if __name__ == '__main__':
    test_selected_sg2_matches_grid_and_direct_equations()
