"""Independent numerical and storage checks for SG11/SG12/SG-Diag."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.segmentation.observation_consensus import consensus_maps
from src.segmentation.observation_reliability import configuration_names, reliability_maps
from src.segmentation.propagation_diagnostics import ObservationDiagnosticWriter, histogram_auc, change_histograms

SPEC = json.loads(Path('configs/sg11_sg12_observation_reliability.json').read_text())


def test_matches_sg8_and_independent_cosine_selection():
    generator = torch.Generator().manual_seed(31)
    x, seed = torch.rand(248, 41, generator=generator), torch.rand(41, generator=generator)
    maps, d = reliability_maps(x, seed, SPEC)
    old, _ = consensus_maps(x[:, None], seed[None], (), (.8,))
    # SG8 order REF, trim10, trim20, median, C/rho.80, D/rho.80.
    torch.testing.assert_close(maps[:4], old[[0, 4, 2, 3]])
    scores = torch.stack([torch.stack([F.cosine_similarity(x[i], x[j], dim=0)
                                        for j in range(248) if j != i]).mean() for i in range(248)])
    torch.testing.assert_close(d['reliability'], scores)
    order = torch.argsort(scores, descending=True, stable=True)
    for j, k in enumerate((124, 174, 199, 224)):
        assert int(d['selected'][j].sum()) == k
        torch.testing.assert_close(maps[10+j], x[order[:k]].mean(0))
    assert len(configuration_names(SPEC)) == len(maps) == 14


def test_consensus_rejects_disagreement_and_conventional_even_median():
    seed = torch.tensor([.5, .5])
    x = torch.tensor([[.1, .9]]*5+[[.9, .9]]*5)
    maps, _ = reliability_maps(x, seed, SPEC)
    names = configuration_names(SPEC)
    for name in names:
        if '/SG11/' in name:
            torch.testing.assert_close(maps[names.index(name)], torch.tensor([.5, .9]))
    torch.testing.assert_close(maps[names.index('W/A/median')], torch.tensor([.5, .9]))


def test_histogram_auc_matches_pairwise_ranks_and_threshold_precision():
    scores = np.array([0., 0., 1., 1., 2., 2., 3., 3.])
    labels = np.array([1, 0, 0, 0, 1, 0, 1, 1])
    p = np.bincount(scores[labels == 1].astype(int), minlength=4)
    n = np.bincount(scores[labels == 0].astype(int), minlength=4)
    result = histogram_auc(p, n)
    comparisons = [(a > b) + .5*(a == b) for a in scores[labels == 1] for b in scores[labels == 0]]
    assert np.isclose(result['roc_auc'], np.mean(comparisons))
    ap = sum(np.sum((scores == t) & (labels == 1))/labels.sum()*labels[scores >= t].mean()
             for t in sorted(set(scores), reverse=True))
    assert np.isclose(result['pr_auc'], ap)
    assert histogram_auc(p*0, n)['roc_auc'] is None


def test_diagnostic_checkpoint_idempotent(tmp_path):
    x = torch.rand(248, 64*64, generator=torch.Generator().manual_seed(5))
    seed = torch.linspace(0, 1, 64*64)
    _, diagnostics = reliability_maps(x, seed, SPEC)
    target = np.indices((32, 32))[0] >= 16
    hist, totals = change_histograms(seed, diagnostics, target, SPEC['diagnostic'])
    assert np.all(hist.sum(-1) == hist.sum(-1)[..., :1])
    assert np.array_equal(hist[:, :, 0].sum((1, 2)), totals[:, 2])
    path = tmp_path/'diag.npz'
    writer = ObservationDiagnosticWriter(path, SPEC)
    writer.add('id', hist, totals, diagnostics)
    resumed = ObservationDiagnosticWriter(path, SPEC)
    resumed.add('id', hist, totals, diagnostics)
    assert resumed.ids == ['id']
    np.testing.assert_array_equal(resumed.hist, hist)
    resumed.add('id2', hist, totals, diagnostics)
    np.testing.assert_array_equal(resumed.hist, 2*hist)


if __name__ == '__main__':
    import tempfile
    for name, check in sorted(list(globals().items())):
        if name.startswith('test_'):
            print(name, flush=True)
            if check.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as directory:
                    check(Path(directory))
            else:
                check()
    print('All independent numerical and checkpoint checks passed', flush=True)
