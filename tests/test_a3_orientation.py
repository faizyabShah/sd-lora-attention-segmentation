"""A3 layer-score orientation (protocol Section 107): historical vs GoCA."""
import math

import torch
import torch.nn.functional as F

from src.segmentation.sg24_ca import affinity_similarity


def goca_score(query, key, reference):
    """GoCA's resize, including the second permute(2, 3, 0, 1)."""
    ref_hw, hw = math.isqrt(reference.shape[0]), math.isqrt(query.shape[-2])
    total = 0.
    for h in range(query.shape[0]):
        m = torch.softmax(query[h] @ key[h].T / math.sqrt(query.shape[-1]), -1)
        if hw != ref_hw:
            m = F.interpolate(m.reshape(1, hw*hw, hw, hw), size=(ref_hw, ref_hw),
                              mode="bilinear", align_corners=False)
            m = m.reshape(hw, hw, ref_hw, ref_hw).permute(2, 3, 0, 1)
            m = F.interpolate(m.reshape(1, ref_hw*ref_hw, hw, hw), size=(ref_hw, ref_hw), mode="nearest")
            m = m.reshape(ref_hw, ref_hw, ref_hw, ref_hw).permute(2, 3, 0, 1).reshape(ref_hw**2, ref_hw**2)
        total += float((m * reference).sum())
    return total / query.shape[0]


def _inputs(hw, ref_hw, seed):
    torch.manual_seed(seed)
    query, key = torch.randn(3, hw*hw, 8), torch.randn(3, hw*hw, 8)
    reference = torch.softmax(torch.randn(ref_hw**2, ref_hw**2), -1)
    return query, key, reference


def test_default_score_is_unchanged_and_both_returns_it_first():
    for hw in (8, 16, 32):
        q, k, ref = _inputs(hw, 16, hw)
        old = affinity_similarity(q, k, ref)
        implemented, _ = affinity_similarity(q, k, ref, both=True)
        assert float(old) == float(implemented)


def test_goca_orientation_matches_goca_and_differs_after_resize():
    for hw in (8, 32):
        q, k, ref = _inputs(hw, 16, hw)
        implemented, goca = affinity_similarity(q, k, ref, both=True)
        assert abs(float(goca) - goca_score(q, k, ref)) < 1e-5
        # The historical score is the GoCA score of the transposed reference.
        assert abs(float(implemented) - goca_score(q, k, ref.T.contiguous())) < 1e-5
        assert abs(float(implemented) - float(goca)) > 1e-6


def test_orientations_coincide_without_resize():
    q, k, ref = _inputs(16, 16, 5)
    implemented, goca = affinity_similarity(q, k, ref, both=True)
    assert float(implemented) == float(goca)
