"""
Regression + semantics test for the corrected do-mask (UNIFIED_BUILD_PLAN.md §1, §7d).

Guards the two things that must stay true forever:
  1. STRUCTURE: the perturbed gene's QUERY ROW is masked (it stops attending to others,
     self kept); its KEY COLUMN is NOT masked (others keep attending to it). The deleted
     bug was `M[:, perturbed] = -inf`, which this test forbids.
  2. SEMANTICS: with the mask, changing the perturbed gene's input changes OTHER genes'
     outputs (the intervention propagates), while changing another gene's input does NOT
     change the perturbed gene's output (it is clamped by the intervention).

Run: .venv/bin/python tests/test_do_mask.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core.models.do_attention import build_do_mask, MultiHeadDoAttention


def test_mask_structure():
    B, S = 3, 6
    p = torch.tensor([2, 0, 5])
    M = build_do_mask(S, p)
    for b in range(B):
        pi = p[b].item()
        row = M[b, pi]
        # query row: -inf everywhere except the self-diagonal
        assert torch.isinf(row[torch.arange(S) != pi]).all(), "query row must be cut"
        assert row[pi].item() == 0.0, "self-attention must be kept"
        # key column: MUST remain finite (this is the anti-regression check)
        col = M[b, :, pi]
        assert torch.isfinite(col).all(), "key COLUMN must NOT be masked (propagation!)"
    print("PASS  mask structure: query row cut, self kept, key column finite")


def test_no_column_masking_regression():
    """If someone re-adds M[:, perturbed] = -inf, some column entry becomes -inf. Forbid it."""
    M = build_do_mask(8, torch.tensor([4]))
    assert torch.isfinite(M[0, :, 4]).all(), "regression: perturbed key column was masked"
    print("PASS  anti-regression: perturbed key column is finite")


def test_intervention_propagates():
    torch.manual_seed(0)
    attn = MultiHeadDoAttention(d_model=32, n_heads=4).eval()
    B, S, d = 1, 6, 32
    p = 2
    x = torch.randn(B, S, d)
    mask = build_do_mask(S, torch.tensor([p]))

    with torch.no_grad():
        y0 = attn(x, do_mask=mask)

        # (a) perturb the intervention gene's input -> downstream (other) tokens must change
        x_pert = x.clone()
        x_pert[0, p] += 5.0
        y_pert = attn(x_pert, do_mask=mask)
        others = [i for i in range(S) if i != p]
        downstream_change = (y_pert[0, others] - y0[0, others]).abs().max().item()
        assert downstream_change > 1e-4, "intervention did NOT propagate downstream"

        # (b) perturb a DIFFERENT gene's input -> the perturbed gene's output must NOT change
        j = 4
        x_other = x.clone()
        x_other[0, j] += 5.0
        y_other = attn(x_other, do_mask=mask)
        perturbed_change = (y_other[0, p] - y0[0, p]).abs().max().item()
        assert perturbed_change < 1e-5, "perturbed gene wrongly attended to another gene"

    print(f"PASS  semantics: downstream Δ={downstream_change:.3f} (propagates), "
          f"perturbed-gene Δ={perturbed_change:.2e} (clamped)")


if __name__ == "__main__":
    test_mask_structure()
    test_no_column_masking_regression()
    test_intervention_propagates()
    print("\nDO-MASK TESTS PASSED")
