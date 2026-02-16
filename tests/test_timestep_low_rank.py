"""Smoke test for timestep-dependent low-rank DiT.

Tests that:
1. Static low-rank forward pass works (backward compat)
2. Timestep-dependent low-rank forward pass runs
3. Different sigma values produce different outputs (rank gating active)
4. Output shapes are correct
5. LowRankLinear._active_ranks schedule behaves correctly
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf
from models.low_rank import LowRankLinear


def test_active_ranks_schedule():
    """Verify logistic increasing: rank increases as sigma decreases."""
    print("=== Test: _active_ranks schedule ===")
    layer = LowRankLinear(64, 64, rank_percentage=0.5)
    layer._logistic_k = 8.0
    layer._logistic_m = 0.6
    layer._r_min_ratio = 0.4

    sigma_max = 20.0
    sigmas = torch.linspace(0.0, sigma_max, 11)  # 0=clean, 20=noisy
    ranks = layer._active_ranks(sigmas, sigma_max)
    print(f"  Rank of layer: {layer.rank}")
    print(f"  Sigma values: {sigmas.tolist()}")
    print(f"  Active ranks: {ranks.tolist()}")

    # Clean (sigma=0) should have highest rank
    assert ranks[0] >= ranks[-1], (
        f"Expected rank at sigma=0 ({ranks[0]}) >= rank at sigma={sigma_max} ({ranks[-1]})")
    # Noisy (sigma=sigma_max) should have lowest rank (= r_min)
    r_min = max(1, int(round(0.4 * layer.rank)))
    assert ranks[-1] == r_min, f"Expected r_min={r_min}, got {ranks[-1]}"
    # Clean should get full rank
    assert ranks[0] == layer.rank, f"Expected full rank={layer.rank}, got {ranks[0]}"
    print("  PASSED\n")


def test_static_low_rank():
    """Static low-rank forward (no sigma) should work unchanged."""
    print("=== Test: static low-rank forward ===")
    layer = LowRankLinear(64, 32, rank_percentage=0.5)
    x = torch.randn(4, 10, 64)
    y = layer(x)
    assert y.shape == (4, 10, 32), f"Expected (4,10,32), got {y.shape}"
    print(f"  Output shape: {y.shape}  PASSED\n")


def test_timestep_dependent_forward():
    """With sigma injected, forward should produce correct shapes."""
    print("=== Test: timestep-dependent forward ===")
    layer = LowRankLinear(64, 32, rank_percentage=0.5)
    layer._sigma = torch.tensor([0.1, 5.0, 10.0, 19.0])
    layer._sigma_max = 20.0
    layer._r_min_ratio = 0.4
    layer._logistic_k = 8.0
    layer._logistic_m = 0.6

    x = torch.randn(4, 10, 64)
    y = layer(x)
    assert y.shape == (4, 10, 32), f"Expected (4,10,32), got {y.shape}"
    print(f"  Output shape: {y.shape}  PASSED\n")


def test_different_sigmas_different_outputs():
    """Different sigma values should produce different outputs."""
    print("=== Test: different sigma → different output ===")
    layer = LowRankLinear(64, 32, rank_percentage=0.5)
    layer._sigma_max = 20.0
    layer._r_min_ratio = 0.4
    layer._logistic_k = 8.0
    layer._logistic_m = 0.6

    x = torch.randn(1, 10, 64)

    # Same input, low sigma (clean)
    layer._sigma = torch.tensor([0.1])
    y_clean = layer(x).clone()

    # Same input, high sigma (noisy)
    layer._sigma = torch.tensor([19.0])
    y_noisy = layer(x).clone()

    diff = (y_clean - y_noisy).abs().sum().item()
    print(f"  Output diff between sigma=0.1 and sigma=19.0: {diff:.6f}")
    assert diff > 0, "Outputs should differ for different sigma values!"
    print("  PASSED\n")


def test_uniform_sigma_uses_slice():
    """When all sigma values are the same, the fast slice path should run."""
    print("=== Test: uniform sigma batch (slice path) ===")
    layer = LowRankLinear(64, 32, rank_percentage=0.5)
    layer._sigma = torch.tensor([5.0, 5.0, 5.0, 5.0])
    layer._sigma_max = 20.0
    layer._r_min_ratio = 0.4
    layer._logistic_k = 8.0
    layer._logistic_m = 0.6

    x = torch.randn(4, 10, 64)
    y = layer(x)
    assert y.shape == (4, 10, 32), f"Expected (4,10,32), got {y.shape}"
    print(f"  Output shape: {y.shape}  PASSED\n")


if __name__ == "__main__":
    test_active_ranks_schedule()
    test_static_low_rank()
    test_timestep_dependent_forward()
    test_different_sigmas_different_outputs()
    test_uniform_sigma_uses_slice()
    print("=" * 50)
    print("All tests passed!")
