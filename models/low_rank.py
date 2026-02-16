"""Low-rank linear layer for parameter-efficient pretraining.

Supports both static low-rank and timestep-dependent (dynamic) low-rank
modes.  When timestep-dependent mode is active, the number of active rank
slices is determined per-sample by a logistic schedule over the noise
level sigma.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankLinear(nn.Module):
  """A linear layer factorised into two low-rank matrices.

  Replaces a dense ``nn.Linear(in_features, out_features)`` with
  ``y = x @ B^T @ A^T + bias`` where ``A`` has shape
  ``(out_features, rank)`` and ``B`` has shape ``(rank, in_features)``.

  The rank is determined so that the total parameter count of ``A``
  and ``B`` equals a target percentage of the original dense layer's
  parameter count (excluding bias):

      rank = floor(pct * in_features * out_features
                   / (in_features + out_features))

  Timestep-dependent mode
  -----------------------
  When ``_sigma`` is set (injected externally), the layer computes
  per-sample active ranks via a logistic increasing schedule and masks
  the intermediate rank dimension accordingly.

  Args:
    in_features: Size of each input sample.
    out_features: Size of each output sample.
    rank_percentage: Fraction (0, 1] of the original dense layer's
        parameter count to retain.
    bias: If ``True``, adds a learnable bias of shape
        ``(out_features,)``.
  """

  def __init__(
      self,
      in_features: int,
      out_features: int,
      rank_percentage: float = 1.0,
      bias: bool = False,
  ):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.rank_percentage = rank_percentage

    # Compute rank from the target parameter-count percentage.
    dense_params = in_features * out_features
    rank = int(
        math.floor(
            rank_percentage * dense_params
            / (in_features + out_features)
        )
    )
    rank = max(1, min(rank, min(in_features, out_features)))
    self.rank = rank

    # Factorised weight: W ≈ A @ B  (out×r) @ (r×in)
    self.A = nn.Parameter(torch.empty(out_features, rank))
    self.B = nn.Parameter(torch.empty(rank, in_features))

    if bias:
      self.bias = nn.Parameter(torch.zeros(out_features))
    else:
      self.register_parameter('bias', None)

    # ----- timestep-dependent rank gating (set externally) -----
    self._sigma: torch.Tensor | None = None
    self._sigma_max: float | None = None
    self._r_min_ratio: float = 0.4
    self._logistic_k: float = 8.0
    self._logistic_m: float = 0.6

    self.reset_parameters()

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
    if self.bias is not None:
      fan_in = self.in_features
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      nn.init.uniform_(self.bias, -bound, bound)

  # ------------------------------------------------------------------
  # Logistic increasing schedule:  rank increases as sigma decreases
  # ------------------------------------------------------------------
  def _active_ranks(
      self,
      sigma: torch.Tensor,
      sigma_max: float,
  ) -> torch.Tensor:
    """Map per-sample sigma → active rank r(sigma).

    Uses a *logistic increasing* schedule: low sigma (clean data) →
    high rank, high sigma (noisy data) → low rank.

    Args:
      sigma:     [B] float tensor of noise levels.
      sigma_max: scalar, maximum sigma for normalisation.

    Returns:
      [B] long tensor of per-sample active ranks.
    """
    r = self.rank
    r_min = max(1, int(round(self._r_min_ratio * r)))
    k = self._logistic_k
    m = self._logistic_m

    sigma = sigma.to(self.A.device)

    # frac=0 → sigma=sigma_max (noisy), frac=1 → sigma=0 (clean)
    frac = (1.0 - (sigma / (sigma_max + 1e-8))).clamp(0, 1)

    # Normalised sigmoid so that f(0)=0, f(1)=1
    raw_s = torch.sigmoid(torch.tensor(k, device=sigma.device, dtype=sigma.dtype) * (frac - m))
    s_min = torch.sigmoid(torch.tensor(k * (0.0 - m), device=sigma.device, dtype=sigma.dtype))
    s_max = torch.sigmoid(torch.tensor(k * (1.0 - m), device=sigma.device, dtype=sigma.dtype))
    s = (raw_s - s_min) / (s_max - s_min + 1e-8)

    r_t = (r_min + (r - r_min) * s).floor().clamp(min=r_min, max=r)
    return r_t.to(torch.long)

  # ------------------------------------------------------------------
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # ---- static path (no timestep conditioning) --------------------
    if self._sigma is None:
      # x @ B^T  → (..., rank), then  @ A^T → (..., out_features)
      return F.linear(F.linear(x, self.B), self.A, self.bias)

    # ---- timestep-dependent path -----------------------------------
    sigma = self._sigma          # [B]
    sigma_max = self._sigma_max

    B_x = x.shape[0]
    B_t = sigma.shape[0]

    # Expand sigma if batch dims differ (e.g. seq-level flattening)
    if B_x == B_t:
      sigma_expanded = sigma
    elif B_x % B_t == 0:
      repeat = B_x // B_t
      sigma_expanded = sigma.unsqueeze(1).expand(-1, repeat).contiguous().view(-1)
    else:
      repeat = math.ceil(B_x / B_t)
      sigma_expanded = sigma.unsqueeze(1).expand(-1, repeat).contiguous().view(-1)[:B_x]

    r_t = self._active_ranks(sigma_expanded, sigma_max)  # [B_x]
    r = self.rank

    # Fast path: if the whole batch shares the same active rank, slice
    if bool(torch.all(r_t == r_t[0])):
      r_active = int(r_t[0].item())
      Bx = F.linear(x, self.B[:r_active, :])            # (..., r_active)
      return F.linear(Bx, self.A[:, :r_active], self.bias)

    # General path: mixed ranks in batch → mask activations
    Bx = F.linear(x, self.B)                              # (..., r)
    idx = torch.arange(r, device=x.device)
    mask = (idx.unsqueeze(0) < r_t.unsqueeze(1))           # [B_x, r]

    # Handle 3-D tensors (batch, seq, rank)
    if Bx.dim() == 3:
      mask = mask.unsqueeze(1)                             # [B_x, 1, r]

    Bx = Bx * mask
    return F.linear(Bx, self.A, self.bias)

  def extra_repr(self) -> str:
    return (
        f'in_features={self.in_features}, '
        f'out_features={self.out_features}, '
        f'rank={self.rank}, '
        f'rank_percentage={self.rank_percentage}, '
        f'bias={self.bias is not None}'
    )
