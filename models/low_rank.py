"""Low-rank linear layer for parameter-efficient pretraining."""

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

    self.reset_parameters()

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
    if self.bias is not None:
      fan_in = self.in_features
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      nn.init.uniform_(self.bias, -bound, bound)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x @ B^T  → (..., rank), then  @ A^T → (..., out_features)
    return F.linear(F.linear(x, self.B), self.A, self.bias)

  def extra_repr(self) -> str:
    return (
        f'in_features={self.in_features}, '
        f'out_features={self.out_features}, '
        f'rank={self.rank}, '
        f'rank_percentage={self.rank_percentage}, '
        f'bias={self.bias is not None}'
    )
