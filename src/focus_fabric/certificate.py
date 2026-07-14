"""Split-conformal runtime error certificates."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .utils import finite_sample_quantile


@dataclass(frozen=True)
class ConformalCertificate:
    """Marginal upper bound calibrated from error/proxy ratios.

    Under exchangeability, the finite-sample split-conformal coverage statement
    applies to the scalar nonconformity score. This is not a pointwise proof;
    high predicted error is routed to exact fallback.
    """

    qhat: float
    alpha: float
    proxy_floor: float
    calibration_count: int
    empirical_coverage: float

    @classmethod
    def fit(cls, errors: Tensor, proxies: Tensor, *, alpha: float, proxy_floor: float = 1e-4) -> "ConformalCertificate":
        if errors.shape != proxies.shape:
            raise ValueError("errors and proxies must have the same shape")
        ratios = errors.detach().float() / (proxies.detach().float().clamp_min(0) + proxy_floor)
        qhat = float(finite_sample_quantile(ratios, 1.0 - alpha).item())
        upper = qhat * (proxies.detach().float().clamp_min(0) + proxy_floor)
        coverage = float((errors.detach().float() <= upper).float().mean().item())
        return cls(qhat, alpha, proxy_floor, int(errors.numel()), coverage)

    def upper(self, proxy: Tensor | float) -> Tensor:
        tensor = torch.as_tensor(proxy, dtype=torch.float32)
        return self.qhat * (tensor.clamp_min(0) + self.proxy_floor)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "qhat": self.qhat,
            "alpha": self.alpha,
            "proxy_floor": self.proxy_floor,
            "calibration_count": self.calibration_count,
            "empirical_coverage": self.empirical_coverage,
        }
