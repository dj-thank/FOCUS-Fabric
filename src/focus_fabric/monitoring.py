"""Online sentinels for conformal exchangeability and routing drift.

Split-conformal calibration is marginal under exchangeability.  It is not a
universal per-query proof.  A deployment therefore needs sparse exact audits.
This module tracks miscoverage with a finite-sample Hoeffding confidence bound
and raises a recompile/strict-fallback signal when the lower confidence bound
exceeds the calibrated target.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import random
from typing import Deque


@dataclass
class DriftSentinel:
    target_miscoverage: float = 0.05
    confidence: float = 0.99
    window: int = 256
    minimum_audits: int = 32
    base_audit_probability: float = 0.01
    near_boundary_probability: float = 0.25
    seed: int = 0
    _miscovered: Deque[bool] = field(default_factory=deque, init=False, repr=False)
    _relative_error: Deque[float] = field(default_factory=deque, init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.target_miscoverage < 1:
            raise ValueError("target_miscoverage must lie in (0,1)")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie in (0,1)")
        if self.window < self.minimum_audits or self.minimum_audits < 1:
            raise ValueError("window must be >= minimum_audits >= 1")
        self._miscovered = deque(maxlen=self.window)
        self._relative_error = deque(maxlen=self.window)
        self._rng = random.Random(self.seed)

    def audit_probability(self, certificate_upper: float, tolerance: float) -> float:
        if tolerance <= 0:
            return 1.0
        proximity = min(max(certificate_upper / tolerance, 0.0), 1.0)
        return min(
            1.0,
            self.base_audit_probability
            + self.near_boundary_probability * proximity * proximity,
        )

    def should_audit(self, certificate_upper: float, tolerance: float) -> bool:
        return self._rng.random() < self.audit_probability(certificate_upper, tolerance)

    def observe(self, *, true_error: float, certificate_upper: float) -> None:
        if true_error < 0 or certificate_upper < 0:
            raise ValueError("errors and upper bounds must be non-negative")
        self._miscovered.append(true_error > certificate_upper)
        self._relative_error.append(true_error)

    @property
    def audits(self) -> int:
        return len(self._miscovered)

    @property
    def empirical_miscoverage(self) -> float:
        return (
            sum(self._miscovered) / len(self._miscovered)
            if self._miscovered
            else 0.0
        )

    def confidence_radius(self) -> float:
        if not self._miscovered:
            return 1.0
        delta = 1.0 - self.confidence
        return math.sqrt(math.log(1.0 / delta) / (2.0 * len(self._miscovered)))

    @property
    def lower_miscoverage_bound(self) -> float:
        return max(0.0, self.empirical_miscoverage - self.confidence_radius())

    @property
    def upper_miscoverage_bound(self) -> float:
        return min(1.0, self.empirical_miscoverage + self.confidence_radius())

    @property
    def triggered(self) -> bool:
        return (
            self.audits >= self.minimum_audits
            and self.lower_miscoverage_bound > self.target_miscoverage
        )

    def action(self) -> str:
        if self.triggered:
            return "strict_fallback_and_recompile"
        if self.audits < self.minimum_audits:
            return "collect_audits"
        return "continue"

    def report(self) -> dict[str, float | int | bool | str]:
        return {
            "audits": self.audits,
            "empirical_miscoverage": self.empirical_miscoverage,
            "lower_miscoverage_bound": self.lower_miscoverage_bound,
            "upper_miscoverage_bound": self.upper_miscoverage_bound,
            "target_miscoverage": self.target_miscoverage,
            "triggered": self.triggered,
            "action": self.action(),
            "mean_true_error": (
                sum(self._relative_error) / len(self._relative_error)
                if self._relative_error
                else 0.0
            ),
        }
