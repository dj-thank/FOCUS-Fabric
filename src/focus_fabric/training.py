"""FOCUS-native and fabric-aware optimization objectives."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from focus_native.model import FocusTransformer


@dataclass(frozen=True)
class NativeLossWeights:
    language_model: float = 1.0
    focus_language_model: float = 1.0
    logit_distillation: float = 0.25
    attention_output: float = 0.50
    attention_mass: float = 0.25
    jacobian_tail: float = 0.05
    route_distance: float = 0.01

    def validate(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("native loss weights must be non-negative")


@dataclass
class NativeLossResult:
    loss: Tensor
    terms: dict[str, Tensor]

    def detached(self) -> dict[str, float]:
        return {
            name: float(value.detach().float().item())
            for name, value in self.terms.items()
        }


def focus_native_loss(
    model: FocusTransformer,
    input_ids: Tensor,
    labels: Tensor,
    *,
    cold_len: int,
    weights: NativeLossWeights = NativeLossWeights(),
    loss_mask: Tensor | None = None,
) -> NativeLossResult:
    """Train exact teacher and differentiable FOCUS-prefix student jointly."""

    weights.validate()
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("input_ids and labels must have matching shape [B,T]")
    if not 0 < cold_len < input_ids.shape[1]:
        raise ValueError("cold_len must leave both cold and hot tokens")
    teacher = model(input_ids, labels=labels, loss_mask=loss_mask, focus_mode=False)
    student = model(
        input_ids,
        labels=labels,
        loss_mask=loss_mask,
        focus_mode=True,
        focus_cold_len=cold_len,
        focus_alpha=1.0,
    )
    if teacher.loss is None or student.loss is None:
        raise RuntimeError("model failed to return language-model losses")
    suffix = slice(cold_len, None)
    distillation = F.mse_loss(
        student.logits[:, suffix].float(),
        teacher.logits[:, suffix].detach().float(),
    )
    terms = {
        "exact_lm": teacher.loss,
        "focus_lm": student.loss,
        "logit_distillation": distillation,
        "attention_output_nmse": student.aux["output_nmse"],
        "attention_logmass_mse": student.aux["logmass_mse"],
        "jacobian_tail_ratio": student.aux["tail_ratio"],
        "route_distance": student.aux["route_distance"],
    }
    total = (
        weights.language_model * terms["exact_lm"]
        + weights.focus_language_model * terms["focus_lm"]
        + weights.logit_distillation * terms["logit_distillation"]
        + weights.attention_output * terms["attention_output_nmse"]
        + weights.attention_mass * terms["attention_logmass_mse"]
        + weights.jacobian_tail * terms["jacobian_tail_ratio"]
        + weights.route_distance * terms["route_distance"]
    )
    terms["total"] = total
    return NativeLossResult(total, terms)
