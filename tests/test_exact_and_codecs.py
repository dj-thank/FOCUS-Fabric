from __future__ import annotations

import torch

from focus_fabric.codecs import GaussianMixtureCodec
from focus_fabric.exact import exact_head_summary
from focus_fabric.kernels.gaussian_eval import torch_gaussian_eval
from focus_fabric.types import merge_summaries


def test_exact_summary_merge_is_exact() -> None:
    torch.manual_seed(1)
    query = torch.randn(8)
    keys = torch.randn(17, 8)
    values = torch.randn(17, 8)
    merged = merge_summaries(
        [
            exact_head_summary(query, keys[:7], values[:7], 8**-0.5),
            exact_head_summary(query, keys[7:], values[7:], 8**-0.5),
        ]
    )
    full = exact_head_summary(query, keys, values, 8**-0.5)
    assert torch.allclose(merged.output, full.output, atol=2e-6, rtol=2e-6)
    assert torch.allclose(merged.log_mass, full.log_mass, atol=2e-6, rtol=2e-6)


def test_gaussian_reference_matches_codec_formula() -> None:
    torch.manual_seed(2)
    keys = torch.randn(32, 8)
    values = torch.randn(32, 8)
    query = torch.randn(8)
    codec = GaussianMixtureCodec.compile(
        keys,
        values,
        clusters=2,
        rank=3,
        scale=8**-0.5,
        seed=7,
        iterations=4,
    )
    direct = codec.evaluate(query).summary
    output, mass = torch_gaussian_eval(
        query.unsqueeze(0),
        codec.mean_keys.unsqueeze(0),
        codec.mean_values.unsqueeze(0),
        codec.log_counts.unsqueeze(0),
        codec.basis.unsqueeze(0),
        codec.variances.unsqueeze(0),
        codec.cross.unsqueeze(0),
        codec.scale,
    )
    assert torch.allclose(output[0], direct.output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(mass[0], direct.log_mass, atol=1e-5, rtol=1e-5)
