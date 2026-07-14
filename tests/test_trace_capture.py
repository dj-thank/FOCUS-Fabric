from __future__ import annotations

import torch
import torch.nn.functional as F

from focus_fabric.trace_capture import capture_sdpa_traces


def test_sdpa_capture_records_actual_boundary_and_restores_function() -> None:
    torch.manual_seed(4)
    original = F.scaled_dot_product_attention
    query = torch.randn(1, 4, 6, 8)
    key = torch.randn(1, 2, 6, 8)
    value = torch.randn(1, 2, 6, 8)
    with capture_sdpa_traces() as traces:
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)
    assert F.scaled_dot_product_attention is original
    assert output.shape == query.shape
    assert len(traces) == 1
    trace = traces[0]
    assert torch.equal(trace.query, query)
    expanded_key, expanded_value = trace.expanded_kv()
    assert expanded_key.shape == query.shape
    assert expanded_value.shape == query.shape
    assert trace.is_causal
    assert trace.enable_gqa
