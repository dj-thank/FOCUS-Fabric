#!/usr/bin/env python3
"""Collect post-positional Q/K/V traces from a Hugging Face causal LM.

The script forces the Transformers SDPA path and intercepts the tensors passed
to PyTorch's scaled-dot-product-attention function.  This captures the actual
attention boundary rather than raw q_proj/k_proj outputs.  Architectures that
bypass PyTorch SDPA fail explicitly instead of silently emitting invalid traces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from focus_fabric.trace_capture import capture_sdpa_traces


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output .safetensors path")
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--include-token-ids", action="store_true")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("install the optional dependency set: pip install -e '.[hf]'") from error

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but CUDA is unavailable")
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)
    requested_dtype = dtype_from_name(args.dtype)
    if args.device == "cpu" and requested_dtype == torch.float16:
        raise SystemExit("float16 CPU execution is not supported by this collector; use float32/bfloat16")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=requested_dtype,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    with torch.no_grad(), capture_sdpa_traces() as traces:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if not traces:
        raise SystemExit(
            "no PyTorch SDPA calls were captured; this architecture/backend bypasses the supported boundary"
        )

    tensors: dict[str, torch.Tensor] = {}
    trace_metadata: list[dict[str, Any]] = []
    for index, trace in enumerate(traces):
        prefix = f"attention.{index:04d}"
        tensors[f"{prefix}.query"] = trace.query.contiguous()
        tensors[f"{prefix}.key_native"] = trace.key.contiguous()
        tensors[f"{prefix}.value_native"] = trace.value.contiguous()
        expanded_key, expanded_value = trace.expanded_kv()
        tensors[f"{prefix}.key_expanded"] = expanded_key.contiguous()
        tensors[f"{prefix}.value_expanded"] = expanded_value.contiguous()
        trace_metadata.append(
            {
                "index": index,
                "query_shape": list(trace.query.shape),
                "key_shape": list(trace.key.shape),
                "value_shape": list(trace.value.shape),
                "query_heads": trace.query_heads,
                "key_value_heads": trace.key_value_heads,
                "is_causal": trace.is_causal,
                "enable_gqa": trace.enable_gqa,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    metadata_path = args.metadata_output or args.output.with_suffix(".json")
    token_bytes = input_ids.detach().cpu().numpy().tobytes()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "model": args.model,
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "device": args.device,
        "dtype": args.dtype,
        "prompt_sha256": sha256_bytes(text.encode("utf-8")),
        "input_ids_sha256": sha256_bytes(token_bytes),
        "input_tokens": int(input_ids.numel()),
        "traces": trace_metadata,
        "capture_boundary": "torch.nn.functional.scaled_dot_product_attention",
        "scope_note": "Q/K/V are the tensors presented to PyTorch SDPA; unsupported custom attention backends fail closed.",
    }
    if args.include_token_ids:
        metadata["input_ids"] = input_ids.detach().cpu().tolist()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"tensors": str(args.output), "metadata": str(metadata_path), "calls": len(traces)}, indent=2))


if __name__ == "__main__":
    main()
