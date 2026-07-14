"""Checkpoint I/O for the repaired FOCUS-Native demonstrator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .config import ModelConfig
from .model import FocusTransformer
from .tokenizer import HybridTokenizer


def _find_weights(path: Path) -> tuple[Path, Path]:
    if path.suffix == ".safetensors":
        if not path.exists():
            raise FileNotFoundError(path)
        return path, path.parent
    direct = path / "model.safetensors"
    nested = path / "final" / "model.safetensors"
    if direct.exists():
        return direct, path
    if nested.exists():
        return nested, nested.parent
    raise FileNotFoundError(f"no model.safetensors found under {path}")


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[FocusTransformer, HybridTokenizer, dict[str, Any]]:
    weights, directory = _find_weights(Path(path))
    raw_state = load_file(str(weights), device="cpu")
    config_path = directory / "config.json"
    if config_path.exists():
        config = ModelConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    else:
        config = ModelConfig(memory_code_enabled="copy_gate.weight" in raw_state)
    config.validate()
    model = FocusTransformer(config)
    state = dict(raw_state)
    # Tied checkpoints serialize one alias.  Restore the missing alias before
    # strict loading so corruption is not hidden behind strict=False.
    if config.tie_embeddings and "token_embedding.weight" not in state:
        if "lm_head.weight" not in state:
            raise RuntimeError("tied checkpoint contains neither embedding alias")
        state["token_embedding.weight"] = state["lm_head.weight"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint architecture mismatch; missing={missing}, unexpected={unexpected}"
        )
    model.to(device=device)
    if dtype is not None:
        model.to(dtype=dtype)
    tokenizer = HybridTokenizer(config.vocab_size)
    metadata_path = directory / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    metadata.setdefault("checkpoint", str(weights))
    metadata.setdefault(
        "tokenizer_status",
        "compatibility byte tokenizer; original legacy symbolic mapping unavailable",
    )
    return model, tokenizer, metadata


def save_checkpoint(
    directory: str | Path,
    model: FocusTransformer,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(target / "model.safetensors"))
    (target / "config.json").write_text(
        json.dumps(model.config.as_dict(), indent=2), encoding="utf-8"
    )
    payload = dict(metadata or {})
    payload.setdefault("format", "focus-native-repaired-v1")
    (target / "metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return target
