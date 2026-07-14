"""Sequential decoding and generation utilities for FOCUS-Native."""
from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Iterable

import torch
from torch import Tensor

from .cache import ModelCache
from .config import CacheConfig
from .model import FocusTransformer


@dataclass
class GenerationResult:
    token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    prefill_seconds: float
    decode_seconds: float
    cache_report: dict[str, object]
    step_logits: Tensor | None = None

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.decode_seconds if self.decode_seconds else 0.0


@torch.no_grad()
def create_cache(
    model: FocusTransformer,
    config: CacheConfig,
    *,
    archive_device: str | torch.device = "cpu",
) -> ModelCache:
    parameter = next(model.parameters())
    # Cache tensors use the model parameter dtype in this runtime.  Derive the
    # accounting width from that dtype so fp16/bf16 deployments do not inherit
    # the fp32 default merely because the caller omitted ``dtype_bytes``.
    runtime_config = replace(config, dtype_bytes=parameter.element_size())
    return ModelCache.create(
        n_layers=model.config.n_layers,
        n_heads=model.config.n_heads,
        head_dim=model.config.head_dim,
        config=runtime_config,
        device=parameter.device,
        dtype=parameter.dtype,
        archive_device=archive_device,
    )


@torch.no_grad()
def decode_step(
    model: FocusTransformer,
    token_id: int | Tensor,
    position: int,
    cache: ModelCache,
) -> Tensor:
    """Process one token and return next-token logits [vocab]."""
    if position >= model.config.max_seq_len:
        raise ValueError(f"position {position} exceeds max_seq_len={model.config.max_seq_len}")
    device = model.token_embedding.weight.device
    token = torch.as_tensor(token_id, device=device, dtype=torch.long).reshape(1)
    x = model.token_embedding(token) + model.position_embedding[position].to(
        device=device, dtype=model.token_embedding.weight.dtype
    ).unsqueeze(0)
    copy_state: Tensor | None = None

    for layer_index, layer in enumerate(model.layers):
        normalized = layer.attn_norm(x).unsqueeze(1)  # [1,1,D]
        q, k, v = layer.attn.project_qkv(normalized)
        q_one = q[0, :, 0]
        k_one = k[0, :, 0]
        v_one = v[0, :, 0]
        if (
            model.config.memory_code_enabled
            and layer_index == model.config.n_layers - 1
        ):
            v_one = v_one.clone()
            v_one[model.config.memory_code_head] = model.memory_codebook[token[0]].to(
                device=v_one.device, dtype=v_one.dtype
            )
        attended = cache.layers[layer_index].append_and_attend(
            q_one, k_one, v_one, layer.attn
        )
        if (
            model.config.memory_code_enabled
            and layer_index == model.config.n_layers - 1
        ):
            copy_state = attended[model.config.memory_code_head]
        projection_attended = attended
        if (
            model.config.memory_code_enabled
            and layer_index == model.config.n_layers - 1
        ):
            projection_attended = attended.clone()
            projection_attended[model.config.memory_code_head] = 0
        attended_flat = projection_attended.reshape(1, -1)
        x = x + layer.attn.o_proj(attended_flat)
        x = x + layer.ffn(layer.ffn_norm(x))
    normalized_final = model.final_norm(x)
    logits = model.lm_head(normalized_final)[0]
    if model.config.memory_code_enabled:
        if copy_state is None or model.copy_gate is None:
            raise RuntimeError("memory-code state is unavailable during decode")
        copy_logits = model.config.memory_code_scale * torch.mv(
            model.memory_codebook.float(),
            torch.nn.functional.normalize(copy_state.float(), dim=-1),
        )
        gate = torch.sigmoid(model.copy_gate(normalized_final).float())[0, 0]
        logits = logits.float() + gate * copy_logits
        logits = logits.to(model.lm_head.weight.dtype)
    return logits


@torch.no_grad()
def prefill(
    model: FocusTransformer,
    token_ids: Iterable[int],
    cache: ModelCache,
    *,
    retain_logits: bool = False,
) -> tuple[Tensor, Tensor | None]:
    logits: list[Tensor] | None = [] if retain_logits else None
    last: Tensor | None = None
    for position, token_id in enumerate(token_ids):
        last = decode_step(model, int(token_id), position, cache)
        if logits is not None:
            logits.append(last.detach().cpu())
    if last is None:
        raise ValueError("prompt must contain at least one token")
    stacked = torch.stack(logits) if logits else None
    return last, stacked


@torch.no_grad()
def generate(
    model: FocusTransformer,
    prompt_ids: Iterable[int],
    cache_config: CacheConfig,
    *,
    max_new_tokens: int,
    eos_id: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int = 0,
    archive_device: str | torch.device = "cpu",
    retain_logits: bool = False,
) -> GenerationResult:
    model.eval()
    prompt = [int(token) for token in prompt_ids]
    if not prompt:
        raise ValueError("prompt_ids cannot be empty")
    if len(prompt) + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus generation exceeds model max_seq_len")
    cache = create_cache(model, cache_config, archive_device=archive_device)
    t0 = time.perf_counter()
    last_logits, prompt_logits = prefill(model, prompt, cache, retain_logits=retain_logits)
    prefill_seconds = time.perf_counter() - t0

    generator = torch.Generator(device=last_logits.device)
    generator.manual_seed(seed)
    generated: list[int] = []
    retained: list[Tensor] | None = [] if retain_logits else None
    t1 = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = last_logits
        if retained is not None:
            retained.append(logits.detach().cpu())
        if temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            scaled = logits.float() / temperature
            if top_k is not None and 0 < top_k < scaled.numel():
                threshold = torch.topk(scaled, top_k).values[-1]
                scaled = torch.where(scaled < threshold, -torch.inf, scaled)
            probabilities = torch.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        position = len(prompt) + len(generated) - 1
        last_logits = decode_step(model, next_id, position, cache)
    decode_seconds = time.perf_counter() - t1
    elapsed = time.perf_counter() - t0

    all_logits = None
    if retain_logits:
        pieces = []
        if prompt_logits is not None:
            pieces.append(prompt_logits)
        if retained:
            pieces.append(torch.stack(retained))
        all_logits = torch.cat(pieces, dim=0) if pieces else None
    return GenerationResult(
        token_ids=[*prompt, *generated],
        prompt_tokens=len(prompt),
        generated_tokens=len(generated),
        elapsed_seconds=elapsed,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        cache_report=cache.memory_report(),
        step_logits=all_logits,
    )


@torch.no_grad()
def sequential_logits(
    model: FocusTransformer,
    token_ids: Iterable[int],
    cache_config: CacheConfig,
) -> tuple[Tensor, dict[str, object]]:
    """Return one next-token logit vector per supplied token."""
    cache = create_cache(model, cache_config)
    outputs = [
        decode_step(model, token_id, position, cache)
        for position, token_id in enumerate(token_ids)
    ]
    return torch.stack(outputs), cache.memory_report()


@torch.no_grad()
def decode_step_with_trace(
    model: FocusTransformer,
    token_id: int | Tensor,
    position: int,
    cache: ModelCache,
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Decode one token and expose per-layer Q/K/V and attention output."""
    if position >= model.config.max_seq_len:
        raise ValueError(f"position {position} exceeds max_seq_len={model.config.max_seq_len}")
    device = model.token_embedding.weight.device
    token = torch.as_tensor(token_id, device=device, dtype=torch.long).reshape(1)
    x = model.token_embedding(token) + model.position_embedding[position].to(
        device=device, dtype=model.token_embedding.weight.dtype
    ).unsqueeze(0)
    traces: list[dict[str, Tensor]] = []
    copy_state: Tensor | None = None
    for layer_index, layer in enumerate(model.layers):
        normalized = layer.attn_norm(x).unsqueeze(1)
        q, k, v = layer.attn.project_qkv(normalized)
        q_one, k_one, v_one = q[0, :, 0], k[0, :, 0], v[0, :, 0]
        if (
            model.config.memory_code_enabled
            and layer_index == model.config.n_layers - 1
        ):
            v_one = v_one.clone()
            v_one[model.config.memory_code_head] = model.memory_codebook[token[0]].to(
                device=v_one.device, dtype=v_one.dtype
            )
        attended = cache.layers[layer_index].append_and_attend(q_one, k_one, v_one, layer.attn)
        if (
            model.config.memory_code_enabled
            and layer_index == model.config.n_layers - 1
        ):
            copy_state = attended[model.config.memory_code_head]
        traces.append({
            "q": q_one.detach().clone(),
            "k": k_one.detach().clone(),
            "v": v_one.detach().clone(),
            "attention": attended.detach().clone(),
        })
        projection_attended = attended
        if (
            model.config.memory_code_enabled
            and layer_index == model.config.n_layers - 1
        ):
            projection_attended = attended.clone()
            projection_attended[model.config.memory_code_head] = 0
        x = x + layer.attn.o_proj(projection_attended.reshape(1, -1))
        x = x + layer.ffn(layer.ffn_norm(x))
    normalized_final = model.final_norm(x)
    logits = model.lm_head(normalized_final)[0]
    if model.config.memory_code_enabled:
        if copy_state is None or model.copy_gate is None:
            raise RuntimeError("memory-code state is unavailable during traced decode")
        copy_logits = model.config.memory_code_scale * torch.mv(
            model.memory_codebook.float(),
            torch.nn.functional.normalize(copy_state.float(), dim=-1),
        )
        gate = torch.sigmoid(model.copy_gate(normalized_final).float())[0, 0]
        logits = (logits.float() + gate * copy_logits).to(model.lm_head.weight.dtype)
    return logits, traces
