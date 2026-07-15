"""Direct integration of FOCUS-Fabric into the repaired decoder model."""
from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Iterable

import torch
from torch import Tensor
import torch.nn.functional as F

from focus_native.model import FocusTransformer

from .config import FabricConfig
from .fabric import MemoryFabricLayer


@dataclass
class FabricModelCache:
    layers: list[MemoryFabricLayer]

    @classmethod
    def create(
        cls, model: FocusTransformer, config: FabricConfig
    ) -> "FabricModelCache":
        parameter = next(model.parameters())
        compiler = replace(config.compiler, dtype_bytes=parameter.element_size())
        runtime = replace(
            config,
            n_heads=model.config.n_heads,
            head_dim=model.config.head_dim,
            scale=model.config.head_dim ** -0.5,
            compiler=compiler,
        )
        runtime.validate()
        return cls(
            [
                MemoryFabricLayer.create(
                    runtime, device=parameter.device, dtype=parameter.dtype
                )
                for _ in range(model.config.n_layers)
            ]
        )

    def report(self, *, include_pages: bool = False) -> dict[str, Any]:
        reports = [
            layer.report(include_pages=include_pages) for layer in self.layers
        ]
        additive = (
            "active_bytes",
            "archive_bytes",
            "exact_kv_bytes",
            "queries",
            "page_head_evaluations",
            "fallback_decisions",
            "fallback_tokens",
            "archive_bytes_read",
            "pages_compiled",
            "pages_merged",
            "compacted_tokens",
            "compile_seconds",
            "attention_seconds",
            "invalid_codec_outputs",
        )
        aggregate: dict[str, Any] = {
            key: sum(float(report[key]) for report in reports) for key in additive
        }
        aggregate["active_compression"] = (
            aggregate["exact_kv_bytes"] / aggregate["active_bytes"]
            if aggregate["active_bytes"]
            else 1.0
        )
        aggregate["fallback_rate"] = (
            aggregate["fallback_decisions"] / aggregate["page_head_evaluations"]
            if aggregate["page_head_evaluations"]
            else 0.0
        )
        representations: dict[str, int] = {}
        for report in reports:
            for name, count in report["representations"].items():
                representations[name] = representations.get(name, 0) + int(count)
        aggregate["representations"] = representations
        aggregate["layers"] = reports
        return aggregate


@dataclass
class FabricGenerationResult:
    token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    elapsed_seconds: float
    cache_report: dict[str, Any]
    step_logits: Tensor | None = None

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.decode_seconds if self.decode_seconds else 0.0


@torch.no_grad()
def decode_step_fabric(
    model: FocusTransformer,
    token_id: int | Tensor,
    position: int,
    cache: FabricModelCache,
) -> Tensor:
    if position >= model.config.max_seq_len:
        raise ValueError(f"position {position} exceeds model maximum")
    device = model.token_embedding.weight.device
    token = torch.as_tensor(token_id, device=device, dtype=torch.long).reshape(1)
    x = model.token_embedding(token) + model.position_embedding[position].to(
        device=device, dtype=model.token_embedding.weight.dtype
    ).unsqueeze(0)
    copy_state: Tensor | None = None
    for layer_index, layer in enumerate(model.layers):
        normalized = layer.attn_norm(x).unsqueeze(1)
        q, k, v = layer.attn.project_qkv(normalized)
        q_one, k_one, v_one = q[0, :, 0], k[0, :, 0], v[0, :, 0]
        if model.config.memory_code_enabled and layer_index == model.config.n_layers - 1:
            v_one = v_one.clone()
            v_one[model.config.memory_code_head] = model.memory_codebook[token[0]].to(
                device=v_one.device, dtype=v_one.dtype
            )
        attended = cache.layers[layer_index].append_and_attend(q_one, k_one, v_one)
        if model.config.memory_code_enabled and layer_index == model.config.n_layers - 1:
            copy_state = attended[model.config.memory_code_head]
            projected_heads = attended.clone()
            projected_heads[model.config.memory_code_head] = 0
        else:
            projected_heads = attended
        x = x + layer.attn.o_proj(projected_heads.reshape(1, -1))
        x = x + layer.ffn(layer.ffn_norm(x))
    normalized_final = model.final_norm(x)
    logits = model.lm_head(normalized_final)[0]
    if model.config.memory_code_enabled:
        if copy_state is None or model.copy_gate is None:
            raise RuntimeError("memory-code state unavailable")
        copy_logits = model.config.memory_code_scale * torch.mv(
            model.memory_codebook.float(), F.normalize(copy_state.float(), dim=-1)
        )
        gate = torch.sigmoid(model.copy_gate(normalized_final).float())[0, 0]
        logits = (logits.float() + gate * copy_logits).to(model.lm_head.weight.dtype)
    return logits


@torch.no_grad()
def sequential_logits_fabric(
    model: FocusTransformer,
    token_ids: Iterable[int],
    config: FabricConfig,
) -> tuple[Tensor, dict[str, Any]]:
    cache = FabricModelCache.create(model, config)
    outputs = [
        decode_step_fabric(model, int(token), position, cache)
        for position, token in enumerate(token_ids)
    ]
    if not outputs:
        raise ValueError("token_ids cannot be empty")
    return torch.stack(outputs), cache.report(include_pages=False)


@torch.no_grad()
def generate_fabric(
    model: FocusTransformer,
    prompt_ids: Iterable[int],
    config: FabricConfig,
    *,
    max_new_tokens: int,
    eos_id: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int = 0,
    retain_logits: bool = False,
) -> FabricGenerationResult:
    prompt = [int(token) for token in prompt_ids]
    if not prompt:
        raise ValueError("prompt cannot be empty")
    if len(prompt) + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus generation exceeds max_seq_len")
    model.eval()
    cache = FabricModelCache.create(model, config)
    retained: list[Tensor] | None = [] if retain_logits else None
    started = time.perf_counter()
    last_logits: Tensor | None = None
    for position, token in enumerate(prompt):
        last_logits = decode_step_fabric(model, token, position, cache)
        if retained is not None:
            retained.append(last_logits.detach().cpu())
    assert last_logits is not None
    prefill_seconds = time.perf_counter() - started

    generator = torch.Generator(device=last_logits.device)
    generator.manual_seed(seed)
    generated: list[int] = []
    decode_started = time.perf_counter()
    for _ in range(max_new_tokens):
        if temperature <= 0:
            next_id = int(torch.argmax(last_logits).item())
        else:
            scaled = last_logits.float() / temperature
            if top_k is not None and 0 < top_k < scaled.numel():
                threshold = torch.topk(scaled, top_k).values[-1]
                scaled = torch.where(scaled < threshold, -torch.inf, scaled)
            next_id = int(
                torch.multinomial(
                    torch.softmax(scaled, dim=-1), 1, generator=generator
                ).item()
            )
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        position = len(prompt) + len(generated) - 1
        last_logits = decode_step_fabric(model, next_id, position, cache)
        if retained is not None:
            retained.append(last_logits.detach().cpu())
    decode_seconds = time.perf_counter() - decode_started
    return FabricGenerationResult(
        token_ids=[*prompt, *generated],
        prompt_tokens=len(prompt),
        generated_tokens=len(generated),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        elapsed_seconds=time.perf_counter() - started,
        cache_report=cache.report(include_pages=False),
        step_logits=torch.stack(retained) if retained else None,
    )
