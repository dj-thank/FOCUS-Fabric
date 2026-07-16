"""FOCUS-Native decoder-only Transformer.

The model can train with exact causal attention or with an old-prefix page
replaced by an analytically compiled, query-conditioned low-rank operator.
This makes cache compression part of the optimization objective rather than a
post-hoc approximation only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig


@dataclass
class AttentionAux:
    output_nmse: Tensor
    logmass_mse: Tensor
    tail_ratio: Tensor
    route_distance: Tensor

    @classmethod
    def zeros(cls, reference: Tensor) -> "AttentionAux":
        zero = reference.new_zeros(())
        return cls(zero, zero, zero, zero)


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None
    aux: dict[str, Tensor]
    traces: list[dict[str, Tensor]] | None = None
    copy_logits: Tensor | None = None
    copy_gate: Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


def sinusoidal_positions(max_len: int, dim: int) -> Tensor:
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    exponent = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    frequencies = torch.exp(-math.log(10000.0) * exponent)
    table = torch.zeros(max_len, dim, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * frequencies)
    table[:, 1::2] = torch.cos(position * frequencies)
    return table


def deterministic_memory_codes(vocab_size: int, dim: int, seed: int) -> Tensor:
    """Create reproducible, approximately orthogonal token identity codes."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    codes = torch.empty(vocab_size, dim, dtype=torch.float32)
    codes.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
    return F.normalize(codes, dim=-1)


class FocusSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, *, memory_router: bool = False) -> None:
        super().__init__()
        self.config = config
        self.memory_router_enabled = bool(memory_router)
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        if self.memory_router_enabled:
            if not config.memory_code_enabled:
                raise ValueError("memory router requires a memory-code head")
            self.memory_q_proj = nn.Linear(config.d_model, self.head_dim, bias=False)
            self.memory_k_proj = nn.Linear(config.d_model, self.head_dim, bias=False)
            self.memory_router_log_temperature = nn.Parameter(
                torch.tensor(math.log(config.memory_router_temperature), dtype=torch.float32)
            )
        else:
            self.memory_q_proj = None
            self.memory_k_proj = None
            self.register_parameter("memory_router_log_temperature", None)

        anchors = torch.randn(config.n_heads, config.focus_patches, self.head_dim)
        anchors = F.normalize(anchors, dim=-1)
        self.focus_anchors = nn.Parameter(anchors)
        basis = torch.randn(
            config.n_heads,
            config.focus_patches,
            self.head_dim,
            config.focus_rank,
        )
        self.focus_basis_raw = nn.Parameter(basis / math.sqrt(self.head_dim))
        self.register_buffer(
            "focus_radius",
            torch.full((config.n_heads, config.focus_patches), float("inf")),
        )

    def basis(self) -> Tensor:
        # QR makes the runtime/compiler basis orthonormal and keeps the native
        # rank constraint explicit. Compute in fp32 for stability.
        q, _ = torch.linalg.qr(self.focus_basis_raw.float(), mode="reduced")
        return q.to(self.focus_basis_raw.dtype)

    def memory_temperature(self) -> Tensor:
        if not self.memory_router_enabled or self.memory_router_log_temperature is None:
            return self.focus_anchors.new_tensor(0.0)
        minimum = self.config.memory_router_min_temperature
        maximum = self.config.memory_router_max_temperature
        return self.memory_router_log_temperature.exp().clamp(minimum, maximum)

    def normalize_memory_qk(self, raw_q: Tensor, raw_k: Tensor) -> tuple[Tensor, Tensor]:
        """Map raw router projections to Q/K whose scaled dot is T*cosine."""
        if not self.memory_router_enabled:
            raise RuntimeError("memory router is not enabled for this attention layer")
        temperature = self.memory_temperature().to(device=raw_q.device, dtype=torch.float32)
        # Attention later multiplies q·k by self.scale.  Multiplying each unit
        # vector by sqrt(T/self.scale) therefore yields logits T*cos(theta).
        factor = torch.sqrt(temperature / self.scale)
        q = F.normalize(raw_q.float(), dim=-1) * factor
        k = F.normalize(raw_k.float(), dim=-1) * factor
        return q.to(raw_q.dtype), k.to(raw_k.dtype)

    def project_memory_qk(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.memory_q_proj is None or self.memory_k_proj is None:
            raise RuntimeError("memory router is not enabled for this attention layer")
        return self.normalize_memory_qk(self.memory_q_proj(x), self.memory_k_proj(x))

    def project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        if self.memory_router_enabled:
            memory_q, memory_k = self.project_memory_qk(x)
            head = self.config.memory_code_head
            q = q.clone()
            k = k.clone()
            q[:, head] = memory_q
            k[:, head] = memory_k
        return q, k, v

    def _exact_causal(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        length = q.shape[-2]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal = torch.ones(length, length, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        logmass = torch.logsumexp(scores.float(), dim=-1).to(scores.dtype)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        return torch.matmul(weights, v), logmass

    def compile_page(self, keys: Tensor, values: Tensor) -> dict[str, Tensor]:
        """Differentiably compile [B,H,N,D] K/V into native operators."""
        anchors = self.focus_anchors
        basis = self.basis()
        scores = torch.einsum("hmd,bhnd->bhmn", anchors, keys) * self.scale
        probabilities = torch.softmax(scores.float(), dim=-1).to(keys.dtype)
        output0 = torch.einsum("bhmn,bhnd->bhmd", probabilities, values)
        logmass0 = torch.logsumexp(scores.float(), dim=-1).to(keys.dtype)
        mean_key = torch.einsum("bhmn,bhnd->bhmd", probabilities, keys)
        moment = torch.einsum(
            "bhmn,bhnv,bhnk->bhmvk", probabilities, values, keys
        )
        jacobian = self.scale * (
            moment - output0.unsqueeze(-1) * mean_key.unsqueeze(-2)
        )
        left = torch.einsum("bhmvk,hmkr->bhmvr", jacobian, basis)
        gradient = self.scale * mean_key

        centered = keys.unsqueeze(2) - mean_key.unsqueeze(3)
        projected = torch.einsum("bhmnk,hmkr->bhmnr", centered, basis)
        hessian_small = self.scale**2 * torch.einsum(
            "bhmn,bhmnr,bhmns->bhmrs", probabilities, projected, projected
        )
        reconstructed = torch.einsum("bhmvr,hmkr->bhmvk", left, basis)
        residual = jacobian - reconstructed
        tail_ratio = residual.float().square().mean() / jacobian.float().square().mean().clamp_min(1e-8)
        return {
            "output0": output0,
            "logmass0": logmass0,
            "left": left,
            "gradient": gradient,
            "hessian_small": hessian_small,
            "tail_ratio": tail_ratio,
        }

    def evaluate_page(
        self,
        queries: Tensor,
        compiled: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate a compiled page for Q shaped [B,H,Q,D]."""
        anchors = self.focus_anchors
        basis = self.basis()
        distances = torch.sum(
            (queries.unsqueeze(-2) - anchors.unsqueeze(0).unsqueeze(2)) ** 2,
            dim=-1,
        ).sqrt()
        route = distances.argmin(dim=-1)
        selector = F.one_hot(route, num_classes=anchors.shape[1]).to(queries.dtype)
        selected_anchor = torch.einsum("bhqm,hmd->bhqd", selector, anchors)
        selected_basis = torch.einsum("bhqm,hmdr->bhqdr", selector, basis)
        output0 = torch.einsum("bhqm,bhmd->bhqd", selector, compiled["output0"])
        logmass0 = torch.einsum("bhqm,bhm->bhq", selector, compiled["logmass0"])
        left = torch.einsum("bhqm,bhmdr->bhqdr", selector, compiled["left"])
        gradient = torch.einsum("bhqm,bhmd->bhqd", selector, compiled["gradient"])
        hessian_small = torch.einsum(
            "bhqm,bhmrs->bhqrs", selector, compiled["hessian_small"]
        )
        delta = queries - selected_anchor
        reduced = torch.einsum("bhqdr,bhqd->bhqr", selected_basis, delta)
        output = output0 + torch.einsum("bhqdr,bhqr->bhqd", left, reduced)
        logmass = logmass0 + torch.sum(gradient * delta, dim=-1)
        logmass = logmass + 0.5 * torch.einsum(
            "bhqr,bhqrs,bhqs->bhq", reduced, hessian_small, reduced
        )
        selected_distance = distances.gather(-1, route.unsqueeze(-1)).squeeze(-1)
        return output, logmass, route, selected_distance

    def _focus_suffix(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cold_len: int,
    ) -> tuple[Tensor, AttentionAux]:
        exact, _ = self._exact_causal(q, k, v)
        if cold_len <= 0 or cold_len >= q.shape[-2]:
            return exact, AttentionAux.zeros(exact)

        q_suffix = q[:, :, cold_len:, :]
        k_cold, v_cold = k[:, :, :cold_len, :], v[:, :, :cold_len, :]
        compiled = self.compile_page(k_cold, v_cold)
        approx_cold, approx_logmass, _, route_distance = self.evaluate_page(q_suffix, compiled)

        cold_scores = torch.matmul(q_suffix, k_cold.transpose(-1, -2)) * self.scale
        exact_cold_logmass = torch.logsumexp(cold_scores.float(), dim=-1).to(q.dtype)
        exact_cold_weights = torch.softmax(cold_scores.float(), dim=-1).to(q.dtype)
        exact_cold = torch.matmul(exact_cold_weights, v_cold)

        k_hot, v_hot = k[:, :, cold_len:, :], v[:, :, cold_len:, :]
        hot_scores = torch.matmul(q_suffix, k_hot.transpose(-1, -2)) * self.scale
        suffix = q_suffix.shape[-2]
        causal = torch.ones(suffix, suffix, dtype=torch.bool, device=q.device).triu(1)
        hot_scores = hot_scores.masked_fill(causal, torch.finfo(hot_scores.dtype).min)
        hot_logmass = torch.logsumexp(hot_scores.float(), dim=-1).to(q.dtype)
        hot_weights = torch.softmax(hot_scores.float(), dim=-1).to(q.dtype)
        hot_output = torch.matmul(hot_weights, v_hot)

        merged_logmass = torch.logaddexp(approx_logmass, hot_logmass)
        merged = (
            torch.exp(approx_logmass - merged_logmass).unsqueeze(-1) * approx_cold
            + torch.exp(hot_logmass - merged_logmass).unsqueeze(-1) * hot_output
        )
        output = exact.clone()
        output[:, :, cold_len:, :] = merged

        teacher = exact_cold.detach()
        output_nmse = (approx_cold - teacher).float().square().mean() / teacher.float().square().mean().clamp_min(1e-6)
        logmass_mse = (approx_logmass - exact_cold_logmass.detach()).float().square().mean()
        aux = AttentionAux(
            output_nmse=output_nmse,
            logmass_mse=logmass_mse,
            tail_ratio=compiled["tail_ratio"],
            route_distance=route_distance.float().mean(),
        )
        return output, aux

    def forward(
        self,
        x: Tensor,
        *,
        focus_mode: bool = False,
        focus_cold_len: int | None = None,
        focus_alpha: float = 1.0,
        return_trace: bool = False,
        return_attention: bool = False,
        value_override: Tensor | None = None,
        value_override_head: int | None = None,
    ) -> tuple[Tensor, AttentionAux, dict[str, Tensor] | None]:
        q, k, v = self.project_qkv(x)
        if value_override is not None:
            if value_override_head is None or not 0 <= value_override_head < self.n_heads:
                raise ValueError("a valid value_override_head is required")
            expected = (x.shape[0], x.shape[1], self.head_dim)
            if tuple(value_override.shape) != expected:
                raise ValueError(f"value_override expected {expected}, got {tuple(value_override.shape)}")
            v = v.clone()
            v[:, value_override_head] = value_override.to(device=v.device, dtype=v.dtype)
        exact, _ = self._exact_causal(q, k, v)
        aux = AttentionAux.zeros(x)
        attention = exact
        if focus_mode and focus_cold_len is not None:
            focus_attention, aux = self._focus_suffix(q, k, v, focus_cold_len)
            attention = exact.lerp(focus_attention, float(focus_alpha))
        batch, _, length, _ = attention.shape
        projection_attention = attention
        if value_override is not None and value_override_head is not None:
            # The identity-code head is a pointer channel, not a semantic value
            # head.  Keep its random code out of the residual stream while
            # exposing it to the dedicated copy decoder below.
            projection_attention = attention.clone()
            projection_attention[:, value_override_head] = 0
        projected = projection_attention.transpose(1, 2).reshape(batch, length, -1)
        trace = None
        if return_trace or return_attention:
            trace = {}
            if return_trace:
                trace.update({"q": q.detach(), "k": k.detach(), "v": v.detach()})
            if return_attention:
                # Kept attached to the graph for memory-code pointer training.
                trace["attention_heads"] = attention
        return self.o_proj(projected), aux, trace


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, *, memory_router: bool = False) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = FocusSelfAttention(config, memory_router=memory_router)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor, **kwargs: Any) -> tuple[Tensor, AttentionAux, dict[str, Tensor] | None]:
        attention, aux, trace = self.attn(self.attn_norm(x), **kwargs)
        x = x + attention
        x = x + self.ffn(self.ffn_norm(x))
        return x, aux, trace


class FocusTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.register_buffer(
            "position_embedding",
            sinusoidal_positions(config.max_seq_len, config.d_model),
            persistent=False,
        )
        self.layers = nn.ModuleList([
            Block(
                config,
                memory_router=(config.memory_router_active and index == config.n_layers - 1),
            )
            for index in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        if config.memory_code_enabled:
            self.register_buffer(
                "memory_codebook",
                deterministic_memory_codes(
                    config.vocab_size, config.head_dim, config.memory_code_seed
                ),
                persistent=False,
            )
            self.copy_gate = nn.Linear(config.d_model, 1, bias=True)
        else:
            self.register_buffer(
                "memory_codebook", torch.empty(0, config.head_dim), persistent=False
            )
            self.copy_gate = None
        self.apply(self._init_weights)
        if self.copy_gate is not None:
            nn.init.zeros_(self.copy_gate.weight)
            nn.init.constant_(self.copy_gate.bias, -3.0)
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, std=0.02 / math.sqrt(2 * config.n_layers))
            nn.init.normal_(layer.ffn.down.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        focus_mode: bool = False,
        focus_cold_len: int | None = None,
        focus_alpha: float = 1.0,
        return_traces: bool = False,
    ) -> ModelOutput:
        batch, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError(f"sequence length {length} exceeds {self.config.max_seq_len}")
        positions = self.position_embedding[:length].to(self.token_embedding.weight.dtype)
        x = self.token_embedding(input_ids) + positions.unsqueeze(0)
        traces: list[dict[str, Tensor]] | None = [] if return_traces else None
        aux_items: list[AttentionAux] = []
        copy_attention: Tensor | None = None
        memory_values = (
            self.memory_codebook[input_ids]
            if self.config.memory_code_enabled
            else None
        )
        for layer_index, layer in enumerate(self.layers):
            is_memory_layer = (
                self.config.memory_code_enabled
                and layer_index == self.config.n_layers - 1
            )
            x, aux, trace = layer(
                x,
                focus_mode=focus_mode,
                focus_cold_len=focus_cold_len,
                focus_alpha=focus_alpha,
                return_trace=return_traces,
                return_attention=is_memory_layer,
                value_override=memory_values if is_memory_layer else None,
                value_override_head=(
                    self.config.memory_code_head if is_memory_layer else None
                ),
            )
            aux_items.append(aux)
            if is_memory_layer:
                if trace is None or "attention_heads" not in trace:
                    raise RuntimeError("memory-code attention state was not returned")
                copy_attention = trace["attention_heads"][:, self.config.memory_code_head]
            if traces is not None and trace is not None:
                traces.append({name: trace[name] for name in ("q", "k", "v")})
        normalized = self.final_norm(x)
        base_logits = self.lm_head(normalized)
        copy_logits: Tensor | None = None
        copy_gate: Tensor | None = None
        logits = base_logits
        if self.config.memory_code_enabled:
            if copy_attention is None or self.copy_gate is None:
                raise RuntimeError("memory-code state is unavailable")
            copy_state = F.normalize(copy_attention.float(), dim=-1)
            copy_logits = self.config.memory_code_scale * torch.einsum(
                "bld,vd->blv", copy_state, self.memory_codebook.float()
            )
            copy_gate = torch.sigmoid(self.copy_gate(normalized).float())
            logits = base_logits.float() + copy_gate * copy_logits
            logits = logits.to(base_logits.dtype)
        loss = None
        if labels is not None:
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).view(batch, length)
            if loss_mask is None:
                loss = token_loss.mean()
            else:
                denominator = loss_mask.sum().clamp_min(1.0)
                loss = (token_loss * loss_mask).sum() / denominator
        keys = ("output_nmse", "logmass_mse", "tail_ratio", "route_distance")
        aux_dict = {
            key: torch.stack([getattr(item, key) for item in aux_items]).mean()
            for key in keys
        }
        aux_dict["copy_gate_mean"] = (
            copy_gate.mean() if copy_gate is not None else logits.new_zeros(())
        )
        return ModelOutput(
            logits=logits, loss=loss, aux=aux_dict, traces=traces,
            copy_logits=copy_logits, copy_gate=copy_gate,
        )

    @torch.no_grad()
    def initialize_memory_router_from_shared(self) -> None:
        """Warm-start the dedicated memory router from the shared Q/K rows.

        This is intended for upgrading a legacy memory-code checkpoint.  The
        normalization changes the effective logits, so the initialization is
        only a stable starting point; the router should still be contrastively
        trained on held-out layouts.
        """
        if not self.config.memory_router_active:
            raise RuntimeError("memory router is not active")
        attention = self.layers[-1].attn
        if attention.memory_q_proj is None or attention.memory_k_proj is None:
            raise RuntimeError("final attention layer has no memory router")
        head = self.config.memory_code_head
        start = head * self.config.head_dim
        end = start + self.config.head_dim
        attention.memory_q_proj.weight.copy_(attention.q_proj.weight[start:end])
        attention.memory_k_proj.weight.copy_(attention.k_proj.weight[start:end])
        attention.memory_router_log_temperature.fill_(
            math.log(self.config.memory_router_temperature)
        )

    @torch.no_grad()
    def initialize_focus_head_geometry(
        self,
        batches: list[Tensor],
        *,
        layer_index: int,
        head: int,
        kmeans_iterations: int = 10,
        max_points: int = 4096,
        seed: int = 0,
    ) -> None:
        """Reinitialize one head's FOCUS atlas while preserving every other head."""
        if not 0 <= layer_index < self.config.n_layers:
            raise ValueError("invalid layer_index")
        if not 0 <= head < self.config.n_heads:
            raise ValueError("invalid head")
        snapshots = [
            (
                layer.attn.focus_anchors.detach().clone(),
                layer.attn.focus_basis_raw.detach().clone(),
                layer.attn.focus_radius.detach().clone(),
            )
            for layer in self.layers
        ]
        self.initialize_focus_geometry(
            batches,
            kmeans_iterations=kmeans_iterations,
            max_points=max_points,
            seed=seed,
        )
        for index, (layer, snapshot) in enumerate(zip(self.layers, snapshots)):
            anchors, basis, radius = snapshot
            if index != layer_index:
                layer.attn.focus_anchors.copy_(anchors)
                layer.attn.focus_basis_raw.copy_(basis)
                layer.attn.focus_radius.copy_(radius)
                continue
            keep = torch.ones(self.config.n_heads, dtype=torch.bool, device=anchors.device)
            keep[head] = False
            layer.attn.focus_anchors[keep].copy_(anchors[keep])
            layer.attn.focus_basis_raw[keep].copy_(basis[keep])
            layer.attn.focus_radius[keep].copy_(radius[keep])

    @torch.no_grad()
    def initialize_focus_geometry(
        self,
        batches: list[Tensor],
        *,
        kmeans_iterations: int = 10,
        max_points: int = 4096,
        seed: int = 0,
    ) -> None:
        """Initialize anchors/bases from real query traces.

        Anchors are Lloyd k-means centers in each layer/head query space.  Each
        patch basis is initialized with principal components of the routed
        query cloud.  This removes the arbitrary unit-sphere initialization
        before post-hoc fitting or FOCUS-Native training.
        """
        if not batches:
            raise ValueError("at least one calibration batch is required")
        was_training = self.training
        self.eval()
        traces_by_layer: list[list[Tensor]] = [[] for _ in self.layers]
        for input_ids in batches:
            output = self(input_ids, return_traces=True)
            assert output.traces is not None
            for layer_index, trace in enumerate(output.traces):
                traces_by_layer[layer_index].append(trace["q"].detach().float().cpu())

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for layer_index, layer in enumerate(self.layers):
            q = torch.cat(traces_by_layer[layer_index], dim=0)  # [B,H,L,D]
            for head in range(self.config.n_heads):
                points = q[:, head].reshape(-1, self.config.head_dim)
                if points.shape[0] > max_points:
                    permutation = torch.randperm(points.shape[0], generator=generator)[:max_points]
                    points = points[permutation]
                patches = self.config.focus_patches
                if points.shape[0] < patches:
                    raise ValueError("not enough query points to initialize patches")
                initial = torch.randperm(points.shape[0], generator=generator)[:patches]
                centers = points[initial].clone()
                assignment = torch.zeros(points.shape[0], dtype=torch.long)
                for _ in range(kmeans_iterations):
                    distances = torch.cdist(points, centers)
                    assignment = distances.argmin(dim=-1)
                    new_centers = centers.clone()
                    for patch in range(patches):
                        members = points[assignment == patch]
                        if members.numel():
                            new_centers[patch] = members.mean(dim=0)
                        else:
                            farthest = distances.min(dim=-1).values.argmax()
                            new_centers[patch] = points[farthest]
                    if torch.allclose(new_centers, centers, rtol=1e-4, atol=1e-5):
                        centers = new_centers
                        break
                    centers = new_centers
                layer.attn.focus_anchors[head].copy_(
                    centers.to(layer.attn.focus_anchors.device, layer.attn.focus_anchors.dtype)
                )

                rank = self.config.focus_rank
                for patch in range(patches):
                    members = points[assignment == patch]
                    if members.shape[0] >= 2:
                        centered = members - members.mean(dim=0, keepdim=True)
                        try:
                            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                            candidate = vh[:rank].transpose(0, 1)
                        except RuntimeError:
                            candidate = torch.randn(
                                self.config.head_dim, rank, generator=generator
                            )
                    else:
                        candidate = torch.randn(
                            self.config.head_dim, rank, generator=generator
                        )
                    if candidate.shape[1] < rank:
                        padding = torch.randn(
                            self.config.head_dim,
                            rank - candidate.shape[1],
                            generator=generator,
                        )
                        candidate = torch.cat([candidate, padding], dim=1)
                    orthogonal, _ = torch.linalg.qr(candidate.float(), mode="reduced")
                    layer.attn.focus_basis_raw[head, patch].copy_(
                        orthogonal[:, :rank].to(
                            layer.attn.focus_basis_raw.device,
                            layer.attn.focus_basis_raw.dtype,
                        )
                    )
        self.calibrate_focus_radii(batches, quantile=0.995, margin=1.10)
        self.train(was_training)

    @torch.no_grad()
    def calibrate_focus_radii(
        self,
        batches: list[Tensor],
        quantile: float = 0.995,
        margin: float = 1.10,
    ) -> None:
        if not 0 < quantile <= 1:
            raise ValueError("quantile must be in (0,1]")
        collected: list[list[list[Tensor]]] = [
            [[ ] for _ in range(self.config.focus_patches)] for _ in range(self.config.n_layers * self.config.n_heads)
        ]
        for input_ids in batches:
            output = self(input_ids, return_traces=True)
            assert output.traces is not None
            for layer_index, trace in enumerate(output.traces):
                q = trace["q"]
                anchors = self.layers[layer_index].attn.focus_anchors
                distances = torch.sum(
                    (q.unsqueeze(-2) - anchors.unsqueeze(0).unsqueeze(2)) ** 2,
                    dim=-1,
                ).sqrt()
                route = distances.argmin(dim=-1)
                selected = distances.gather(-1, route.unsqueeze(-1)).squeeze(-1)
                for head in range(self.config.n_heads):
                    slot = layer_index * self.config.n_heads + head
                    for patch in range(self.config.focus_patches):
                        vals = selected[:, head][route[:, head] == patch]
                        if vals.numel():
                            collected[slot][patch].append(vals.cpu())
        for layer_index, layer in enumerate(self.layers):
            radii = torch.empty_like(layer.attn.focus_radius)
            for head in range(self.config.n_heads):
                slot = layer_index * self.config.n_heads + head
                all_head = [x for patch in collected[slot] for x in patch]
                global_values = torch.cat(all_head) if all_head else torch.tensor([1.0])
                global_radius = torch.quantile(global_values, quantile)
                for patch in range(self.config.focus_patches):
                    values = collected[slot][patch]
                    radius = torch.quantile(torch.cat(values), quantile) if values else global_radius
                    radii[head, patch] = radius * margin
            layer.attn.focus_radius.copy_(radii.to(layer.attn.focus_radius.device))
