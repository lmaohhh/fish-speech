import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer:
    def __init__(self, r: int, lora_alpha: float, lora_dropout: float = 0.0, merge_weights: bool = True):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else (lambda x: x)
        self.merged = False
        self.merge_weights = merge_weights


class LoRAEmbedding(nn.Embedding, LoRALayer):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "lora_A"):
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1).to(self.weight.dtype) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1).to(self.weight.dtype) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x)
        if self.r > 0 and not self.merged:
            after_A = F.embedding(
                x,
                self.lora_A.transpose(0, 1).to(dtype=result.dtype),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            result = result + (after_A @ self.lora_B.transpose(0, 1).to(dtype=result.dtype)) * self.scaling
        return result


class LoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        super().__init__(in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).to(self.weight.dtype) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).to(self.weight.dtype) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, bias=self.bias)
        if self.r > 0 and not self.merged:
            result = result + (self.lora_dropout(x) @ self.lora_A.to(dtype=x.dtype).t() @ self.lora_B.to(dtype=x.dtype).t()) * self.scaling
        return result


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True


@dataclass
class LoraConfig:
    r: int
    lora_alpha: float
    lora_dropout: float = 0.0
    # Valid values: "attention", "mlp", "embeddings", "output",
    #               "fast_attention", "fast_mlp", "fast_embeddings", "fast_output"
    target_modules: list = field(
        default_factory=lambda: ["attention", "fast_attention"]
    )


def _replace_embedding(old_embed, lora_config):
    new_embed = LoRAEmbedding(
        num_embeddings=old_embed.num_embeddings,
        embedding_dim=old_embed.embedding_dim,
        padding_idx=old_embed.padding_idx,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
    )
    new_embed.weight.data.copy_(old_embed.weight.data)
    return new_embed


def setup_lora(model, lora_config):
    targets = set(lora_config.target_modules)
    linears = []

    # Slow transformer: targeted by unprefixed names (e.g. "attention")
    slow_attention = "attention" in targets
    slow_mlp = "mlp" in targets
    slow_embeddings = "embeddings" in targets
    slow_output = "output" in targets

    # Fast transformer: targeted by unprefixed names (backwards compat) OR "fast_*"
    fast_attention = slow_attention or "fast_attention" in targets
    fast_mlp = slow_mlp or "fast_mlp" in targets
    fast_embeddings = slow_embeddings or "fast_embeddings" in targets
    fast_output = slow_output or "fast_output" in targets

    if slow_embeddings:
        model.embeddings = _replace_embedding(model.embeddings, lora_config)
        model.codebook_embeddings = _replace_embedding(
            model.codebook_embeddings, lora_config
        )

    if slow_output and hasattr(model, "output"):
        linears.append((model, "output"))

    for layer in model.layers:
        if slow_attention:
            linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
        if slow_mlp:
            linears.extend(
                [
                    (layer.feed_forward, "w1"),
                    (layer.feed_forward, "w2"),
                    (layer.feed_forward, "w3"),
                ]
            )

    if hasattr(model, "fast_layers"):
        if fast_embeddings:
            model.fast_embeddings = _replace_embedding(
                model.fast_embeddings, lora_config
            )
        if fast_output:
            linears.append((model, "fast_output"))

        for layer in model.fast_layers:
            if fast_attention:
                linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
            if fast_mlp:
                linears.extend(
                    [
                        (layer.feed_forward, "w1"),
                        (layer.feed_forward, "w2"),
                        (layer.feed_forward, "w3"),
                    ]
                )

    for module, layer_name in linears:
        old_linear = getattr(module, layer_name)
        updated_linear = LoRALinear(
            in_features=old_linear.in_features,
            out_features=old_linear.out_features,
            bias=old_linear.bias is not None,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
        )
        updated_linear.weight.data.copy_(old_linear.weight.data)
        if old_linear.bias is not None:
            updated_linear.bias.data.copy_(old_linear.bias.data)
        setattr(module, layer_name, updated_linear)

    # Mark only the LoRA layers as trainable
    mark_only_lora_as_trainable(model, bias="none")


def get_merged_state_dict(model):
    model.eval()
    state_dict = model.state_dict()
    for name in list(state_dict.keys()):
        if "lora" in name:
            state_dict.pop(name)
    return state_dict
