from typing import Any, Optional

import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import fish_speech.utils as utils

CODEBOOK_PAD_TOKEN_ID = 0
from fish_speech.models.text2semantic.llama import NaiveTransformer

log = utils.RankedLogger(__name__, rank_zero_only=True)


def compute_chunked_base_loss(hidden_states, lm_head, labels, chunk_size: int = 512):
    """
    Computes cross-entropy loss in small token chunks to avoid materializing
    the massive (seq_len, vocab_size) logits tensor (1.19 GB in FP16).
    Peak memory per chunk: only ~150 MB.
    Mathematically identical to standard cross-entropy.
    """
    flat_h = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_targets = labels.reshape(-1)

    valid_mask = flat_targets != -100
    total_valid = valid_mask.sum()
    if total_valid == 0:
        return torch.tensor(
            0.0,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            requires_grad=True,
        )

    total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
    num_tokens = flat_h.size(0)

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        chunk_h = flat_h[start:end]
        chunk_targets = flat_targets[start:end]

        if (chunk_targets != -100).any():
            if isinstance(lm_head, nn.Linear):
                chunk_logits = lm_head(chunk_h)
            elif isinstance(lm_head, nn.Embedding):
                chunk_logits = F.linear(chunk_h, lm_head.weight)
            elif isinstance(lm_head, torch.Tensor):
                chunk_logits = F.linear(chunk_h, lm_head)
            else:
                chunk_logits = F.linear(chunk_h, lm_head.weight)

            chunk_loss = F.cross_entropy(
                chunk_logits.float(),
                chunk_targets,
                ignore_index=-100,
                reduction="sum",
            )
            total_loss = total_loss + chunk_loss

    return total_loss / torch.clamp(total_valid, min=1)


def compute_chunked_semantic_loss(
    fast_hidden_states: torch.Tensor,
    fast_output: nn.Module,
    filtered_codebook_labels: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Computes semantic cross-entropy loss in small token chunks directly from
    fast_hidden_states, avoiding the 245.76 MB codebook_logits allocation.
    Saves ~450 MB VRAM during training.
    """
    flat_h = fast_hidden_states.reshape(-1, fast_hidden_states.size(-1))
    flat_labels = filtered_codebook_labels.reshape(-1)

    valid_mask = flat_labels != -100
    total_valid = valid_mask.sum()
    if total_valid == 0:
        return torch.tensor(
            0.0,
            device=fast_hidden_states.device,
            dtype=fast_hidden_states.dtype,
            requires_grad=True,
        )

    total_loss = torch.tensor(0.0, device=fast_hidden_states.device, dtype=torch.float32)
    num_tokens = flat_h.size(0)

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        chunk_h = flat_h[start:end]
        chunk_labels = flat_labels[start:end]

        if (chunk_labels != -100).any():
            chunk_logits = fast_output(chunk_h)
            chunk_loss = F.cross_entropy(
                chunk_logits.float(),
                chunk_labels,
                ignore_index=-100,
                reduction="sum",
            )
            total_loss = total_loss + chunk_loss

    return total_loss / torch.clamp(total_valid, min=1)


class TextToSemantic(L.LightningModule):
    def __init__(
        self,
        model: NaiveTransformer,
        optimizer: Any,
        lr_scheduler: Any,
    ):
        super().__init__()

        self.model = model
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

    def forward(self, x):
        return self.model(x)

    def on_save_checkpoint(self, checkpoint):
        # Save only LoRA parameters
        state_dict = checkpoint["state_dict"]
        use_lora = any("lora" in name for name in state_dict.keys())
        if not use_lora:
            return

        for name in list(state_dict.keys()):
            if "lora" not in name:
                state_dict.pop(name)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Get weight decay parameters ONLY for trainable (requires_grad=True) parameters
        # This prevents AdamW from tracking 4.6B frozen parameters which causes huge RAM spikes
        weight_decay_parameters, other_parameters = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ".bias" in name or "norm.weight" in name or ".embeddings." in name:
                other_parameters.append(param)
            else:
                weight_decay_parameters.append(param)

        optimizer = self.optimizer_builder(
            [
                {"params": weight_decay_parameters},
                {"params": other_parameters, "weight_decay": 0.0},
            ]
        )

        # Print the parameters and their weight decay
        for i in optimizer.param_groups:
            log.info(
                f"Set weight decay: {i['weight_decay']} for {len(i['params'])} parameters"
            )

        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    # Copied from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L90
    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, codebook_size, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length, codebook_size)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels.clone()
        loss_mask = labels != -100

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def _step(self, batch, batch_idx, stage: str):
        is_train = stage == "train"

        if is_train:
            # Key part to make lora work
            # Otherwise the parameters are merged, which lead to incorrect gradients
            self.model.train()

        # Do positive and negative samples in the same batch to speed up training
        labels = batch["labels"]
        outputs = self.model(
            inp=batch["inputs"],
            key_padding_mask=batch["attention_masks"],
            labels=batch["labels"],
        )
        token_logits = outputs.token_logits
        codebook_logits = outputs.codebook_logits

        # Generate labels (Chunked Cross-Entropy saves ~2.3 GB VRAM during training)
        if token_logits is None:
            lm_head = (
                self.model.embeddings
                if self.model.config.tie_word_embeddings
                else self.model.output
            )
            base_loss = compute_chunked_base_loss(
                hidden_states=outputs.hidden_states,
                lm_head=lm_head,
                labels=labels[:, 0],
                chunk_size=512,
            )
        else:
            base_loss = F.cross_entropy(
                token_logits.view(-1, token_logits.size(-1)),
                labels[:, 0].reshape(-1),
                ignore_index=-100,
            )

        token_ids = labels[:, 0]
        semantic_mask = (token_ids >= self.model.tokenizer.semantic_begin_id) & (
            token_ids <= self.model.tokenizer.semantic_end_id
        )
        all_codebook_labels = labels[:, 1 : 1 + self.model.config.num_codebooks]
        all_codebook_labels_permuted = all_codebook_labels.permute(0, 2, 1)
        filtered_codebook_labels = all_codebook_labels_permuted[semantic_mask]
        if codebook_logits is None and outputs.fast_hidden_states is not None:
            semantic_loss = compute_chunked_semantic_loss(
                fast_hidden_states=outputs.fast_hidden_states,
                fast_output=self.model.fast_output,
                filtered_codebook_labels=filtered_codebook_labels,
                chunk_size=512,
            )
        else:
            semantic_loss = F.cross_entropy(
                codebook_logits.reshape(-1, codebook_logits.size(-1)),
                filtered_codebook_labels.reshape(-1),
                ignore_index=-100,
            )

        loss = base_loss + semantic_loss

        self.log(
            f"{stage}/loss",
            loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=True,
            logger=True,
            sync_dist=not is_train,
        )

        self.log(
            f"{stage}/base_loss",
            base_loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
        )

        self.log(
            f"{stage}/semantic_loss",
            semantic_loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
        )

        # Top-5 accuracy (memory-efficient chunked computation)
        accuracy = self.get_accuracy(
            codebook_logits,
            filtered_codebook_labels,
            fast_hidden_states=outputs.fast_hidden_states,
            fast_output=self.model.fast_output,
        )
        self.log(
            f"{stage}/top_5_accuracy",
            accuracy,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=True,
            logger=True,
            sync_dist=not is_train,
        )

        return loss

    def get_accuracy(self, logits, labels, fast_hidden_states=None, fast_output=None):
        if logits is None and fast_hidden_states is not None and fast_output is not None:
            flat_h = fast_hidden_states.reshape(-1, fast_hidden_states.size(-1))
            flat_labels = labels.reshape(-1)
            valid_mask = (flat_labels != -100) & (flat_labels != CODEBOOK_PAD_TOKEN_ID)
            if not valid_mask.any():
                return torch.tensor(0.0, device=labels.device)

            v_h = flat_h[valid_mask]
            v_l = flat_labels[valid_mask]
            total_correct = 0
            chunk_size = 512

            with torch.no_grad():
                for start in range(0, v_h.size(0), chunk_size):
                    chunk_logits = fast_output(v_h[start : start + chunk_size])
                    _, indices = chunk_logits.topk(5, dim=-1)
                    total_correct += indices.eq(v_l[start : start + chunk_size].unsqueeze(-1)).sum().item()

            return torch.tensor(total_correct / v_h.size(0), device=labels.device)

        if logits is None:
            return torch.tensor(0.0, device=labels.device)

        mask = (labels != -100) & (labels != CODEBOOK_PAD_TOKEN_ID)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        _, indices = logits.topk(5, dim=-1)
        correct = indices.eq(labels.unsqueeze(-1))
        correct[~mask] = 0
        correct = correct.sum()
        accuracy = correct / mask.sum()

        return accuracy

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
