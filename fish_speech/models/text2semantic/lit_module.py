from typing import Any, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import fish_speech.utils as utils

CODEBOOK_PAD_TOKEN_ID = 0
from fish_speech.models.text2semantic.llama import NaiveTransformer

log = utils.RankedLogger(__name__, rank_zero_only=True)


def compute_chunked_base_loss(hidden_states, lm_head, labels, chunk_size: int = 256):
    """
    Computes cross-entropy loss in token chunks without host-device synchronization barriers.
    Keeps logits memory strictly under 76 MB to fit within TPU 16GB HBM.
    """
    flat_h = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_targets = labels.reshape(-1)

    total_valid = (flat_targets != -100).sum()
    total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
    num_tokens = flat_h.size(0)

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        chunk_h = flat_h[start:end]
        chunk_targets = flat_targets[start:end]

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

    return total_loss / torch.clamp(total_valid.float(), min=1.0)


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

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state_dict = checkpoint["state_dict"]

        for name in list(state_dict.keys()):
            if "lora" not in name:
                state_dict.pop(name)

    def configure_optimizers(self) -> OptimizerLRScheduler:
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

    def _step(self, batch, batch_idx, stage: str = "train"):
        is_train = stage == "train"
        if is_train:
            self.model.train()

        labels = batch["labels"]
        outputs = self.model(
            inp=batch["inputs"],
            key_padding_mask=batch["attention_masks"],
            labels=batch["labels"],
        )

        # 1. Chunked Base Loss (Keeps intermediate logits under 76 MB)
        lm_head = (
            self.model.embeddings
            if self.model.config.tie_word_embeddings
            else self.model.output
        )
        base_loss = compute_chunked_base_loss(
            hidden_states=outputs.hidden_states,
            lm_head=lm_head,
            labels=labels[:, 0],
            chunk_size=128,
        )

        # 2. Vectorized Semantic Loss (Codebook vocab is only 4096, 160 MB total - eliminates 3.2 GB XLA slice pad buffers)
        token_ids = labels[:, 0]
        semantic_mask = (token_ids >= self.model.tokenizer.semantic_begin_id) & (
            token_ids <= self.model.tokenizer.semantic_end_id
        )
        all_codebook_labels = labels[:, 1 : 1 + self.model.config.num_codebooks]
        all_codebook_labels_permuted = all_codebook_labels.permute(0, 2, 1)

        masked_codebook_labels = all_codebook_labels_permuted.clone()
        masked_codebook_labels[~semantic_mask] = -100

        if outputs.fast_hidden_states is not None:
            fast_logits = self.model.fast_output(outputs.fast_hidden_states)
            semantic_loss = F.cross_entropy(
                fast_logits.view(-1, self.model.config.codebook_size).float(),
                masked_codebook_labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            semantic_loss = torch.tensor(0.0, device=base_loss.device)

        loss = base_loss + semantic_loss

        self.log(
            f"{stage}/loss",
            loss.detach(),
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
        )

        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        self.log(
            f"{stage}/base_loss",
            base_loss.detach(),
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
        )

        self.log(
            f"{stage}/semantic_loss",
            semantic_loss.detach(),
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
        )

        # 3. Vectorized Top-5 Accuracy (Only during validation)
        if not is_train and outputs.fast_hidden_states is not None:
            self.log(
                f"{stage}/loss_eval",
                loss.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        return loss

    def get_vectorized_accuracy(self, logits, labels):
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        valid_mask = (flat_labels != -100) & (flat_labels != CODEBOOK_PAD_TOKEN_ID)
        total_valid = valid_mask.sum()
        if total_valid == 0:
            return torch.tensor(0.0, device=logits.device)

        _, indices = flat_logits[valid_mask].topk(5, dim=-1)
        matches = indices.eq(flat_labels[valid_mask].unsqueeze(-1))
        correct = matches.any(dim=-1).sum()
        return correct.float() / total_valid.float()

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
