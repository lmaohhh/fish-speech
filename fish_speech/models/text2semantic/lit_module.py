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

        # 1. Direct Vectorized Base Loss (Zero Python loops, Native TPU MXU acceleration)
        lm_head = (
            self.model.embeddings
            if self.model.config.tie_word_embeddings
            else self.model.output
        )
        if isinstance(lm_head, nn.Embedding):
            base_logits = F.linear(outputs.hidden_states, lm_head.weight)
        elif isinstance(lm_head, nn.Linear):
            base_logits = lm_head(outputs.hidden_states)
        elif isinstance(lm_head, torch.Tensor):
            base_logits = F.linear(outputs.hidden_states, lm_head)
        else:
            base_logits = F.linear(outputs.hidden_states, lm_head.weight)

        base_loss = F.cross_entropy(
            base_logits.view(-1, base_logits.size(-1)).float(),
            labels[:, 0].reshape(-1),
            ignore_index=-100,
        )

        # 2. Direct Vectorized Semantic Loss
        token_ids = labels[:, 0]
        semantic_mask = (token_ids >= self.model.tokenizer.semantic_begin_id) & (
            token_ids <= self.model.tokenizer.semantic_end_id
        )
        all_codebook_labels = labels[:, 1 : 1 + self.model.config.num_codebooks]
        all_codebook_labels_permuted = all_codebook_labels.permute(0, 2, 1)
        filtered_codebook_labels = all_codebook_labels_permuted[semantic_mask]

        if outputs.fast_hidden_states is not None:
            fast_logits = self.model.fast_output(outputs.fast_hidden_states)
            semantic_loss = F.cross_entropy(
                fast_logits.view(-1, fast_logits.size(-1)).float(),
                filtered_codebook_labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            semantic_loss = torch.tensor(0.0, device=base_loss.device)

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
            "step",
            float(self.global_step),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
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

        # 3. Vectorized Top-5 Accuracy (Single operation without slice loops)
        if not is_train and outputs.fast_hidden_states is not None:
            accuracy = self.get_vectorized_accuracy(fast_logits, filtered_codebook_labels)
            self.log(
                f"{stage}/top_5_accuracy",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
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
