import os
import shutil
import threading
from pathlib import Path
from typing import Optional

from lightning.pytorch.callbacks import Callback
from loguru import logger


class AutoSyncCallback(Callback):
    """
    Automatically copies checkpoints to /kaggle/working/ for instant 1-click download
    and optionally syncs to Hugging Face Hub in a non-blocking background thread.
    """

    def __init__(
        self,
        export_dir: str = "/kaggle/working",
        hf_repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.export_dir = Path(export_dir)
        self.hf_repo_id = hf_repo_id or os.environ.get("HF_REPO_ID")
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

    def _async_hf_upload(self, local_path: str, remote_filename: str):
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.hf_token)
            api.create_repo(repo_id=self.hf_repo_id, exist_ok=True, private=True)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"checkpoints/{remote_filename}",
                repo_id=self.hf_repo_id,
            )
            logger.info(f"☁️ [CloudSync] Uploaded {remote_filename} to HuggingFace {self.hf_repo_id}")
        except Exception as e:
            logger.warning(f"⚠️ [CloudSync] HuggingFace upload failed: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Trigger copy whenever a new checkpoint is written in ckpt_dir
        ckpt_dir = Path(trainer.default_root_dir) / "checkpoints"
        if not ckpt_dir.exists():
            return

        if self.export_dir.exists():
            for ckpt_file in ckpt_dir.glob("*.ckpt"):
                target = self.export_dir / ckpt_file.name
                if not target.exists() or target.stat().st_mtime < ckpt_file.stat().st_mtime:
                    try:
                        shutil.copyfile(ckpt_file, target)
                        # Also update last checkpoint alias
                        last_target = self.export_dir / "remielle_lora_last.ckpt"
                        shutil.copyfile(ckpt_file, last_target)
                        logger.info(f"💾 [AutoSync] Synced {ckpt_file.name} to {self.export_dir}")

                        if self.hf_repo_id and self.hf_token:
                            threading.Thread(
                                target=self._async_hf_upload,
                                args=(str(ckpt_file), ckpt_file.name),
                                daemon=True,
                            ).start()
                    except Exception as e:
                        logger.warning(f"AutoSync copy warning: {e}")
