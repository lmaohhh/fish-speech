import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import click
import torch
import torch.nn as nn
from loguru import logger

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torchao
from torchao.quantization import fpx_weight_only, quantize_
from fish_speech.models.text2semantic.llama import DualARTransformer


def is_slow_transformer_linear(mod: nn.Module, fqn: str) -> bool:
    """
    Bộ lọc chính xác: Chỉ lượng tử hóa các tầng Linear của Slow Transformer.
    Bảo vệ 100% Fast Transformer, Codec, Embedding và Norms ở chuẩn bfloat16.
    """
    if not isinstance(mod, nn.Linear):
        return False

    name = fqn.lower()
    if "fast_layers" in name or "fast_output" in name or "embeddings" in name or "head" in name:
        return False

    tokens = set(re.split(r"[._/]", name))
    protected_tokens = {"codebook", "embeddings", "embed", "norm", "bias"}
    if bool(tokens & protected_tokens):
        return False

    return True


def quantize_fish_speech_torchao_nvfp4(
    checkpoint_path: Path,
    output_dir: Path
):
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📂 Nạp mô hình gốc từ: {checkpoint_path}")
    logger.info(f"💾 Thư mục xuất TorchAO NVFP4: {output_dir}")

    # Copy các file cấu hình và từ điển
    for file in checkpoint_path.glob("*"):
        if file.is_file() and file.suffix in [".json", ".jinja", ".tiktoken", ".txt", ".pth"]:
            if file.name != "model.pth":
                dest = output_dir / file.name
                if not dest.exists():
                    shutil.copyfile(file, dest)

    logger.info("⏳ Đang nạp mô hình DualARTransformer...")
    model = DualARTransformer.from_pretrained(str(checkpoint_path), load_weights=True)
    model = model.to(device="cpu", dtype=torch.bfloat16)

    logger.info(f"🚀 Đang áp dụng lượng tử hóa TorchAO FP4 (E2M1 NVFP4) cho Slow Transformer...")
    t0 = time.time()
    quantize_(model, fpx_weight_only(ebits=2, mbits=1), filter_fn=is_slow_transformer_linear)
    logger.info(f"⚡ Lượng tử hóa TorchAO hoàn tất trong {time.time() - t0:.2f}s!")

    out_file = output_dir / "model.pth"
    logger.info(f"💾 Đang lưu mô hình TorchAO ra đĩa: {out_file}...")
    torch.save(model.state_dict(), out_file)

    # Ghi metadata
    metadata = {
        "quant_engine": "torchao",
        "quant_type": "fpx_weight_only_e2m1_nvfp4",
        "ebits": 2,
        "mbits": 1,
        "slow_ar_format": "torchao_affine_quantized_fp4_e2m1",
        "fast_ar_format": "torch.bfloat16_preserved",
        "torchao_version": torchao.__version__,
        "torch_version": torch.__version__
    }
    with open(output_dir / "quant_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    orig_size_mb = os.path.getsize(checkpoint_path / "model.pth") / (1024 * 1024)
    quant_size_mb = os.path.getsize(out_file) / (1024 * 1024)

    logger.info("=" * 70)
    logger.info("🎉 QUÁ TRÌNH LƯỢNG TỬ HÓA TORCHAO NVFP4 (E2M1) ĐÃ HOÀN TẤT!")
    logger.info(f"📊 Dung lượng gốc (BF16)       : {orig_size_mb:.2f} MB (~{orig_size_mb/1024:.2f} GB)")
    logger.info(f"📊 Dung lượng sau nén (TorchAO) : {quant_size_mb:.2f} MB (~{quant_size_mb/1024:.2f} GB)")
    logger.info(f"📉 Tỷ lệ cắt giảm bộ nhớ       : -{100 * (1 - quant_size_mb/orig_size_mb):.1f}%")
    logger.info("=" * 70)


@click.command()
@click.option("--checkpoint-path", type=click.Path(exists=True), required=True, help="Input FP16/BF16 model directory")
@click.option("--output", type=str, required=True, help="Output TorchAO NVFP4 model directory")
def main(checkpoint_path, output):
    quantize_fish_speech_torchao_nvfp4(Path(checkpoint_path), Path(output))


if __name__ == "__main__":
    main()
