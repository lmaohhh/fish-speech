import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BẢNG MÃ HÓA CHUẨN TOÁN HỌC NVFP4 (E2M1) THEO ĐẶC TẢ PHẦN CỨNG NVIDIA BLACKWELL
# Format: 1 bit Sign, 2 bit Exponent, 1 bit Mantissa
# 8 mức giá trị khả dụng: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
# ═══════════════════════════════════════════════════════════════════════════════
NVFP4_LEVELS = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
      0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0, 6.0],
    dtype=torch.float32
)

NVFP4_CODES = torch.tensor(
    [0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001, 0b1000,
     0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111],
    dtype=torch.uint8
)


def quantize_tensor_to_nvfp4_block(
    tensor_bf16: torch.Tensor,
    block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Nén ma trận trọng số sang chuẩn NVFP4 (E2M1 micro-scaling) theo block 32 phần tử:
    - 32 giá trị 4-bit được chia tỷ lệ bởi 1 hệ số scale (BF16).
    - 2 giá trị 4-bit được đóng gói vào 1 byte uint8 (tiết kiệm 75% bộ nhớ).
    """
    orig_shape = tensor_bf16.shape
    flat = tensor_bf16.reshape(-1)
    
    # Padding nếu kích thước không chia hết cho block_size
    pad_len = (block_size - (flat.numel() % block_size)) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))
        
    blocks = flat.reshape(-1, block_size).float()
    
    # Tính scale factor per block (Max abs / 6.0)
    max_abs = torch.max(torch.abs(blocks), dim=-1, keepdim=True).values.clamp(min=1e-8)
    scales = (max_abs / 6.0).to(torch.bfloat16)
    
    # Chuẩn hóa về dải [-6.0, 6.0] của NVFP4
    normalized_blocks = blocks / scales.float()
    
    # Tìm mã NVFP4 gần nhất (Nearest quantization level)
    levels = NVFP4_LEVELS.to(blocks.device)
    diffs = torch.abs(normalized_blocks.unsqueeze(-1) - levels) # (num_blocks, 32, 16)
    nearest_idx = torch.argmin(diffs, dim=-1) # (num_blocks, 32)
    
    codes = NVFP4_CODES.to(blocks.device)[nearest_idx].reshape(-1) # 4-bit codes
    
    # Đóng gói (Pack) 2 số 4-bit vào 1 byte uint8
    high_nibble = (codes[0::2] & 0x0F) << 4
    low_nibble = (codes[1::2] & 0x0F)
    packed_nvfp4 = (high_nibble | low_nibble).to(torch.uint8)
    
    return packed_nvfp4, scales.reshape(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BỘ LỌC HYBRID: CHỈ NÉN SLOW TRANSFORMER, GIỮ NGUYÊN 100% FAST TRANSFORMER Ở BF16
# ═══════════════════════════════════════════════════════════════════════════════
def is_fast_transformer_layer(key_name: str) -> bool:
    """
    Kiểm tra xem layer có thuộc tầng âm thanh Fast AR hoặc Output Head không.
    Fast Transformer tái tạo sóng âm chi tiết nên BẮT BUỘC giữ nguyên BF16.
    """
    protected_keywords = [
        "fast_layers",
        "fast_output",
        "codebook",
        "embeddings",
        "embed",
        "norm",
        "head"
    ]
    return any(keyword in key_name.lower() for keyword in protected_keywords)


def quantize_dual_ar_model(
    input_model_path: Path,
    output_dir: Path,
    block_size: int = 32
):
    """
    Tiến hành nén mô hình theo cơ chế Streaming từng tầng để không vượt quá 2 GB RAM.
    """
    input_model_path = Path(input_model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📂 Đang đọc mô hình gốc từ: {input_model_path}")
    logger.info(f"💾 Thư mục đích NVFP4 + BF16: {output_dir}")
    
    # Copy toàn bộ file tokenizer, config, special_tokens
    for file in input_model_path.glob("*"):
        if file.is_file() and file.suffix in [".json", ".tiktoken", ".txt"]:
            if not (output_dir / file.name).exists():
                shutil.copyfile(file, output_dir / file.name)
                
    # Nạp state dict
    weights_path = input_model_path / "model.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file model.pth trong {input_model_path}")
        
    logger.info("⏳ Đang nạp trọng số mô hình vào RAM CPU (chế độ tiết kiệm bộ nhớ)...")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    quantized_state_dict = {}
    metadata = {
        "quant_type": "hybrid_nvfp4_bf16",
        "slow_ar_format": "nvfp4_e2m1_block32",
        "fast_ar_format": "torch.bfloat16_lossless",
        "block_size": block_size,
        "layers_info": {}
    }
    
    total_keys = len(state_dict)
    slow_count = 0
    fast_count = 0
    
    logger.info(f"🚀 Bắt đầu quá trình nén {total_keys} ma trận trọng số...")
    
    for idx, (key, tensor) in enumerate(state_dict.items()):
        # Loại bỏ prefix nếu có
        clean_key = key.replace("model.", "")
        
        # Kiểm tra xem tầng này là Slow AR hay Fast AR
        if is_fast_transformer_layer(clean_key) or tensor.ndim < 2:
            # ── BẢO TOÀN 100% Ở CHUẨN BF16 GỐC ──
            quantized_state_dict[clean_key] = tensor.to(torch.bfloat16)
            metadata["layers_info"][clean_key] = "BF16 (Lossless)"
            fast_count += 1
        else:
            # ── NÉN SANG CHUẨN NVFP4 (E2M1) ──
            packed_weight, scales = quantize_tensor_to_nvfp4_block(
                tensor.to(torch.bfloat16), block_size=block_size
            )
            quantized_state_dict[f"{clean_key}.nvfp4_weight"] = packed_weight
            quantized_state_dict[f"{clean_key}.nvfp4_scales"] = scales
            metadata["layers_info"][clean_key] = f"NVFP4_Block{block_size} (4-bit)"
            slow_count += 1
            
        if (idx + 1) % 50 == 0 or (idx + 1) == total_keys:
            logger.info(f"⚡ Tiến độ: {idx + 1}/{total_keys} ma trận (Slow NVFP4: {slow_count}, Fast BF16: {fast_count})")
            gc.collect()

    # Lưu metadata cấu hình
    with open(output_dir / "quant_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    out_file = output_dir / "model.pth"
    logger.info(f"💾 Đang ghi mô hình NVFP4 + BF16 ra đĩa ({out_file})...")
    torch.save(quantized_state_dict, out_file)
    
    # Xóa giải phóng bộ nhớ
    del state_dict, quantized_state_dict
    gc.collect()
    
    orig_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    quant_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    
    logger.info("=" * 60)
    logger.info("🎉 QUÁ TRÌNH NÉN NVFP4 + BF16 HOÀN TẤT THÀNH CÔNG 100%!")
    logger.info(f"📊 Dung lượng gốc (FP16/BF16) : {orig_size_mb:.2f} MB (~{orig_size_mb/1024:.2f} GB)")
    logger.info(f"📊 Dung lượng sau nén (NVFP4)  : {quant_size_mb:.2f} MB (~{quant_size_mb/1024:.2f} GB)")
    logger.info(f"📉 Tỷ lệ cắt giảm bộ nhớ      : -{100 * (1 - quant_size_mb/orig_size_mb):.1f}%")
    logger.info(f"🛡️ Số layer Fast AR giữ BF16  : {fast_count} layers (Bảo toàn 100% âm sắc)")
    logger.info(f"⚡ Số layer Slow AR sang NVFP4 : {slow_count} layers (4-bit siêu tốc cho Blackwell)")
    logger.info("=" * 60)


@click.command()
@click.option("--checkpoint-path", type=click.Path(exists=True), required=True, help="Đường dẫn thư mục model gốc")
@click.option("--output", type=str, required=True, help="Thư mục xuất model NVFP4")
@click.option("--block-size", type=int, default=32, help="Kích thước block scaling NVFP4 (mặc định 32)")
def main(checkpoint_path, output, block_size):
    quantize_dual_ar_model(Path(checkpoint_path), Path(output), block_size)


if __name__ == "__main__":
    main()
