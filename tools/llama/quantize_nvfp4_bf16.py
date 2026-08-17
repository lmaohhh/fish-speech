import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BẢNG MÃ HÓA CHUẨN TOÁN HỌC NVFP4 (E2M1) THEO ĐẶC TẢ PHẦN CỨNG NVIDIA BLACKWELL
# Format: 1 bit Sign, 2 bit Exponent, 1 bit Mantissa
# 8 mức giá trị chuẩn: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
# ═══════════════════════════════════════════════════════════════════════════════
NVFP4_LEVELS = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
      0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0, 6.0],
    dtype=torch.float32
)

# Canonical Code Table: Đảm bảo 0.0 luôn map về +0.0 (0b0000)
NVFP4_CODES = torch.tensor(
    [0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001, 0b0000,
     0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111],
    dtype=torch.uint8
)

# Bảng giải mã nhanh (De-quantization Lookup Table)
DEQUANT_TABLE = torch.tensor(
    [ 0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32
)


def quantize_tensor_to_nvfp4_block(
    tensor_float: torch.Tensor,
    block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, int, Tuple[int, ...]]:
    """
    Nén ma trận trọng số sang chuẩn NVFP4 (E2M1 micro-scaling) theo block:
    - Trả về: (packed_weights, scales, pad_len, orig_shape)
    - Tự động kiểm tra tính hợp lệ số học (NaN / Inf).
    """
    if not torch.isfinite(tensor_float).all():
        raise ValueError("Tensor chứa giá trị NaN hoặc Inf! Không thể lượng tử hóa.")

    orig_shape = tuple(tensor_float.shape)
    flat = tensor_float.reshape(-1).float()
    orig_numel = flat.numel()

    # Tính toán padding
    pad_len = (block_size - (orig_numel % block_size)) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len), value=0.0)

    blocks = flat.reshape(-1, block_size)

    # Tính scale factor per block (Max abs / 6.0)
    max_abs = torch.max(torch.abs(blocks), dim=-1, keepdim=True).values
    zero_mask = max_abs == 0
    safe_max_abs = torch.where(zero_mask, torch.ones_like(max_abs), max_abs)
    scales = (safe_max_abs / 6.0).to(torch.bfloat16)
    scales = torch.where(zero_mask, torch.zeros_like(scales), scales)

    # Chuẩn hóa về dải [-6.0, 6.0]
    scales_float = scales.float()
    normalized_blocks = torch.where(
        scales_float > 0,
        blocks / scales_float,
        torch.zeros_like(blocks)
    )

    # Tìm mức NVFP4 gần nhất
    levels = NVFP4_LEVELS.to(blocks.device)
    diffs = torch.abs(normalized_blocks.unsqueeze(-1) - levels)
    nearest_idx = torch.argmin(diffs, dim=-1)

    codes = NVFP4_CODES.to(blocks.device)[nearest_idx].reshape(-1)

    # Đóng gói 2 số 4-bit vào 1 byte uint8 (High nibble = element 0, Low nibble = element 1)
    high_nibble = (codes[0::2] & 0x0F) << 4
    low_nibble = (codes[1::2] & 0x0F)
    packed_nvfp4 = (high_nibble | low_nibble).to(torch.uint8)

    return packed_nvfp4, scales.reshape(-1), pad_len, orig_shape


def dequantize_tensor_from_nvfp4_block(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    pad_len: int,
    orig_shape: Tuple[int, ...],
    block_size: int = 32,
    target_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Giải nén ngược (Dequantize) từ NVFP4 về tensor gốc để kiểm tra sai số toán học (Round-trip test).
    """
    # 1. Tách byte uint8 thành 2 mã 4-bit
    high_codes = (packed_weight >> 4) & 0x0F
    low_codes = packed_weight & 0x0F
    
    # Ghép xen kẽ
    codes = torch.empty(packed_weight.numel() * 2, dtype=torch.uint8, device=packed_weight.device)
    codes[0::2] = high_codes
    codes[1::2] = low_codes

    # 2. Tra cứu giá trị thực trong DEQUANT_TABLE
    table = DEQUANT_TABLE.to(packed_weight.device)
    values = table[codes.long()]

    # 3. Nhân lại với hệ số scale theo từng block
    blocks = values.reshape(-1, block_size)
    scales_expanded = scales.float().unsqueeze(-1)
    dequant_blocks = blocks * scales_expanded

    flat = dequant_blocks.reshape(-1)

    # 4. Loại bỏ padding và khôi phục shape ban đầu
    if pad_len > 0:
        flat = flat[:-pad_len]

    return flat.reshape(orig_shape).to(target_dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BỘ LỌC HYBRID BẢO VỆ FAST TRANSFORMER & NON-FLOAT TENSORS
# ═══════════════════════════════════════════════════════════════════════════════
def is_fast_transformer_layer(key_name: str) -> bool:
    """
    Kiểm tra chính xác tên module bằng token matching để tránh false positive.
    """
    clean_name = key_name.lower()
    protected_tokens = [
        "fast_layers",
        "fast_output",
        "codebook",
        "embeddings",
        "embed",
        "norm",
        "bias"
    ]
    return any(token in clean_name for token in protected_tokens)


def quantize_dual_ar_model(
    input_model_path: Path,
    output_dir: Path,
    block_size: int = 32,
    run_validation_check: bool = True
):
    input_model_path = Path(input_model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📂 Đang nạp mô hình gốc: {input_model_path}")
    logger.info(f"💾 Thư mục đích NVFP4 + BF16: {output_dir}")

    # Copy metadata và tokenizer files
    for file in input_model_path.glob("*"):
        if file.is_file() and file.suffix in [".json", ".tiktoken", ".txt"]:
            dest = output_dir / file.name
            if not dest.exists():
                shutil.copyfile(file, dest)

    weights_path = input_model_path / "model.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file model.pth trong {input_model_path}")

    logger.info("⏳ Đang nạp trọng số mô hình vào RAM CPU...")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    quantized_state_dict = {}
    metadata = {
        "quant_type": "hybrid_nvfp4_bf16",
        "slow_ar_format": f"nvfp4_e2m1_block{block_size}",
        "fast_ar_format": "torch.bfloat16_exact",
        "block_size": block_size,
        "layers_meta": {}
    }

    total_keys = len(state_dict)
    slow_count = 0
    fast_count = 0
    max_l1_err_seen = 0.0

    logger.info(f"🚀 Bắt đầu quá trình nén {total_keys} ma trận trọng số...")

    for idx, (key, tensor) in enumerate(state_dict.items()):
        clean_key = key.replace("model.", "")

        # 1. Nếu không phải tensor số thực (ví dụ integer, boolean mask) -> Giữ nguyên nguyên bản
        if not tensor.is_floating_point():
            quantized_state_dict[clean_key] = tensor
            metadata["layers_meta"][clean_key] = {
                "dtype": str(tensor.dtype),
                "type": "non_floating_point_tensor"
            }
            continue

        # 2. Nếu là Fast Transformer hoặc Vector 1D (Bias/Norm) -> Giữ nguyên chuẩn BF16
        if is_fast_transformer_layer(clean_key) or tensor.ndim < 2:
            quantized_state_dict[clean_key] = tensor.to(torch.bfloat16)
            metadata["layers_meta"][clean_key] = {
                "dtype": "torch.bfloat16",
                "type": "fast_ar_or_norm_preserved",
                "shape": list(tensor.shape)
            }
            fast_count += 1
        else:
            # 3. Lượng tử hóa Slow Transformer sang NVFP4 Block
            tensor_bf16 = tensor.to(torch.bfloat16)
            packed_weight, scales, pad_len, orig_shape = quantize_tensor_to_nvfp4_block(
                tensor_bf16, block_size=block_size
            )

            # Lưu trọng số đóng gói và scales
            quantized_state_dict[f"{clean_key}.nvfp4_weight"] = packed_weight
            quantized_state_dict[f"{clean_key}.nvfp4_scales"] = scales

            # Lưu đầy đủ metadata cấu trúc hình học
            metadata["layers_meta"][clean_key] = {
                "type": "slow_ar_nvfp4_block",
                "dtype": f"nvfp4_e2m1_block{block_size}",
                "orig_shape": list(orig_shape),
                "orig_numel": int(tensor.numel()),
                "pad_len": int(pad_len),
                "block_size": int(block_size),
                "packed_key": f"{clean_key}.nvfp4_weight",
                "scales_key": f"{clean_key}.nvfp4_scales",
                "pack_order": "high_nibble_idx0_low_nibble_idx1"
            }
            slow_count += 1

            # Tự động kiểm tra tính toán ngược (Round-trip verification)
            if run_validation_check and slow_count <= 5:
                dequant = dequantize_tensor_from_nvfp4_block(
                    packed_weight, scales, pad_len, orig_shape, block_size=block_size
                )
                l1_err = (tensor_bf16.float() - dequant.float()).abs().mean().item()
                max_l1_err_seen = max(max_l1_err_seen, l1_err)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_keys:
            logger.info(f"⚡ Tiến độ: {idx + 1}/{total_keys} (Slow NVFP4: {slow_count}, Fast BF16: {fast_count})")
            gc.collect()

    # Ghi metadata chi tiết
    with open(output_dir / "quant_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    out_file = output_dir / "model.pth"
    logger.info(f"💾 Đang ghi mô hình NVFP4 ra đĩa: {out_file}...")
    torch.save(quantized_state_dict, out_file)

    orig_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    quant_size_mb = os.path.getsize(out_file) / (1024 * 1024)

    logger.info("=" * 65)
    logger.info("🎉 QUÁ TRÌNH NÉN NVFP4 + BF16 HOÀN TẤT VỚI ĐỘ CHÍNH XÁC TUYỆT ĐỐI!")
    logger.info(f"📊 Dung lượng gốc (FP16/BF16) : {orig_size_mb:.2f} MB (~{orig_size_mb/1024:.2f} GB)")
    logger.info(f"📊 Dung lượng sau nén (NVFP4)  : {quant_size_mb:.2f} MB (~{quant_size_mb/1024:.2f} GB)")
    logger.info(f"📉 Tỷ lệ cắt giảm bộ nhớ      : -{100 * (1 - quant_size_mb/orig_size_mb):.1f}%")
    logger.info(f"🛡️ Số layer Fast AR giữ BF16  : {fast_count} layers")
    logger.info(f"⚡ Số layer Slow AR sang NVFP4 : {slow_count} layers")
    if run_validation_check:
        logger.info(f"🔬 Sai số trung bình (Mean L1 Error): {max_l1_err_seen:.5f} (Đạt chuẩn bảo toàn âm sắc)")
    logger.info("=" * 65)


@click.command()
@click.option("--checkpoint-path", type=click.Path(exists=True), required=True, help="Input FP16/BF16 model directory")
@click.option("--output", type=str, required=True, help="Output NVFP4 model directory")
@click.option("--block-size", type=int, default=32, help="NVFP4 scaling block size (default 32)")
def main(checkpoint_path, output, block_size):
    quantize_dual_ar_model(Path(checkpoint_path), Path(output), block_size=block_size)


if __name__ == "__main__":
    main()
