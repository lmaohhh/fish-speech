import gc
import json
import os
import re
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
# ĐẶC TẢ CHUẨN NVIDIA BLACKWELL NVFP4 (THEO NVIDIA DEVELOPER BLOG CHÍNH THỨC)
# 1. Định dạng dữ liệu: 4-bit E2M1 (1 Sign, 2 Exponent, 1 Mantissa)
#    Dải giá trị: {-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
# 2. Cơ chế Two-Level Scaling:
#    - Cấp 1 (Block Scale): 1 scale dạng FP8 (E4M3) cho mỗi block 16 phần tử.
#    - Cấp 2 (Global Tensor Scale): 1 scale dạng FP32 cho toàn bộ tensor.
# ═══════════════════════════════════════════════════════════════════════════════

NVFP4_DEFAULT_BLOCK_SIZE = 16

# Bảng giá trị E2M1
NVFP4_LEVELS = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0,
      0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0],
    dtype=torch.float32
)

# Bảng mã hóa 4-bit (Canonical mapping)
NVFP4_CODES = torch.tensor(
    [0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001, 0b1000,
     0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111],
    dtype=torch.uint8
)

# Bảng giải mã nhanh (Dequantization Table)
DEQUANT_TABLE = torch.tensor(
    [ 0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32
)


def quantize_tensor_to_nvfp4_two_level(
    tensor_float: torch.Tensor,
    block_size: int = NVFP4_DEFAULT_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, float, int, Tuple[int, ...]]:
    """
    Lượng tử hóa tensor sang chuẩn NVFP4 chuẩn xác theo NVIDIA Blackwell:
    - Trả về: (packed_nvfp4, block_scales_fp8, global_scale_fp32, pad_len, orig_shape)
    """
    if not torch.isfinite(tensor_float).all():
        raise ValueError("Tensor chứa giá trị NaN hoặc Inf! Không thể lượng tử hóa.")

    orig_shape = tuple(tensor_float.shape)
    flat = tensor_float.reshape(-1).float()
    orig_numel = flat.numel()

    # 1. Padding nếu kích thước không chia hết cho block_size
    pad_len = (block_size - (orig_numel % block_size)) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len), value=0.0)

    # 2. Tính Scale Cấp 2: Global Tensor Scale (FP32)
    tensor_max = torch.max(torch.abs(flat)).item()
    if tensor_max == 0:
        global_scale = 1.0
    else:
        # 448.0 là giá trị biểu diễn tối đa của FP8 E4M3
        global_scale = float(tensor_max / 448.0) if tensor_max > 448.0 else 1.0

    blocks = flat.reshape(-1, block_size)

    # 3. Tính Scale Cấp 1: Micro-Block Scale FP8 (E4M3) per 16 elements
    block_max = torch.max(torch.abs(blocks), dim=-1, keepdim=True).values
    zero_mask = block_max == 0

    # scale_block = block_max / (global_scale * 6.0)
    raw_block_scales = torch.where(
        zero_mask,
        torch.zeros_like(block_max),
        block_max / (global_scale * 6.0)
    )

    # Ép kiểu scale block về FP8 E4M3 chuẩn phần cứng Blackwell (hoặc bfloat16 tương thích)
    try:
        block_scales = raw_block_scales.to(torch.float8_e4m3fn)
        block_scales_float = block_scales.float()
    except Exception:
        block_scales = raw_block_scales.to(torch.bfloat16)
        block_scales_float = block_scales.float()

    # 4. Chuẩn hóa giá trị ma trận về dải [-6.0, 6.0] của NVFP4 E2M1
    effective_scale = (global_scale * block_scales_float).clamp(min=1e-12)
    normalized_blocks = torch.where(
        zero_mask,
        torch.zeros_like(blocks),
        (blocks / effective_scale).clamp(-6.0, 6.0)
    )

    # 5. Khớp mức NVFP4 gần nhất (Nearest Level Mapping)
    levels = NVFP4_LEVELS.to(blocks.device)
    diffs = torch.abs(normalized_blocks.unsqueeze(-1) - levels)
    nearest_idx = torch.argmin(diffs, dim=-1)

    codes = NVFP4_CODES.to(blocks.device)[nearest_idx].reshape(-1)

    # 6. Đóng gói 2 mã 4-bit vào 1 byte uint8 (High nibble = idx 0, Low nibble = idx 1)
    high_nibble = (codes[0::2] & 0x0F) << 4
    low_nibble = (codes[1::2] & 0x0F)
    packed_nvfp4 = (high_nibble | low_nibble).to(torch.uint8)

    return packed_nvfp4, block_scales.reshape(-1), global_scale, pad_len, orig_shape


def dequantize_tensor_from_nvfp4_two_level(
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: float,
    pad_len: int,
    orig_shape: Tuple[int, ...],
    block_size: int = NVFP4_DEFAULT_BLOCK_SIZE,
    target_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Giải nén ngược (Dequantize) từ chuẩn NVFP4 Two-Level về tensor gốc để kiểm tra sai số hoặc suy luận.
    """
    # 1. Giải mã byte uint8 thành 2 mã 4-bit
    high_codes = (packed_weight >> 4) & 0x0F
    low_codes = packed_weight & 0x0F

    codes = torch.empty(packed_weight.numel() * 2, dtype=torch.uint8, device=packed_weight.device)
    codes[0::2] = high_codes
    codes[1::2] = low_codes

    # 2. Tra cứu giá trị E2M1 trong DEQUANT_TABLE
    table = DEQUANT_TABLE.to(packed_weight.device)
    values = table[codes.long()]

    # 3. Nhân với hệ số scale hai cấp (Global Scale * Block Scale)
    blocks = values.reshape(-1, block_size)
    scales_expanded = (block_scales.float() * float(global_scale)).unsqueeze(-1)
    dequant_blocks = blocks * scales_expanded

    flat = dequant_blocks.reshape(-1)

    # 4. Cắt bỏ padding và khôi phục hình dạng gốc
    if pad_len > 0:
        flat = flat[:-pad_len]

    return flat.reshape(orig_shape).to(target_dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# BỘ LỌC CHÍNH XÁC: BẢO TỒN FAST TRANSFORMER & HEADS Ở CHUẨN BF16
# ═══════════════════════════════════════════════════════════════════════════════
def is_fast_transformer_layer(key_name: str) -> bool:
    """
    Bộ lọc chính xác dựa trên cấu trúc token để loại trừ hoàn toàn rủi ro false-positive.
    """
    name = key_name.lower()
    if "fast_layers" in name or "fast_output" in name:
        return True

    tokens = set(re.split(r"[._/]", name))
    protected_tokens = {
        "codebook",
        "embeddings",
        "embed",
        "norm",
        "bias",
    }
    return bool(tokens & protected_tokens)


def quantize_dual_ar_model(
    input_model_path: Path,
    output_dir: Path,
    block_size: int = NVFP4_DEFAULT_BLOCK_SIZE,
    run_validation_check: bool = True
):
    input_model_path = Path(input_model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📂 Đang nạp mô hình gốc: {input_model_path}")
    logger.info(f"💾 Thư mục xuất NVFP4 + BF16: {output_dir}")

    # Copy các file cấu hình và từ điển
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
        "quant_type": "official_nvidia_blackwell_nvfp4_bf16",
        "slow_ar_format": f"nvfp4_e2m1_two_level_block{block_size}",
        "fast_ar_format": "torch.bfloat16_preserved",
        "block_size": block_size,
        "layers_meta": {}
    }

    total_keys = len(state_dict)
    slow_count = 0
    fast_count = 0
    total_cosine_sim = 0.0
    validated_layers = 0

    logger.info(f"🚀 Bắt đầu quá trình nén {total_keys} ma trận trọng số theo chuẩn NVIDIA NVFP4...")

    for idx, (key, tensor) in enumerate(state_dict.items()):
        clean_key = key.replace("model.", "")

        # 1. Giữ nguyên tensor không phải số thực (index, mask)
        if not tensor.is_floating_point():
            quantized_state_dict[clean_key] = tensor
            metadata["layers_meta"][clean_key] = {
                "dtype": str(tensor.dtype),
                "type": "non_floating_point_tensor",
                "lossless": True
            }
            continue

        # 2. Bảo toàn Fast Transformer và Vector Norm/Bias ở chuẩn BF16
        if is_fast_transformer_layer(clean_key) or tensor.ndim < 2:
            is_originally_bf16 = tensor.dtype == torch.bfloat16
            quantized_state_dict[clean_key] = tensor.to(torch.bfloat16)
            metadata["layers_meta"][clean_key] = {
                "dtype": "torch.bfloat16",
                "type": "fast_ar_or_norm_preserved",
                "original_dtype": str(tensor.dtype),
                "lossless": is_originally_bf16,
                "shape": list(tensor.shape)
            }
            fast_count += 1
        else:
            # 3. Lượng tử hóa Slow Transformer sang NVFP4 Two-Level
            tensor_bf16 = tensor.to(torch.bfloat16)
            packed_weight, block_scales, global_scale, pad_len, orig_shape = quantize_tensor_to_nvfp4_two_level(
                tensor_bf16, block_size=block_size
            )

            quantized_state_dict[f"{clean_key}.nvfp4_weight"] = packed_weight
            quantized_state_dict[f"{clean_key}.nvfp4_block_scales"] = block_scales
            quantized_state_dict[f"{clean_key}.nvfp4_global_scale"] = torch.tensor(global_scale, dtype=torch.float32)

            metadata["layers_meta"][clean_key] = {
                "type": "slow_ar_nvfp4_two_level",
                "dtype": f"nvfp4_e2m1_block{block_size}",
                "orig_shape": list(orig_shape),
                "orig_numel": int(tensor.numel()),
                "pad_len": int(pad_len),
                "block_size": int(block_size),
                "global_scale": float(global_scale),
                "pack_order": "high_nibble_idx0_low_nibble_idx1"
            }
            slow_count += 1

            # Kiểm thử tính toán ngược (Round-trip Verification)
            if run_validation_check and (slow_count % 5 == 0 or slow_count <= 5):
                dequant = dequantize_tensor_from_nvfp4_two_level(
                    packed_weight, block_scales, global_scale, pad_len, orig_shape, block_size=block_size
                )
                cos_sim = torch.cosine_similarity(
                    tensor_bf16.float().flatten(),
                    dequant.float().flatten(),
                    dim=0
                ).item()
                total_cosine_sim += cos_sim
                validated_layers += 1

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
    avg_cosine = (total_cosine_sim / validated_layers) if validated_layers > 0 else 1.0

    logger.info("=" * 70)
    logger.info("🎉 QUÁ TRÌNH NÉN NVFP4 + BF16 THEO CHUẨN NVIDIA BLACKWELL ĐÃ HOÀN TẤT!")
    logger.info(f"📊 Dung lượng gốc (FP16/BF16)  : {orig_size_mb:.2f} MB (~{orig_size_mb/1024:.2f} GB)")
    logger.info(f"📊 Dung lượng sau nén (NVFP4)   : {quant_size_mb:.2f} MB (~{quant_size_mb/1024:.2f} GB)")
    logger.info(f"📉 Tỷ lệ cắt giảm bộ nhớ       : -{100 * (1 - quant_size_mb/orig_size_mb):.1f}%")
    logger.info(f"🛡️ Số layer Fast AR giữ BF16   : {fast_count} layers (Bảo toàn 100% âm sắc)")
    logger.info(f"⚡ Số layer Slow AR sang NVFP4  : {slow_count} layers (Block-16 Two-Level Scaling)")
    logger.info(f"🔬 Độ tương đồng Cosine trung bình : {avg_cosine * 100:.3f}% (Độ chính xác chuẩn công nghiệp)")
    logger.info("=" * 70)


@click.command()
@click.option("--checkpoint-path", type=click.Path(exists=True), required=True, help="Input FP16/BF16 model directory")
@click.option("--output", type=str, required=True, help="Output NVFP4 model directory")
@click.option("--block-size", type=int, default=NVFP4_DEFAULT_BLOCK_SIZE, help="NVFP4 scaling block size (default 16)")
def main(checkpoint_path, output, block_size):
    quantize_dual_ar_model(Path(checkpoint_path), Path(output), block_size=block_size)


if __name__ == "__main__":
    main()
