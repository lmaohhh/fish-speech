import gc
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import click
import hydra
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger

from fish_speech.models.text2semantic.llama import BaseTransformer
from fish_speech.models.text2semantic.lora import get_merged_state_dict


from safetensors.torch import save_file


@click.command()
@click.option("--lora-config", type=str, default="r_32_alpha_64")
@click.option("--base-weight", type=str, default="checkpoints/s2-pro")
@click.option("--lora-weight", type=str, required=True)
@click.option("--output", type=str, required=True)
@click.option("--quantize-fp8/--no-quantize-fp8", default=True, help="Directly quantize Slow Transformer to FP8 (output size ~4.0 GB)")
def merge(lora_config, base_weight, lora_weight, output, quantize_fp8):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Merging {base_weight} and {lora_weight} into {output} (quantize_fp8={quantize_fp8})"
    )

    with initialize(version_base="1.3", config_path="../../fish_speech/configs/lora"):
        cfg = compose(config_name=lora_config)

    lora_config_obj = instantiate(cfg)
    logger.info(f"Loaded lora config {lora_config_obj}")

    llama_model = BaseTransformer.from_pretrained(
        path=base_weight,
        load_weights=True,
        lora_config=lora_config_obj,
    )
    logger.info(f"Loaded llama model")

    llama_state_dict = llama_model.state_dict()
    llama_state_dict = {k: v for k, v in llama_state_dict.items() if "lora" not in k}
    lora_state_dict = torch.load(lora_weight, map_location="cpu", weights_only=False)

    if "state_dict" in llama_state_dict:
        llama_state_dict = llama_state_dict["state_dict"]

    if "state_dict" in lora_state_dict:
        lora_state_dict = lora_state_dict["state_dict"]

    # remove prefix model.
    if any(k.startswith("model.") for k in llama_state_dict.keys()):
        llama_state_dict = {
            k.replace("model.", ""): v
            for k, v in llama_state_dict.items()
            if k.startswith("model.")
        }
    if any(k.startswith("model.") for k in lora_state_dict.keys()):
        lora_state_dict = {
            k.replace("model.", ""): v
            for k, v in lora_state_dict.items()
            if k.startswith("model.")
        }

    logger.info(f"Found {len(llama_state_dict)} keys in base model")
    logger.info(f"Found {len(lora_state_dict)} keys in lora weights")

    merged_state_dict = llama_state_dict | lora_state_dict
    llama_model.load_state_dict(merged_state_dict, strict=True)
    logger.info(f"Merged model loaded into memory")

    # Trigger eval mode to apply LoRA merge to base weights
    llama_model.eval()
    state_to_save = llama_model.state_dict()

    # Drop any remaining lora keys
    final_dict = {k: v for k, v in state_to_save.items() if "lora" not in k}

    # Selective FP8 Quantization in RAM if requested (Saves Slow AR to FP8, keeps Fast AR in BF16)
    if quantize_fp8:
        logger.info("⚡ In-memory FP8 Quantization: Converting Slow Transformer to FP8...")
        from tools.llama.quantize_fp8 import quantize_tensor_to_fp8_e4m3

        processed_dict = {}
        for k, v in final_dict.items():
            is_linear_weight = (
                v.ndim == 2
                and any(p in k for p in ["wqkv", "wo", "w1", "w2", "w3"])
                and not any(s in k for s in ["fast_layers", "fast_project", "fast_output", "fast_embeddings", "norm", "embed", "bias"])
            )
            if is_linear_weight:
                w_fp8, scale = quantize_tensor_to_fp8_e4m3(v)
                processed_dict[k] = w_fp8
                processed_dict[f"{k}_scale"] = scale
            else:
                processed_dict[k] = v.to(torch.bfloat16) if v.dtype == torch.float32 else v
        final_dict = processed_dict
        logger.info("✅ In-memory FP8 Quantization complete!")

    # Shard and save as .safetensors (2 shards of ~2.0 GB each)
    logger.info(f"💾 Saving clean .safetensors to {output}...")
    keys = list(final_dict.keys())
    split_idx = len(keys) // 2

    shard1 = {k: final_dict[k] for k in keys[:split_idx]}
    shard2 = {k: final_dict[k] for k in keys[split_idx:]}

    shard1_name = "model-00001-of-00002.safetensors"
    shard2_name = "model-00002-of-00002.safetensors"

    save_file(shard1, str(output / shard1_name))
    save_file(shard2, str(output / shard2_name))

    # Write weight map index JSON
    weight_map = {}
    for k in shard1.keys():
        weight_map[k] = shard1_name
    for k in shard2.keys():
        weight_map[k] = shard2_name

    index_data = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in final_dict.values())},
        "weight_map": weight_map,
    }
    import json
    with open(output / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=2)

    # Copy config and tokenizer files
    for file in Path(base_weight).glob("*"):
        if file.is_file() and file.suffix in [".json", ".tiktoken", ".txt", ".bin"]:
            if "safetensors" in file.name:
                continue
            dest = output / file.name
            if not dest.exists():
                shutil.copyfile(file, dest)

    logger.info(f"🎉 100% HOÀN TẤT! Mô hình đã được ghi an toàn tại {output}!")
    del llama_model, llama_state_dict, lora_state_dict, merged_state_dict, final_dict
    gc.collect()


if __name__ == "__main__":
    merge()
