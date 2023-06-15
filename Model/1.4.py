from typing import Any, Literal
from pathlib import Path
import torch
import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch.nn import functional as F
import re

from torch.nn.functional import interpolate

def save_state_dict(state: dict[str, Any], path: str, format: Literal["ckpt", "safetensors"]) -> None:
    if format == "ckpt":
        torch.save(state, Path(path).open('wb'))
    elif format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ModuleNotFoundError('In order to use safetensors, run "pip install safetensors"')
        state = {k: v.contiguous().to_dense() for k, v in state.items()}
        save_file(state, path)

def load_model(path: Path, device: str) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path, device=device)
    elif path.suffix == ".pth":
        pth = torch.load(path, map_location=device)
        return pth["model"]
    else:
        ckpt = torch.load(path, map_location=device)
        return ckpt.get("state_dict", ckpt)

def tensor_size_mb(tensor: torch.Tensor) -> float:
    return tensor.nelement() * tensor.element_size() / (1024 * 1024)

def main(input: str, inputB: str):
    input = Path(input)
    inputB = Path(inputB)

    input_model = load_model(input, "cuda")
    inputB_model = load_model(inputB, "cuda")

    # Create directory for tensor if it doesn't exist
    save_dir = f"{input.stem} - {inputB.stem}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for layer_name, tensor in input_model.items():
        if "cond_stage_model.transformer.text_model.encoder." in layer_name or layer_name in inputB_model:
            b_tensor = None
            # 处理TE    
            if "cond_stage_model.transformer.text_model.encoder." in layer_name:
                split_parts = layer_name.split("cond_stage_model.transformer.text_model.encoder.layers.")

                # 提取索引值
                index = int(split_parts[1].split(".")[0])
                if "self_attn.k_proj.bias" in layer_name:
                    k_proj_bias = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.k_proj.bias"]
                    v_proj_bias = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.v_proj.bias"]
                    q_proj_bias = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.q_proj.bias"]

                    del inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.k_proj.bias"]
                    del inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.v_proj.bias"]
                    del inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.q_proj.bias"]
                    print(index)
                    #inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.in_proj.bias"] = torch.cat([k_proj_bias, v_proj_bias, q_proj_bias], dim=0)  # Shape: [3072]

                if "self_attn.k_proj.weight" in layer_name:
                    k_proj_weight = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.k_proj.weight"]
                    v_proj_weight = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.v_proj.weight"]
                    q_proj_weight = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.q_proj.weight"]
                    
                    del inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.k_proj.weight"]
                    del inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.v_proj.weight"]
                    del inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.q_proj.weight"]
                    print(index)
                    #inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.in_proj.weight"] = torch.cat([k_proj_weight, v_proj_weight, q_proj_weight], dim=0)  # Shape: [3072, 1024]
                    
                    
        else:
            # 处理B模型缺少A模型层的情况
            continue


    save_state_dict(inputB_model, "ac.ckpt", "ckpt")  # Save filtered state_dict

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="2.1模型")
parser.add_argument("inputB", type=str, help="1.x模型")
args = parser.parse_args()

if __name__ == "__main__":

    if not args.input or not args.inputB:
        parser.print_help()
        exit()

    main(args.input, args.inputB)

