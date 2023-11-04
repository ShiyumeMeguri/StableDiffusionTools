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
        if "cond_stage_model.model." in layer_name or layer_name in inputB_model:
            b_tensor = None
            # 处理UNet
            if "model.diffusion_model." in layer_name:
                b_tensor = inputB_model[layer_name]
                # 先比较AB模型的形状是否不同
                if tensor.shape != b_tensor.shape:
                    if b_tensor.ndim == 4:
                        # 使用squeeze函数移除大小为1的维度
                        a_tensor = b_tensor.squeeze(-1).squeeze(-1)
                    elif b_tensor.ndim == 2:
                        a_tensor = F.pad(b_tensor, (0, 256))
                    else:
                        a_tensor = b_tensor  # 如果不是以上两种情况，不修改tensor

                    #print(f"{layer_name} 形状改变 \t {a_tensor.shape} \t {b_tensor.shape}")
                    input_model[layer_name] = a_tensor
                    continue
            # 处理TE    
            if "cond_stage_model.model." in layer_name:
                index = 22
                if "cond_stage_model.model.transformer.resblocks." in layer_name:
                    index = int(layer_name.split("resblocks.")[1].split(".")[0])
                if "attn.in_proj_bias" in layer_name:
                    if index > 11:
                        input_model[layer_name] = torch.zeros_like(input_model[layer_name])
                        continue
                    k = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.k_proj.bias"]
                    v = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.v_proj.bias"]
                    q = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.q_proj.bias"]

                    k_proj_bias = F.pad(k, (0, 256))
                    v_proj_bias = F.pad(v, (0, 256))
                    q_proj_bias = F.pad(q, (0, 256))
                    
                    b_tensor = torch.cat([k_proj_bias, v_proj_bias, q_proj_bias], dim=0)  # Shape: [3072]

                if "attn.in_proj_weight" in layer_name:
                    if index > 11:
                        input_model[layer_name] = torch.zeros_like(input_model[layer_name])
                        continue
                    k = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.k_proj.weight"]
                    v = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.v_proj.weight"]
                    q = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.q_proj.weight"]

                    k_proj_weight = F.pad(k, (0, 256, 0, 256))
                    v_proj_weight = F.pad(v, (0, 256, 0, 256))
                    q_proj_weight = F.pad(q, (0, 256, 0, 256))
                    b_tensor = torch.cat([k_proj_weight, v_proj_weight, q_proj_weight], dim=0)  # Shape: [3072, 1024]
                    
                if "attn.out_proj.bias" in layer_name:
                    if index > 11:
                        input_model[layer_name] = torch.zeros_like(input_model[layer_name])
                        continue
                    b = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.out_proj.bias"]
                    b_tensor = F.pad(b, (0, 256))

                if "attn.out_proj.weight" in layer_name:
                    if index > 11:
                        input_model[layer_name] = torch.zeros_like(input_model[layer_name])
                        continue
                    w = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.self_attn.out_proj.weight"]
                    b_tensor = F.pad(w, (0, 256, 0, 256))

                if "ln_1.bias" in layer_name or "ln_1.weight" in layer_name or "ln_2.bias" in layer_name or "ln_2.weight" in layer_name:
                    if index > 11:
                        input_model[layer_name] = torch.zeros_like(input_model[layer_name])
                        continue
                    if "ln_1.bias" in layer_name:
                        b = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.layer_norm1.bias"]
                        b_tensor = F.pad(b, (0, 256))
                    if "ln_1.weight" in layer_name:
                        w = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.layer_norm1.weight"]
                        b_tensor = F.pad(w, (0, 256))
                    if "ln_2.bias" in layer_name:
                        b = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.layer_norm2.bias"]
                        b_tensor = F.pad(b, (0, 256))
                    if "ln_2.weight" in layer_name:
                        w = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.layer_norm2.weight"]
                        b_tensor = F.pad(w, (0, 256))

                if "mlp.c_fc.bias" in layer_name:
                    if index > 11:
                        input_model[layer_name] = torch.zeros_like(input_model[layer_name])
                        continue
                    b = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.mlp.fc1.bias"]
                    b_tensor = F.pad(b, (0, 1024))

                if "mlp.c_fc.weight" in layer_name:
                    if index > 11:
                        input_model[layer_name] = torch.zeros_like(input_model[layer_name])
                        continue
                    w = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.mlp.fc1.weight"]
                    b_tensor = F.pad(w, (0, 256, 0, 1024))

                if "mlp.c_proj.bias" in layer_name:
                    if index > 11:
                        input_model[layer_name] = torch.zeros_like(input_model[layer_name])
                        continue
                    b = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.mlp.fc2.bias"]
                    b_tensor = F.pad(b, (0, 256))

                if "mlp.c_proj.weight" in layer_name:
                    if index > 11:
                        input_model[layer_name] = torch.zeros_like(input_model[layer_name])
                        continue
                    w = inputB_model[f"cond_stage_model.transformer.text_model.encoder.layers.{index}.mlp.fc2.weight"]
                    b_tensor = F.pad(w, (0, 1024, 0, 256))

                if "ln_final" in layer_name:
                    if "bias" in layer_name:
                        b = inputB_model[f"cond_stage_model.transformer.text_model.final_layer_norm.bias"]
                        b_tensor = F.pad(b, (0, 256))
                    if "weight" in layer_name:
                        w = inputB_model[f"cond_stage_model.transformer.text_model.final_layer_norm.weight"]
                        b_tensor = F.pad(w, (0, 256))

                if "positional_embedding" in layer_name:
                    w = inputB_model[f"cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"]
                    b_tensor = F.pad(w, (0, 256))

                if "token_embedding" in layer_name:
                    w = inputB_model[f"cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
                    b_tensor = F.pad(w, (0, 256))

                if "text_model.embeddings.position_ids" in layer_name:
                    b_tensor = inputB_model[f"cond_stage_model.transformer.text_model.embeddings.position_ids"]

            if b_tensor != None:
                # 用新的tensor更新模型
                print(f"{layer_name} \t {b_tensor.shape}")
                input_model[layer_name] = b_tensor
        else:
            # 处理B模型缺少A模型层的情况
            continue


    save_state_dict(input_model, "ac.ckpt", "ckpt")  # Save filtered state_dict

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="2.1模型")
parser.add_argument("inputB", type=str, help="1.x模型")
args = parser.parse_args()

if __name__ == "__main__":

    if not args.input or not args.inputB:
        parser.print_help()
        exit()

    main(args.input, args.inputB)

