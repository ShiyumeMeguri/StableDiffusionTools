from typing import Any, Literal
from pathlib import Path
import torch
import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

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

    input_model = load_model(input, "cpu")
    inputB_model = load_model(inputB, "cpu")

    # Create directory for tensor if it doesn't exist
    save_dir = f"{input.stem} - {inputB.stem}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for layer_name, tensor in input_model.items():
        if layer_name in inputB_model:
            b_tensor = inputB_model[layer_name]

            # 先比较AB模型的形状是否不同
            if tensor.shape != b_tensor.shape:
                # 对于第一个权重
                if b_tensor.ndim == 4:
                    # 使用squeeze函数移除大小为1的维度
                    a_tensor = b_tensor.squeeze(-1).squeeze(-1)

                # 对于第二个权重
                elif b_tensor.ndim == 2:
                    # 使用插值函数调整尺寸
                    a_tensor = torch.nn.functional.interpolate(b_tensor.float().unsqueeze(0), size=[1024], mode='linear', align_corners=False).squeeze(0).half()

                else:
                    a_tensor = b_tensor  # 如果不是以上两种情况，不修改tensor

                print(f"{layer_name} \t\t {a_tensor.shape} \t\t {tensor.shape}")
                input_model[layer_name] = a_tensor
                continue
                # 用新的tensor更新模型
            input_model[layer_name] = b_tensor
        else:
            # 处理B模型缺少A模型层的情况
            continue


    save_state_dict(input_model, "ac.ckpt", "ckpt")  # Save filtered state_dict

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("inputB", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
args = parser.parse_args()

if __name__ == "__main__":

    if not args.input or not args.inputB:
        parser.print_help()
        exit()

    main(args.input, args.inputB)

