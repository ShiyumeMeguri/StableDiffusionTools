from typing import Any, Literal
from pathlib import Path
import torch
import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

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

def main(inputBase: str, inputFine: str, save_dir: str, model_type: str):
    inputBase = Path(inputBase)
    inputFine = Path(inputFine)

    inputBase_model = load_model(inputBase, "cpu")
    inputFine_model = load_model(inputFine, "cpu")

    save_suffix = inputFine.suffix[1:]
    if save_dir == "":
        save_dir = f"./{inputFine.stem}_Restone{model_type}.{save_suffix}"

    # 首先，读取TensorRestoreConfig.txt文件，获取需要恢复的层名称
    restore_layers = set()
    with open(f'TensorRestoneConfig{model_type}.txt', 'r') as file:
        for line in file:
            # 去除每行的空白符（如换行符）并添加到集合中
            restore_layers.add(line.strip())

    restone_model = {}
    for layer_name, tensor in inputBase_model.items():
        if layer_name in restore_layers:
            # 如果层名称在TensorRestoreConfig.txt中，则从inputBase_model恢复权重
            restone_model[layer_name] = tensor
        else:
            if layer_name in inputFine_model:
                restone_model[layer_name] = inputFine_model[layer_name]
            else:
                # 可以根据需要处理不存在的层，例如跳过或记录错误
                print(f"Layer {layer_name} not found in inputFine_model, skipping.")
                continue
    
    save_state_dict(restone_model, save_dir, save_suffix)


parser = argparse.ArgumentParser()
parser.add_argument("inputBase", type=str, help="基础模型 .safetensors or .ckpt file.")
parser.add_argument("inputFine", type=str, help="微调模型 .safetensors or .ckpt file.")
parser.add_argument("--output", type=str,default="", help=" .safetensors or .ckpt file.")
parser.add_argument("--type", type=str, default="SD1.5", help="模型类型")
args = parser.parse_args()

if __name__ == "__main__":

    if not args.inputBase or not args.inputFine:
        parser.print_help()
        exit()

    main(args.inputBase, args.inputFine, args.output, args.type)

