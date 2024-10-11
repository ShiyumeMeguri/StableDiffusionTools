from typing import Any, Literal
from pathlib import Path
import re
import math
import torch
import argparse
import tensorflow as tf
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

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

def calculate_weights(weight_A: torch.Tensor, weight_B: torch.Tensor, ratio: float, mode: str) -> torch.Tensor:
    if mode == "replace":
        return weight_B * ratio
    return weight_A * (1 - ratio) + weight_B * ratio

# 处理层的方法，支持不同的计算方式
def process_layers(
    state_dict_A: dict[str, torch.Tensor], 
    state_dict_B: dict[str, torch.Tensor], 
    config_dict: dict[str, float], 
    mode: str = "linear_combination"  # 默认的计算模式是线性组合
) -> dict[str, torch.Tensor]:
    merged_state_dict = {}

    for layer_name, weight_A in state_dict_A.items():
        if layer_name in config_dict:
            ratio = config_dict[layer_name]
            if state_dict_B and layer_name in state_dict_B:
                weight_B = state_dict_B[layer_name]
                merged_state_dict[layer_name] = calculate_weights(weight_A, weight_B, ratio, mode)
            else:
                if ratio != 0.0:
                    merged_state_dict[layer_name] = weight_A * ratio
                else:
                    merged_state_dict[layer_name] = weight_A

    return merged_state_dict

def save_default_config(model: dict[str, torch.Tensor], config_path: Path):
    with open(config_path, 'w') as f:
        for name in model.keys():
            f.write(f"{name}::1.0\n")  # 默认权重

def load_config(config_path: str) -> dict[str, float]:
    config_dict = {}
    if config_path:
        with open(config_path, "r") as config_file:
            for line in config_file:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if line.startswith("-"):
                    layer_name, ratio = line.split("::")
                    print(f"已删除 {layer_name} 层")
                    continue
                if line:
                    layer_name, ratio = line.split("::")
                    ratio = float(ratio)
                    config_dict[layer_name] = ratio
    return config_dict

def main():
    input_path = Path(args.input)
    config_path = args.config
    output_path = args.output
    mode = args.mode  # 新增参数 mode

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = None

    # 如果没有提供输出路径，则使用input的路径加"merged"后缀，并默认保存为.ckpt格式
    if not output_path:
        if model_path:
            output_path = input_path.with_name(f"{input_path.stem}+{model_path.stem}_merged.ckpt")
        else:
            output_path = input_path.with_name(f"{input_path.stem}_merged.ckpt")
    elif not output_path.endswith(".safetensors") and not output_path.endswith(".ckpt"):
        output_path += ".ckpt"

    output_path = Path(output_path)

    state_dict_A = load_model(input_path, "cpu")
    state_dict_B = None
    if model_path:
        state_dict_B = load_model(model_path, "cpu")

    if not config_path:
        # 没有提供config时，生成默认config文件
        config_path = input_path.with_name(input_path.stem + "_config.txt")
        print(f"缺少配置 已根据输入模型生成了权重配置文件 {config_path}")
        save_default_config(state_dict_A, config_path)
        return

    config_dict = load_config(config_path)
    
    # 在调用 process_layers 时传递 mode 参数
    merged_state_dict = process_layers(state_dict_A, state_dict_B, config_dict, mode)

    format = output_path.suffix[1:]  # Remove the leading dot
    save_state_dict(merged_state_dict, output_path, format)  # Save merged state_dict
    print(f"Saved to {output_path.absolute()}")

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("config", type=str, nargs='?', help="Path to configuration file. If not provided, model layers will be printed.")
parser.add_argument("--output", "-o", type=str, help="Path to output file. If not provided, defaults to input+merged.ckpt.")
parser.add_argument("--model", type=str, help="Path to model file. Must be a .safetensors or .ckpt file.")
parser.add_argument("--mode", type=str, default="linear_combination", help="Mode of weight calculation: 'linear_combination', 'replace', etc.")  # 新增的 mode 参数
args = parser.parse_args()

if __name__ == "__main__":
    if not args.input:
        parser.print_help()
        exit()

    main()