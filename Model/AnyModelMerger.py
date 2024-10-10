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

def save_default_config(model: dict[str, torch.Tensor], config_path: Path):
    with open(config_path, 'w') as f:
        for name in model.keys():
            f.write(f"{name}::0.5\n")  # 默认权重

def process_layers(state_dict: dict[str, torch.Tensor], state_dict_B: dict[str, torch.Tensor], config_dict: dict[str, float]) -> dict[str, torch.Tensor]:
    merged_state_dict = {}
    for layer_name, weight in state_dict.items():
        if layer_name in config_dict:
            ratio = config_dict[layer_name]
            # 检测是否删除这一层
            if layer_name.startswith("-"):
                print(f"已删除 {layer_name} 层.")
                continue

            # 如果state_dict_B存在，则合并
            if state_dict_B:
                merged_state_dict[layer_name] = (weight * (1 - ratio) + state_dict_B[layer_name] * ratio)
            else:
                merged_state_dict[layer_name] = weight

    return merged_state_dict

def load_config(config_path: str) -> dict[str, float]:
    config_dict = {}
    if config_path:
        with open(config_path, "r") as config_file:
            for line in config_file:
                line = line.strip()
                if line and not line.startswith("#"):
                    layer_name, ratio = line.split("::")
                    ratio = float(ratio)
                    config_dict[layer_name] = ratio
    return config_dict

def main():
    input_path = Path(args.input)
    config_path = args.config
    output_path = args.output

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = None

    # 如果没有提供输出路径，则使用input的路径加"merged"后缀，并默认保存为.ckpt格式
    if not output_path:
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
    merged_state_dict = process_layers(state_dict_A, state_dict_B, config_dict)

    format = output_path.suffix[1:]  # Remove the leading dot
    save_state_dict(merged_state_dict, output_path, format)  # Save merged state_dict
    print(f"Saved to {output_path.absolute()}")

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("config", type=str, nargs='?', help="Path to configuration file. If not provided, model layers will be printed.")
parser.add_argument("output", type=str, nargs='?', help="Path to output file. If not provided, defaults to input+merged.ckpt.")
parser.add_argument("--model", type=str, help="Path to model file. Must be a .safetensors or .ckpt file.")
args = parser.parse_args()

if __name__ == "__main__":
    if not args.input:
        parser.print_help()
        exit()

    main()