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
from tqdm import tqdm
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
        
def calculate_weights(weight_A: torch.Tensor, weight_B: torch.Tensor, ratio: float, mode: str, **kwargs) -> torch.Tensor:
    if mode == "replace":
        return weight_B * ratio
    elif mode == "linear_combination":
        return weight_A * (1 - ratio) + weight_B * ratio
    elif mode == "svd":
        dim = kwargs.get('dim', 20480)
        clamp_quantile = kwargs.get('clamp_quantile', 0.99)
        min_diff = kwargs.get('min_diff', 0.01)
        device = kwargs.get('device', "cpu")

        weight_diff = weight_B - weight_A 

        # Only perform SVD if the maximum absolute difference exceeds the threshold
        if torch.max(torch.abs(weight_diff)) > min_diff:
            if device:
                weight_diff = weight_diff.to(device)
            weight_diff = weight_diff.to(torch.float32)

            original_shape = weight_diff.shape  # Record the original shape

            # Reshape the tensor to 2D if necessary
            if weight_diff.dim() < 2:
                mat = weight_diff.view(-1, 1)
            else:
                # Determine if it's a Conv2d layer
                is_conv2d = len(weight_diff.size()) == 4
                if is_conv2d:
                    out_channels, in_channels, kH, kW = weight_diff.size()
                    mat = weight_diff.view(out_channels, -1)
                else:
                    mat = weight_diff

            # Compute SVD
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            rank = min(dim, U.size(1), Vh.size(0))
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]

            U_S = U @ torch.diag(S)

            # Clamp U_S and Vh
            #dist = torch.cat([U_S.flatten(), Vh.flatten()])
            #hi_val = torch.quantile(dist, clamp_quantile)
            #low_val = -hi_val
            #
            #U_S = U_S.clamp(low_val, hi_val)
            #Vh = Vh.clamp(low_val, hi_val)

            # Reconstruct the approximated weight difference
            mat_approx = U_S @ Vh

            # Reshape back to the original shape
            if weight_diff.dim() < 2:
                weight_diff_svd = mat_approx.view(original_shape)
            elif is_conv2d:
                weight_diff_svd = mat_approx.view(out_channels, in_channels, kH, kW)
            else:
                weight_diff_svd = mat_approx.view(original_shape)

            # Merge the weights
            merged_weight = weight_A + weight_diff_svd * ratio
            return merged_weight
    return weight_A

# 处理层的方法，支持不同的计算方式
def process_layers(
    state_dict_A: dict[str, torch.Tensor], 
    state_dict_B: dict[str, torch.Tensor], 
    config_dict: dict[str, float], 
    mode: str = "linear_combination"  # 默认的计算模式是线性组合
) -> dict[str, torch.Tensor]:
    merged_state_dict = {}

    for layer_name, weight_A in tqdm(state_dict_A.items(), desc="Processing layers"):
        if layer_name in config_dict:
            ratio = config_dict[layer_name]
            layer_name_without_model = layer_name
            if state_dict_B and layer_name_without_model in state_dict_B:
                weight_B = state_dict_B[layer_name_without_model]
                merged_state_dict[layer_name] = calculate_weights(weight_A, weight_B, ratio, mode)
            else:
                if ratio != 0.0:
                    merged_state_dict[layer_name] = weight_A * ratio
                else:
                    merged_state_dict[layer_name] = weight_A
        else:
            print(f"层丢失警告: 输入模型的 {layer_name} 层找不到 配置文件可能被错误修改")

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
parser.add_argument("--mode", type=str, default="linear_combination", help="Mode of weight calculation: 'linear_combination', 'replace', 'svd', etc.")  # 新增的 mode 参数
args = parser.parse_args()

if __name__ == "__main__":
    if not args.input:
        parser.print_help()
        exit()

    main()