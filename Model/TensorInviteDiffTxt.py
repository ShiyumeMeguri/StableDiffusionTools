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

    diff_ratios = {}
    diff_sizes = {}
    tensor_shapes = {}
    total_diff_size = 0.0
    for layer_name, tensor in input_model.items():
        if layer_name in inputB_model:
            # Calculating the proportion of different weights
            tensor = tensor.float()
            tensorB = inputB_model[layer_name].float()
            is_close = torch.isclose(tensor, tensorB, rtol=1e-05, atol=1e-08)
            num_different = torch.numel(tensor) - is_close.sum().item()
            diff_ratio = num_different / torch.numel(tensor)
            diff_size = diff_ratio * tensor_size_mb(tensor)
            total_diff_size += diff_size
            diff_ratios[layer_name] = diff_ratio
            diff_sizes[layer_name] = diff_size
            tensor_shapes[layer_name] = list(tensor.size())
        else:
            # 处理B模型缺少A模型层的情况
            continue

    # Save the diff_ratios and diff_sizes to a txt file
    with open(f"{save_dir}/Diff_Ratios.txt", "w") as f:
        for layer_name, diff_ratio in diff_ratios.items():
            diff_size = diff_sizes[layer_name]
            tensor_shape = tensor_shapes[layer_name]
            f.write('{:<10}{:<30}{:<30}{:<50}\n'.format(
                f"{diff_ratio * 100:.3f}%".zfill(7),
                f"Diff Size: {diff_size:.3f} MB",
                f"Shape: {tensor_shape}",
                f"{layer_name}::1.0"
            ))
        f.write(f"\nTotal difference tensor size: {total_diff_size:.3f} MB")


parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("inputB", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
args = parser.parse_args()

if __name__ == "__main__":

    if not args.input or not args.inputB:
        parser.print_help()
        exit()

    main(args.input, args.inputB)
