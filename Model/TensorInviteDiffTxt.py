from pathlib import Path
import torch
import argparse
import numpy as np
import os

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

    # Create directory for saving results if it doesn't exist
    save_dir = f"{input.stem} - {inputB.stem}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    diff_variances = {}
    diff_norms = {}
    tensor_shapes = {}
    total_variance = 0.0
    total_norm_diff = 0.0

    for layer_name, tensor in input_model.items():
        if layer_name in inputB_model:
            # Ensure tensors are float for computation
            tensor = tensor.float()
            tensorB = inputB_model[layer_name].float()

            # Calculate the variance and norm difference between tensors
            diff_tensor = tensor - tensorB
            norm_diff = torch.norm(diff_tensor).item()  # L2 norm difference 范数差异
            variance = torch.var(diff_tensor).item()  # Compute variance 方差差异

            # Accumulate total variance and total norm difference
            total_variance += variance
            total_norm_diff += norm_diff

            # Store variance, norm difference and tensor shape
            diff_variances[layer_name] = variance
            diff_norms[layer_name] = norm_diff
            tensor_shapes[layer_name] = list(tensor.size())
        else:
            # Handle missing layers in model B
            continue

    # Save the diff_variances and norm differences to a txt file
    with open(f"{save_dir}/Diff_Variances.txt", "w") as f:
        f.write('{:<30}{:<30}{:<30}{:<50}{:<30}\n'.format("类似曲线陡峭程度", "类似向量长度变化", "Shape", "Layer Name", ""))
        for layer_name in diff_variances.keys():
            variance = diff_variances[layer_name]
            norm_diff = diff_norms[layer_name]
            tensor_shape = tensor_shapes[layer_name]
            f.write('{:<30}{:<30}{:<30}{:<50}\n'.format(
                f"Variance: {variance:.6f}",
                f"Norm Diff: {norm_diff:.6f}",
                f"Shape: {tensor_shape}",
                f"{layer_name}"
            ))
        f.write(f"\nTotal variance: {total_variance:.6f}\n")
        f.write(f"Total norm difference: {total_norm_diff:.6f}\n")

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("inputB", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
args = parser.parse_args()

if __name__ == "__main__":
    if not args.input or not args.inputB:
        parser.print_help()
        exit()

    main(args.input, args.inputB)
