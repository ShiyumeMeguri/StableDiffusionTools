from pathlib import Path
import torch
import argparse
import os
from tqdm import tqdm
from collections import defaultdict

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

def main(input: str, inputB: str):
    input = Path(input)
    inputB = Path(inputB)

    input_model = load_model(input, "cpu")
    inputB_model = load_model(inputB, "cpu")

    # Create directory for saving results if it doesn't exist
    save_dir = f"{input.stem} - {inputB.stem}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Dictionaries to store results based on prefixes
    variance_files = defaultdict(list)
    norm_files = defaultdict(list)

    total_variance = 0.0
    total_norm_diff = 0.0

    # One loop for calculating differences and sorting into files by prefix
    for layer_name, tensor in tqdm(input_model.items(), desc="Comparing Layers", total=len(input_model)):
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

            # Extract prefix from layer name
            prefix = layer_name.split('.')[0]

            # Store variance information
            variance_line = '{:<30}{:<30}{:<50}\n'.format(
                f"Variance: {variance:.6f}",
                f"Shape: {list(tensor.size())}",
                f"{layer_name}"
            )
            variance_files[f"Diff_Variances.{prefix}.txt"].append((variance, variance_line))

            # Store norm information
            norm_line = '{:<30}{:<30}{:<50}\n'.format(
                f"Norm Diff: {norm_diff:.6f}",
                f"Shape: {list(tensor.size())}",
                f"{layer_name}"
            )
            norm_files[f"Diff_Norms.{prefix}.txt"].append((norm_diff, norm_line))

        else:
            # Handle missing layers in model B (could log or ignore)
            continue

    # Sort and write variance differences to corresponding files
    for filename, lines in variance_files.items():
        # Sort by variance in descending order
        lines.sort(key=lambda x: x[0], reverse=True)
        with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
            f.write('{:<30}{:<30}{:<50}\n'.format("类似曲线陡峭程度", "网络形状","Layer Name"))
            for _, line in lines:
                f.write(line)
            f.write(f"\nTotal variance: {total_variance:.6f}\n")

    # Sort and write norm differences to corresponding files
    for filename, lines in norm_files.items():
        # Sort by norm difference in descending order
        lines.sort(key=lambda x: x[0], reverse=True)
        with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
            f.write('{:<30}{:<30}{:<50}\n'.format("类似向量长度变化", "网络形状", "Layer Name"))
            for _, line in lines:
                f.write(line)
            f.write(f"\nTotal norm difference: {total_norm_diff:.6f}\n")

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("inputB", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
args = parser.parse_args()

if __name__ == "__main__":
    if not args.input or not args.inputB:
        parser.print_help()
        exit()

    main(args.input, args.inputB)
