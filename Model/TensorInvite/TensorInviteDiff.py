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
    for layer_name, tensor in input_model.items():
        if layer_name in inputB_model:
            # Calculating ratio
            tensor = tensor.float()
            tensorB = inputB_model[layer_name].float()
            diff = torch.abs(tensor - tensorB)
            mean_diff = torch.mean(diff)
            meanA = torch.mean(torch.abs(tensor))
            diff_ratio = mean_diff / meanA if meanA != 0 else 0
            diff_ratios[layer_name] = diff_ratio
        else:
            # 处理B模型缺少A模型层的情况
            continue

    # Save the diff_ratios to a txt file
    with open(f"{save_dir}/Diff_Ratios.txt", "w") as f:
        for layer_name, diff_ratio in diff_ratios.items():
            f.write(f"{diff_ratio:.3%}".zfill(7) + "\t\t:\t\t" + f"{layer_name}::1.0\n")

            
    def visualize_tensor1d(tensor_A, tensor_B, tensor_name):
        # Compute difference between tensor_A and tensor_B
        tensor = torch.abs(tensor_A - tensor_B)
        
        # Skip if all values are zero
        if torch.all(tensor == 0):
            return
        
        # Calculating ratio
        mean_diff = torch.mean(tensor)
        meanA = torch.mean(torch.abs(tensor_A))
        diff_ratio = mean_diff / meanA if meanA != 0 else 0

        # Create directory for tensor if it doesn't exist
        save_dir = f"{input.stem} - {inputB.stem}/{tensor_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Handle tensor_A
        if len(tensor_A.shape) == 1:  # for 1D tensor
            visualize_1d_tensor(tensor_A.view(-1), tensor_B.view(-1), save_dir, tensor_name, tensor_A.shape, diff_ratio)
        elif len(tensor_A.shape) == 2:  # for 2D tensor
            visualize_1d_tensor(tensor_A[:, :].view(-1), tensor_B[:, :].view(-1), save_dir, tensor_name, tensor_A.shape, diff_ratio)
        elif len(tensor_A.shape) == 3:  # for 3D tensor
            c = tensor_A.shape[-1]  # assume the last dimension is the channel
            for i in range(c):
                visualize_1d_tensor(tensor_A[:, :, i].view(-1), tensor_B[:, :, i].view(-1), save_dir, tensor_name, f"{tensor_A.shape}_{i}", diff_ratio)
        elif len(tensor_A.shape) == 4:  # for 4D tensor
            h, w, c1, c2 = tensor_A.shape  # assume the last two dimensions are the channels
            for i in range(c1):
                for j in range(c2):
                    visualize_1d_tensor(tensor_A[:, :, i, j].view(-1), tensor_B[:, :, i, j].view(-1), save_dir, tensor_name, f"{tensor_A.shape}_{h}x{w}_{i}_{j}", diff_ratio)

    def visualize_1d_tensor(tensor_1d_A, tensor_1d_B, save_dir, tensor_name, idx, diff_ratio):
        # Creating x-axis data points
        x = np.arange(0, len(tensor_1d_A), 1)

        # Plotting the tensor
        plt.figure(figsize=(40.96, 10.24))  # set the figure size to 1920x1080
        plt.plot(x, tensor_1d_A, linewidth=0.5, linestyle='-', label='Model A', color='blue', alpha=0.5)
        plt.plot(x, tensor_1d_B, linewidth=0.5, linestyle='--', label='Model B', color='red', alpha=0.5)

        # Adding plot title and axis labels
        plt.title(f"{tensor_name} {idx}")
        plt.xlabel("Weight Index")
        plt.ylabel("Value")

        # Adding legend
        plt.legend()

        # Adding ratio text
        plt.text(0.95, 0.95, f"Diff Ratio: {diff_ratio:.2f}", transform=plt.gca().transAxes, fontsize=12, ha='right', va='top')

        # Turn off interactive mode
        plt.ioff()
        # Saving the plot
        plt.savefig(f"{save_dir}/{tensor_name} {idx}.png")
        plt.close()

    
    for layer_name, tensor in input_model.items():
        if layer_name in inputB_model:
            tensorB = inputB_model[layer_name]
            visualize_tensor1d(tensor, tensorB, layer_name)
        else:
            # 处理B模型缺少A模型层的情况
            continue

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("inputB", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
args = parser.parse_args()

if __name__ == "__main__":

    if not args.input or not args.inputB:
        parser.print_help()
        exit()

    main(args.input, args.inputB)

