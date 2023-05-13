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
    else:
        ckpt = torch.load(path, map_location=device)
        return ckpt.get("state_dict", ckpt)

def main(input: str, inputB: str):
    input = Path(input)
    inputB = Path(inputB)

    input_model = load_model(input, "cpu")
    inputB_model = load_model(inputB, "cpu")

    # 张量可视化2 转1维
    def visualize_tensor1d(tensor_A, tensor_B, tensor_name):
        # Compute difference between tensor_A and tensor_B
        tensor = tensor_A - tensor_B
        
        # Skip if all values are zero
        if torch.all(tensor == 0):
            return
            
        # Create directory for tensor if it doesn't exist
        save_dir = f"{input.stem} ^ {inputB.stem}/{tensor_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Calculate the y-axis range using the range of tensor_A values
        y_min = torch.min(tensor_A).item()
        y_max = torch.max(tensor_A).item()
        y_range = y_max - y_min
        
        if len(tensor.shape) == 1:  # for 1D tensor
            visualize_1d_tensor(tensor.view(-1), save_dir, tensor_name, idx=tensor.shape, y_range=y_range)
        elif len(tensor.shape) == 2:  # for 2D tensor
            visualize_1d_tensor(tensor[:, :].view(-1), save_dir, tensor_name, idx=tensor.shape, y_range=y_range)

        elif len(tensor.shape) == 3:  # for 3D tensor
            c = tensor.shape[-1]  # assume the last dimension is the channel
            for i in range(c):
                visualize_1d_tensor(tensor[:, :, i].view(-1), save_dir, tensor_name, idx=f"{tensor.shape}_{i}", y_range=y_range)

        elif len(tensor.shape) == 4:  # for 4D tensor
            h, w, c1, c2 = tensor.shape  # assume the last two dimensions are the channels
            for i in range(c1):
                for j in range(c2):
                    visualize_1d_tensor(tensor[:, :, i, j].view(-1), save_dir, tensor_name, idx=f"{tensor.shape}_{h}x{w}_{i}_{j}", y_range=y_range)

    def visualize_1d_tensor(tensor_1d, save_dir, tensor_name, idx, y_range):
        # Creating x-axis data points
        x = np.arange(0, len(tensor_1d), 1)

        # Plotting the tensor
        plt.figure(figsize=(40.96, 5.12))  # set the figure size to 1920x1080
        plt.plot(x, tensor_1d, linewidth=0.5)

        # Adding plot title and axis labels
        plt.title(f"{tensor_name} {idx}")
        plt.xlabel("Weight Index")
        plt.ylabel("Value")

        # Set the y-axis range based on the range of tensor_A values
        y_mean = torch.mean(tensor_1d).item()
        plt.ylim(y_mean - y_range/2, y_mean + y_range/2)

        # Turn off interactive mode
        plt.ioff()
        # Saving the plot
        plt.savefig(f"{save_dir}/{idx}.png")
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
