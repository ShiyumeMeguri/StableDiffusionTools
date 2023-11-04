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

def main(input: str):
    input = Path(input)

    input_model = load_model(input, "cpu")

    # 张量可视化2 转1维
    def visualize_tensor1d(tensor, tensor_name):
        # Create directory for tensor if it doesn't exist
        save_dir = f"{input.stem}/{tensor_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if len(tensor.shape) == 1:  # for 2D tensor
            visualize_1d_tensor(tensor.view(-1), save_dir, tensor_name, idx=tensor.shape)
        elif len(tensor.shape) == 2:  # for 2D tensor
            visualize_1d_tensor(tensor[:, :].view(-1), save_dir, tensor_name, idx=tensor.shape)

        elif len(tensor.shape) == 3:  # for 3D tensor
            c = tensor.shape[-1]  # assume the last dimension is the channel
            for i in range(c):
                visualize_1d_tensor(tensor[:, :, i].view(-1), save_dir, tensor_name, idx=f"{tensor.shape}_{i}")

        elif len(tensor.shape) == 4:  # for 4D tensor
            h, w, c1, c2 = tensor.shape  # assume the last two dimensions are the channels
            #for i in range(w):
            #    for j in range(c1):
            #        for k in range(c2):
            #            visualize_1d_tensor(tensor[:, i, j, k].view(-1), tensor_name, idx=f"{i}_{j}_{k}")
            for i in range(c1):
                for j in range(c2):
                    visualize_1d_tensor(tensor[:, :, i, j].view(-1), save_dir, tensor_name, idx=f"{tensor.shape}_{h}x{w}_{i}_{j}")

    def visualize_1d_tensor(tensor_1d, save_dir, tensor_name, idx):
        # Creating x-axis data points
        x = np.arange(0, len(tensor_1d), 1)

        # Plotting the tensor
        plt.figure(figsize=(40.96, 5.12))  # set the figure size to 1920x1080
        plt.plot(x, tensor_1d, linewidth=0.5)

        # Adding plot title and axis labels
        plt.title(f"{tensor_name} {idx}")
        plt.xlabel("Weight Index")
        plt.ylabel("Value")

        # Turn off interactive mode
        plt.ioff()
        # Saving the plot
        plt.savefig(f"{save_dir}/{tensor_name} {idx}.png")
        plt.close()
    
    for layer_name, tensor in input_model.items():
        visualize_tensor1d(tensor,layer_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        exit()

    main(args.input)

