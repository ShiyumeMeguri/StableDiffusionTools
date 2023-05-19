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

def save_model(model, path):
    torch.save(model, path)

def main(input: str, inputB: str, output: str):
    input = Path(input)
    inputB = Path(inputB)

    input_model = load_model(input, "cpu")
    inputB_model = load_model(inputB, "cpu")

    for layer_name, tensor in input_model.items():
        if layer_name in inputB_model:
            if "model." in layer_name:
                tensor = tensor.float()
                tensorB = inputB_model[layer_name].float()
                meanA = torch.mean(torch.abs(tensor))
                meanB = torch.mean(torch.abs(tensorB))
                ratio = meanA / meanB 
                if torch.isfinite(ratio):
                    inputB_model[layer_name] = tensorB * ratio
        else:
            continue
    save_model(inputB_model, output)

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("inputB", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("output", type=str, help="Path to output file. This will be the new B model.")
args = parser.parse_args()

if __name__ == "__main__":
    if not args.input or not args.inputB:
        parser.print_help()
        exit()

    main(args.input, args.inputB, args.output)
