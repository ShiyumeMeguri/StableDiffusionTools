from pathlib import Path
import torch
import argparse
import numpy as np
import os
from typing import Any, Literal

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
    save_state_dict(inputB_model, output)

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
