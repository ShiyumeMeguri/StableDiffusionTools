from typing import Any, Literal
from pathlib import Path
import torch
import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

def to_half(sd):
    for key in sd.keys():
        if 'model' in key and sd[key].dtype == torch.float:
            sd[key] = sd[key].half()
    return sd
def save_state_dict(state: dict[str, Any], path: str, format: Literal["ckpt", "safetensors"]) -> None:
    if format == "ckpt":
        state = to_half(state)
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

def main(input: str, inputB: str, output_path: str):
    input = Path(input)
    inputB = Path(inputB)

    input_model = load_model(input, "cpu")
    inputB_model = load_model(inputB, "cpu")
    if not output_path.endswith(".safetensors"):
        output_path += ".ckpt"
    output_path = Path(output_path)

    # Create directory for tensor if it doesn't exist
    save_dir = f"{input.stem} - {inputB.stem}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    replaced_model = {}
    for layer_name, tensor in input_model.items():
        if layer_name in inputB_model:
            input_model[layer_name] = inputB_model[layer_name]
        else:
            # 处理B模型缺少A模型层的情况
            continue
            
    format = output_path.suffix[1:]  # Remove the leading dot
    save_state_dict(input_model, output_path, format)  # Save filtered state_dict
    print(f"Saved to {output_path.absolute()}")


parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("inputB", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("output", type=str, help="Path to output file. Must be a .safetensors or .ckpt file.")
args = parser.parse_args()

if __name__ == "__main__":

    if not args.input or not args.inputB:
        parser.print_help()
        exit()

    main(args.input, args.inputB, args.output)

