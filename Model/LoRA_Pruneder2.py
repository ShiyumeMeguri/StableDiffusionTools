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


def filter_layers(state_dict: dict[str, torch.Tensor], state_dict_B: dict[str, torch.Tensor], config_path: str) -> dict[str, torch.Tensor]:
    with open(config_path, "r") as config_file:
        config_dict = {}
        for line in config_file:
            line = line.strip()
            if line and not line.startswith("#"):
                layer_name, ratio = line.split("::")
                ratio = float(ratio)
                config_dict[layer_name] = ratio

    filtered_state_dict = {}
    for layer_name, weight in state_dict.items():
        if layer_name in config_dict:
            ratio = config_dict[layer_name]
            if ratio == 0.0:
                continue
            if state_dict_B:
                filtered_state_dict[layer_name] = state_dict_B[layer_name] * math.sqrt(abs(ratio))
            else:
                filtered_state_dict[layer_name] = weight * math.sqrt(abs(ratio))
        filtered_state_dict[layer_name] = torch.zeros_like(weight)

    return filtered_state_dict


def main():
    input_path = Path(args.input)
    config_path = args.config
    output_path = args.output

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = None

    if not output_path.endswith(".safetensors"):
        output_path += ".ckpt"
    output_path = Path(output_path)

    state_dict_A = load_model(input_path, "cpu")
    
    state_dict_B = None
    if model_path:
        state_dict_B = load_model(model_path, "cpu")

    filtered_state_dict = filter_layers(state_dict_A, state_dict_B, config_path)

    format = output_path.suffix[1:]  # Remove the leading dot
    save_state_dict(filtered_state_dict, output_path, format)  # Save filtered state_dict
    print(f"Saved to {output_path.absolute()}")


parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("config", type=str, help="Path to configuration file.")
parser.add_argument("output", type=str, help="Path to output file. Must be a .safetensors or .ckpt file.")
parser.add_argument("--model", type=str, help="Path to model file. Must be a .safetensors or .ckpt file.")
args = parser.parse_args()

if __name__ == "__main__":
    if not args.input or not args.output or not args.config:
        parser.print_help()
        exit()

    main()
