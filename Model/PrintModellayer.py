from pathlib import Path
import torch
import argparse

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

def print_layer_info(model: dict[str, torch.Tensor], directory: Path):
    info_path = directory / "model_info.txt"
    max_name_length = max(len(name) for name in model.keys())  # 计算最长层名称长度
    header_format = f"{{:<{max_name_length + 10}}} {{:<20}} {{:>10}} {{:<30}}\n"
    row_format = f"{{:<{max_name_length + 10}}} {{:<20}} {{:10.3f}} {{:<30}}\n"

    with open(info_path, 'w') as f:
        f.write(header_format.format('Layer', 'Dtype', 'Size (MB)', 'Shape'))
        f.write("-" * (max_name_length + 70) + "\n")  # 根据最长名称调整分隔线长度
        for name, tensor in model.items():
            size_mb = tensor.nelement() * tensor.element_size() / (1024 ** 2)
            f.write(row_format.format(name, str(tensor.dtype), size_mb, str(tensor.size())))

def main(input_path: str):
    input_path = Path(input_path)
    input_model = load_model(input_path, "cpu")
    print_layer_info(input_model, input_path.parent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the model file. Must be a .safetensors or .pth or other supported file format.")
    args = parser.parse_args()
    main(args.input)
