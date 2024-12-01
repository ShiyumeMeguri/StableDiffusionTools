from pathlib import Path
import argparse
import torch
from typing import Any, Literal
from tqdm import tqdm
import sys

def save_state_dict(state: dict[str, Any], path: str, format: Literal["ckpt", "safetensors"]) -> None:
    if format == "ckpt":
        torch.save({"state_dict": state}, path)
    elif format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ModuleNotFoundError('需要安装 safetensors，运行 "pip install safetensors"')
        state = {k: v.contiguous().to_dense() for k, v in state.items()}
        save_file(state, path)
    else:
        raise ValueError(f"不支持的格式: {format}")

def load_model(path: Path, device: str) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device=device)
    elif path.suffix == ".ckpt":
        ckpt = torch.load(path, map_location=device)
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        else:
            return ckpt
    else:
        raise ValueError(f"不支持的文件扩展名: {path.suffix}")

def main():
    parser = argparse.ArgumentParser(description="平均一个文件夹中所有 .ckpt 或 .safetensors 模型的权重。")
    parser.add_argument("folder", type=str, help="包含 .ckpt 或 .safetensors 模型的文件夹路径。")
    parser.add_argument("--output", "-o", type=str, help="输出文件名（可选）。默认为文件夹名加上 .ckpt 或 .safetensors 扩展名。")
    parser.add_argument("--format", "-f", type=str, choices=["ckpt", "safetensors"], help="输出格式（ckpt 或 safetensors）。如果未指定，将根据文件夹中的模型推断。")
    args = parser.parse_args()

    folder_path = Path(args.folder)
    output_path = args.output
    output_format = args.format

    # 查找文件夹中的所有 .ckpt 或 .safetensors 文件
    model_files = list(folder_path.glob("*.ckpt")) + list(folder_path.glob("*.safetensors"))
    if not model_files:
        print(f"在 {folder_path} 中未找到 .ckpt 或 .safetensors 文件。")
        return

    # 如果未指定输出格式，根据模型文件推断
    if not output_format:
        if any(f.suffix == ".safetensors" for f in model_files):
            output_format = "safetensors"
        else:
            output_format = "ckpt"

    # 如果未指定输出路径，默认为文件夹名加上适当的扩展名
    if not output_path:
        output_filename = folder_path.name + "." + output_format
        output_path = folder_path / output_filename
    else:
        output_path = Path(output_path)

    sum_state_dict = {}
    layer_counts = {}  # 记录每一层出现的次数

    model_count = 0

    for model_file in tqdm(model_files, desc="处理模型"):
        try:
            state_dict = load_model(model_file, device="cpu")
        except Exception as e:
            print(f"加载 {model_file} 失败: {e}", file=sys.stderr)
            continue

        for k, v in state_dict.items():
            if k not in sum_state_dict:
                sum_state_dict[k] = v.float()
                layer_counts[k] = 1
            else:
                sum_state_dict[k] += v.float()
                layer_counts[k] += 1

        model_count += 1

    if model_count == 0:
        print("未处理任何有效模型。")
        return

    # 计算平均权重
    avg_state_dict = {}
    for k, v in sum_state_dict.items():
        avg_state_dict[k] = (v / layer_counts[k]).to(torch.float16)

    # 保存平均后的模型
    save_state_dict(avg_state_dict, str(output_path), output_format)

    print(f"平均模型已保存到 {output_path}")

if __name__ == "__main__":
    main()
