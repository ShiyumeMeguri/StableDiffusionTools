from typing import Any, Literal
from pathlib import Path
import torch
import argparse
import numpy as np
from tqdm import tqdm

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

def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """执行两个张量之间的球面线性插值（SLERP）。"""
    # 将张量展平成一维
    v0_flat = v0.view(-1)
    v1_flat = v1.view(-1)

    # 计算范数
    v0_norm = torch.norm(v0_flat)
    v1_norm = torch.norm(v1_flat)

    # 归一化
    v0_unit = v0_flat / v0_norm
    v1_unit = v1_flat / v1_norm

    # 计算点积并进行数值稳定性处理
    dot = torch.dot(v0_unit, v1_unit)
    dot = torch.clamp(dot, -1.0, 1.0)

    # 计算角度和sin值
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    # 处理角度过小的情况，避免除以零
    if sin_omega.abs() < 1e-6:
        # 使用线性插值
        slerp_result = (1.0 - t) * v0 + t * v1
    else:
        # 计算插值权重
        coeff_0 = torch.sin((1.0 - t) * omega) / sin_omega
        coeff_1 = torch.sin(t * omega) / sin_omega
        # 组合结果并恢复原始形状
        slerp_result = coeff_0 * v0 + coeff_1 * v1

    return slerp_result.view(v0.shape)

def calculate_weights(weight_A: torch.Tensor, weight_B: torch.Tensor, base_weight: torch.Tensor, ratio: float, mode: str) -> torch.Tensor:
    if mode == "replace":
        return weight_B * ratio
    elif mode == "linear_combination":
        return weight_A * (1 - ratio) + weight_B * ratio
    elif mode == "slerp":
        return slerp(weight_A, weight_B, ratio)
    elif mode == "ties":
        # 将所有权重移动到 GPU
        weight_A = weight_A.to(torch.float32).cuda()
        weight_B = weight_B.to(torch.float32).cuda()
        base_weight = base_weight.to(torch.float32).cuda()

        # 计算任务向量
        delta_A = weight_A - base_weight
        delta_B = weight_B - base_weight

        # 构建子空间
        subspace = torch.stack([delta_A.view(-1), delta_B.view(-1)], dim=1)

        # 计算子空间的基
        U, S, V = torch.svd(subspace)

        # 投影到子空间
        proj_A = U.t().matmul(delta_A.view(-1))
        proj_B = U.t().matmul(delta_B.view(-1))

        # 在子空间中插值
        interpolated_proj = proj_A * (1 - ratio) + proj_B * ratio

        # 从子空间还原
        interpolated_delta = U.matmul(interpolated_proj)

        # 重构权重
        interpolated_weight = base_weight + interpolated_delta.view(weight_A.shape)

        # 保持原始数据类型
        interpolated_weight = interpolated_weight.to(torch.float16).cpu()

        return interpolated_weight
    else:
        return weight_A


def process_layers(
    state_dict_A: dict[str, torch.Tensor], 
    state_dict_B: dict[str, torch.Tensor], 
    state_dict_base: dict[str, torch.Tensor],
    config_dict: dict[str, float], 
    mode: str = "slerp"  # 默认的计算模式是 SLERP
) -> dict[str, torch.Tensor]:
    merged_state_dict = {}

    for layer_name, weight_A in tqdm(state_dict_A.items(), desc="Processing layers"):
        if layer_name in config_dict:
            ratio = config_dict[layer_name]
            layer_name_without_model = layer_name#.replace("model.", "")
            if state_dict_B and layer_name_without_model in state_dict_B:
                weight_B = state_dict_B[layer_name_without_model]
                weight_base = state_dict_base[layer_name_without_model]
                merged_state_dict[layer_name] = calculate_weights(weight_A, weight_B, weight_base, ratio, mode)
            else:
                if ratio != 0.0:
                    merged_state_dict[layer_name] = weight_A * ratio
                else:
                    merged_state_dict[layer_name] = weight_A
        else:
            print(f"跳过层: {layer_name}")

    return merged_state_dict

def save_default_config(model: dict[str, torch.Tensor], config_path: Path):
    with open(config_path, 'w') as f:
        for name in model.keys():
            f.write(f"{name}::1.0\n")  # 默认权重

def load_config(config_path: str) -> dict[str, float]:
    config_dict = {}
    if config_path:
        with open(config_path, "r") as config_file:
            for line in config_file:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if line.startswith("-"):
                    layer_name, ratio = line.split("::")
                    print(f"已删除 {layer_name} 层")
                    continue
                if line:
                    layer_name, ratio = line.split("::")
                    ratio = float(ratio)
                    config_dict[layer_name] = ratio
    return config_dict

def main():
    input_path = Path(args.input)
    config_path = args.config
    output_path = args.output
    mode = args.mode  # 新增参数 mode

    model_path = None
    if args.model:
        model_path = Path(args.model)
        
    if mode == "ties" and args.base_model:
        base_model_path = Path(args.base_model)
    else:
        print("ties模式需要提供基模型的路径 --base_model")
        return

    # 加载模型权重
    state_dict_A = load_model(input_path, "cpu")
    state_dict_B = None
    if model_path:
        state_dict_B = load_model(model_path, "cpu")
    state_dict_base = None
    if base_model_path:
        state_dict_base = load_model(base_model_path, "cpu")

    # 如果没有提供输出路径，则根据输入路径和模式确定文件名
    if not output_path:
        if model_path:
            output_path = input_path.with_name(f"{input_path.stem}+{model_path.stem}_{mode}.ckpt")
        else:
            output_path = input_path.with_name(f"{input_path.stem}_{mode}.ckpt")
    elif not output_path.endswith(".safetensors") and not output_path.endswith(".ckpt"):
        output_path += ".ckpt"

    output_path = Path(output_path)

    state_dict_A = load_model(input_path, "cpu")
    state_dict_B = None
    if model_path:
        state_dict_B = load_model(model_path, "cpu")
    if base_model_path:
        state_dict_base = load_model(base_model_path, "cpu")

    if not config_path:
        # 没有提供config时，生成默认config文件
        config_path = input_path.with_name(input_path.stem + "_config.txt")
        print(f"缺少配置 已根据输入模型生成了权重配置文件 {config_path}")
        save_default_config(state_dict_A, config_path)
        return

    config_dict = load_config(config_path)
    
    # 在调用 process_layers 时传递 mode 参数
    merged_state_dict = process_layers(state_dict_A, state_dict_B, state_dict_base, config_dict, mode)
    
    format = output_path.suffix[1:]  # Remove the leading dot
    save_state_dict(merged_state_dict, output_path, format)  # Save merged state_dict
    print(f"Saved to {output_path.absolute()}")

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
parser.add_argument("config", type=str, nargs='?', help="Path to configuration file. If not provided, model layers will be printed.")
parser.add_argument("--output", "-o", type=str, help="Path to output file. If not provided, defaults to input+_<mode>.ckpt.")
parser.add_argument("--model", type=str, help="Path to model file. Must be a .safetensors or .ckpt file.")
parser.add_argument("--base_model", type=str, help="TIES模式必写 基模型文件的路径，必须是 .safetensors 或 .ckpt 文件。")
parser.add_argument("--mode", type=str, default="ties", help="Mode of weight calculation: 'linear_combination', 'replace', 'slerp', 'ties'")  # 添加 'ties' 模式
args = parser.parse_args()

if __name__ == "__main__":
    if not args.input:
        parser.print_help()
        exit()

    main()
