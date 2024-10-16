import argparse
import torch
from pathlib import Path
from tqdm import tqdm

def load_model(path: Path, device: str = 'cpu') -> dict:
    """加载模型到CPU"""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path, device=device)
    else:
        ckpt = torch.load(path, map_location=device)
        return ckpt.get("state_dict", ckpt)

def save_model(state_dict: dict, path: Path) -> None:
    """保存模型"""
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ModuleNotFoundError('需要安装safetensors，请运行 "pip install safetensors"')
        state_dict = {k: v.contiguous().to_dense() for k, v in state_dict.items()}
        save_file(state_dict, path)
    else:
        torch.save({"state_dict": state_dict}, path)

def compute_enhanced_model(model_a, model_b, base_model, positive_ratio, negative_ratio, device='cuda'):
    """在GPU上计算增强模型，融合模型A和B的特征，同时避免负数削弱特征"""
    enhanced_model = {}
    
    for layer_name in tqdm(model_a.keys(), desc="计算增强模型"):
        # 将模型A、B及基础模型的权重移动到GPU进行计算
        a_layer = model_a[layer_name].to(device)
        base_layer = base_model[layer_name].to(device) if layer_name in base_model else a_layer
        b_layer = model_b[layer_name].to(device) if layer_name in model_b else a_layer

        # 计算模型A和模型B相对于基础模型的权重差异
        delta_a = a_layer - base_layer
        delta_b = b_layer - base_layer
        
        # 初始化 diff
        diff = torch.zeros_like(a_layer, device=device)

        # 情况1：delta_a 和 delta_b 同号，使用 delta_b - delta_a 计算差异
        mask_same_sign = (delta_a * delta_b) >= 0
        diff[mask_same_sign] = (delta_b[mask_same_sign] - delta_a[mask_same_sign]) * positive_ratio

        # 情况2：delta_a 和 delta_b 符号相反，取绝对值和的平均并还原符号
        mask_diff_sign = (delta_a * delta_b) < 0
        # 计算绝对值和的平均值
        mean_abs = (delta_a[mask_diff_sign].abs() + delta_b[mask_diff_sign].abs()) * negative_ratio
        # 使用 delta_b 的符号来恢复方向
        diff[mask_diff_sign] = mean_abs * torch.sign(delta_b[mask_diff_sign])

        # 将融合后的权重加回模型A
        enhanced_model[layer_name] = (a_layer + diff).cpu()  # 计算完成后移回CPU

    return enhanced_model

def main():
    parser = argparse.ArgumentParser(description="模型特征合并到A脚本")
    parser.add_argument("model_a", type=str, help="模型A的路径（经过任务A微调）")
    parser.add_argument("model_b", type=str, help="模型B的路径（经过任务B微调）")
    parser.add_argument("base_model", type=str, help="基础模型的路径（未微调的原始模型）")
    parser.add_argument("--positive_ratio", "-p", type=float, default=1.0, help="正数权重比率，推荐0.5。只使用这个的时候会消除很多细节")
    parser.add_argument("--negative_ratio", "-n", type=float, default=1.0, help="负数权重比率，推荐0.5。只使用这个的时候会添加很多细节")
    parser.add_argument("--output", type=str, help="输出模型的文件名（会自动添加比率信息）", required=False)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型到CPU
    model_a = load_model(Path(args.model_a), device='cpu')
    model_b = load_model(Path(args.model_b), device='cpu')
    base_model = load_model(Path(args.base_model), device='cpu')

    # 在GPU上计算增强模型（如果有GPU）
    enhanced_model = compute_enhanced_model(model_a, model_b, base_model, args.positive_ratio, args.negative_ratio, device=device)
    
    # 确定输出文件名
    output_file = Path(args.output if args.output else "enhanced_model.ckpt")
    output_file = output_file.with_stem(f"{output_file.stem}_p{args.positive_ratio}+n{args.negative_ratio}")
    
    # 保存增强模型
    save_model(enhanced_model, output_file)
    print(f"增强模型已保存到 {output_file}")

if __name__ == "__main__":
    main()
