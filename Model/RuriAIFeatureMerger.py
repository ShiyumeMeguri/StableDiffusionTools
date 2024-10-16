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

def compute_enhanced_model(model_a, model_b, base_model, device='cuda'):
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
        mask_same_sign = ((delta_a > 0) & (delta_b > 0)) | ((delta_a < 0) & (delta_b < 0))
        diff[mask_same_sign] = delta_b[mask_same_sign] - delta_a[mask_same_sign]

        # 情况2：delta_a 和 delta_b 符号相反，使用 delta_b 的符号来保留方向
        mask_diff_sign = ((delta_a > 0) & (delta_b < 0)) | ((delta_a < 0) & (delta_b > 0))
        diff[mask_diff_sign] = (delta_a[mask_diff_sign].abs() + delta_b[mask_diff_sign].abs()) * torch.sign(delta_b[mask_diff_sign])

        # 将融合后的权重加回模型A
        enhanced_model[layer_name] = (a_layer + diff).cpu()  # 计算完成后移回CPU

    return enhanced_model

def main():
    parser = argparse.ArgumentParser(description="模型特征合并到A脚本")
    parser.add_argument("--model_a", type=str, required=True, help="模型A的路径（经过任务A微调）")
    parser.add_argument("--model_b", type=str, required=True, help="模型B的路径（经过任务B微调）")
    parser.add_argument("--base_model", type=str, required=True, help="基础模型的路径（未微调的原始模型）")
    parser.add_argument("--output", type=str, required=True, help="输出增强模型的路径")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型到CPU
    model_a = load_model(Path(args.model_a), device='cpu')
    model_b = load_model(Path(args.model_b), device='cpu')
    base_model = load_model(Path(args.base_model), device='cpu')
    
    # 在GPU上计算增强模型（如果有GPU）
    enhanced_model = compute_enhanced_model(model_a, model_b, base_model, device=device)
    
    # 保存增强模型
    save_model(enhanced_model, Path(args.output))
    print(f"增强模型已保存到 {args.output}")

if __name__ == "__main__":
    main()
