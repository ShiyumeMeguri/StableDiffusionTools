import argparse
import torch
from pathlib import Path
from tqdm import tqdm

def load_model(path: Path, device: str) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path, device=device)
    else:
        ckpt = torch.load(path, map_location=device)
        return ckpt.get("state_dict", ckpt)

def save_model(state_dict: dict, path: Path) -> None:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ModuleNotFoundError('需要安装safetensors，请运行 "pip install safetensors"')
        state_dict = {k: v.contiguous().to_dense() for k, v in state_dict.items()}
        save_file(state_dict, path)
    else:
        torch.save({"state_dict": state_dict}, path)

def compute_enhanced_model(model_a, model_b, base_model):
    enhanced_model = {}
    for layer_name in tqdm(model_a.keys(), desc="计算增强模型"):
        a_layer = model_a[layer_name]
        base_layer = base_model[layer_name] if layer_name in base_model else a_layer
        b_layer = model_b[layer_name] if layer_name in model_b else a_layer
        
        delta_a = a_layer - base_layer
        delta_b = b_layer - base_layer
        diff = delta_b - delta_a
        enhanced_model[layer_name] = a_layer + diff
    return enhanced_model

def main():
    parser = argparse.ArgumentParser(description="模型特征合并到A脚本")
    parser.add_argument("--model_a", type=str, required=True, help="模型A的路径（经过任务A微调）")
    parser.add_argument("--model_b", type=str, required=True, help="模型B的路径（经过任务B微调）")
    parser.add_argument("--base_model", type=str, required=True, help="基础模型的路径（未微调的原始模型）")
    parser.add_argument("--output", type=str, required=True, help="输出增强模型的路径")
    args = parser.parse_args()
    
    device = 'cpu'  # 如果有GPU并且想使用，可以设置为 'cuda'

    # 加载模型
    model_a = load_model(Path(args.model_a), device)
    model_b = load_model(Path(args.model_b), device)
    base_model = load_model(Path(args.base_model), device)
    
    # 计算增强模型
    enhanced_model = compute_enhanced_model(model_a, model_b, base_model)
    
    # 保存增强模型
    save_model(enhanced_model, Path(args.output))
    print(f"增强模型已保存到 {args.output}")

if __name__ == "__main__":
    main()
