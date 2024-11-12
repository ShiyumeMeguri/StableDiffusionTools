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

def resize_tensor_to_match(tensor, target_shape):
    """将张量调整为目标形状"""
    if tensor.shape != target_shape:
        # 假设张量是至少二维的，可以用 interpolate 缩放
        if tensor.dim() >= 2:
            tensor = F.interpolate(tensor.unsqueeze(0), size=target_shape[-2:], mode='bilinear', align_corners=False)
            return tensor.squeeze(0)
        else:
            raise ValueError("张量维度小于2，无法使用插值缩放")
    return tensor
    
def compute_enhanced_model(model_a, model_b, base_model_a, base_model_b, same_ratio, reverse_ratio, device='cuda'):
    """在GPU上计算增强模型，融合模型A和B的特征，同时避免负数削弱特征"""
    enhanced_model = {}
    
    for layer_name in tqdm(model_a.keys(), desc="计算增强模型"):
        # 将模型A、B及基础模型的权重移动到GPU进行计算
        a_layer = model_a[layer_name].to(device)
        b_layer = model_b[layer_name].to(device) if layer_name in model_b else a_layer

        # 如果形状不匹配，将模型B的层调整为模型A的层形状
        if a_layer.shape != b_layer.shape:
            b_layer = resize_tensor_to_match(b_layer, a_layer.shape)
        
        # 如果有 base_model_a 和 base_model_b，计算差异；否则直接相减
        if base_model_a and base_model_b:
            base_layer_a = base_model_a[layer_name].to(device) if base_model_a and layer_name in base_model_a else a_layer
            base_layer_b = base_model_b[layer_name].to(device) if base_model_b and layer_name in base_model_b else a_layer

            # 计算模型A和模型B相对于各自基础模型的权重差异
            delta_a = a_layer - base_layer_a
            delta_b = b_layer - base_layer_b
        else:
            delta_a = a_layer
            delta_b = b_layer
            
        diff = torch.zeros_like(a_layer, device=device)

        # 情况1：delta_a 和 delta_b 同号，使用 delta_b - delta_a 计算差异
        mask_same_sign = (delta_a * delta_b) >= 0
        #diff[mask_same_sign] = (delta_b[mask_same_sign] + delta_a[mask_same_sign]) * same_ratio
        diff[mask_same_sign] = (delta_b[mask_same_sign] - delta_a[mask_same_sign]) * same_ratio

        # 情况2：delta_a 和 delta_b 符号相反，取绝对值和的平均并还原符号
        mask_diff_sign = (delta_a * delta_b) < 0
        #diff[mask_diff_sign] = (delta_b[mask_diff_sign] + delta_a[mask_diff_sign]) * reverse_ratio
        diff[mask_diff_sign] = (delta_b[mask_diff_sign] - delta_a[mask_diff_sign]) * reverse_ratio

        # 将融合后的权重加回模型A
        enhanced_model[layer_name] = (a_layer + diff).cpu()  # 计算完成后移回CPU

    return enhanced_model

def run_fusion(model_a, model_b, base_model_a, base_model_b, same_ratio, reverse_ratio, device, output_file):
    """根据参数运行模型融合"""
    # 在GPU上计算增强模型（如果有GPU）
    enhanced_model = compute_enhanced_model(model_a, model_b, base_model_a, base_model_b, same_ratio, reverse_ratio, device=device)
    
    # 保存增强模型
    save_model(enhanced_model, output_file)
    print(f"增强模型已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description="模型特征合并到A脚本")
    parser.add_argument("model_a", type=str, help="模型A的路径（经过任务A微调）")
    parser.add_argument("model_b", type=str, help="模型B的路径（经过任务B微调）")
    parser.add_argument("--base_model_a", type=str, help="基础模型A的路径（建议填写）", required=False)
    parser.add_argument("--base_model_b", type=str, help="基础模型B的路径（可选）", required=False)
    parser.add_argument("--same_ratio", "-s", type=float, default=1.0, help="模型之间的相同特征比率，推荐1.0。")
    parser.add_argument("--reverse_ratio", "-r", type=float, default=1.0, help="模型之间的不同特征比率，推荐0.5。")
    parser.add_argument("--output", type=str, help="输出模型的文件名（会自动添加比率信息）", required=False)
    parser.add_argument("--debug", action="store_true", help="开启调试模式，模型常驻内存，等待用户输入新的参数")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型到CPU（仅加载一次）
    model_a = load_model(Path(args.model_a), device='cpu')
    model_b = load_model(Path(args.model_b), device='cpu')

    # 加载基础模型（如果提供了）
    base_model_a = load_model(Path(args.base_model_a), device='cpu') if args.base_model_a else None
    base_model_b = load_model(Path(args.base_model_b), device='cpu') if args.base_model_b else base_model_a

    # 获取模型文件的名称（不带后缀）
    model_a_name = Path(args.model_a).stem
    model_b_name = Path(args.model_b).stem

    # 构建文件名后缀
    suffix = f"_s{args.same_ratio}+r{args.reverse_ratio}"

    # 如果指定了输出文件名，则在文件名后加上比率信息
    if args.output:
        output_file = Path(args.output).with_stem(Path(args.output).stem + suffix)
    else:
        output_file = Path(args.model_a).parent / f"{model_a_name}+{model_b_name}{suffix}.ckpt"
    
    # 初次运行模型融合
    run_fusion(model_a, model_b, base_model_a, base_model_b, args.same_ratio, args.reverse_ratio, device, output_file)

    # 如果开启了debug模式，进入循环
    if args.debug:
        while True:
            try:
                # 等待用户输入新的same_ratio和reverse_ratio
                print("进入调试模式，输入新的same_ratio和reverse_ratio，回车确认（格式：same_ratio reverse_ratio），按Ctrl+C退出：")
                user_input = input().strip()
                
                if user_input:
                    same_ratio, reverse_ratio = map(float, user_input.split())
                    args.same_ratio = same_ratio
                    args.reverse_ratio = reverse_ratio

                    # 构建新的文件名后缀
                    suffix = f"_s{same_ratio}+r{reverse_ratio}"
                    if args.output:
                        output_file = Path(args.output).with_stem(Path(args.output).stem + suffix)
                    else:
                        output_file = Path(args.model_a).parent / f"{model_a_name}+{model_b_name}{suffix}.ckpt"

                    # 重新进行融合计算
                    run_fusion(model_a, model_b, base_model_a, base_model_b, same_ratio, reverse_ratio, device, output_file)
                else:
                    print("输入无效，请重新输入。")
            except KeyboardInterrupt:
                print("退出调试模式")
                break

if __name__ == "__main__":
    main()