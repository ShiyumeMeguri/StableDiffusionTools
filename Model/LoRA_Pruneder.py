from typing import Any, Literal
from pathlib import Path
import re
import math
import torch
import argparse

DEF_WEIGHT_PRESET = "\
NONE:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
ALL_TE:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n\
ALL:0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n\
INS:1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
IND:1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
INALL:1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
MIDD:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0\n\
OUTD:1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0\n\
OUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1\n\
OUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1\n\
CHARA_TE:1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0\n\
CHARA:0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0\n\
STYLE_TE:1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0\n\
STYLE:0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1\n\
STYLE_PLUS0_TE:1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1\n\
STYLE_PLUS0:0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1\n\
STYLE_PLUS1_TE:1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1\n\
STYLE_PLUS1:0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1\n\
ALL0.5:0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5"
# 修改input层会因为自己的数据集而让构图受到污染 大部分情况下这就是导致图像崩坏的原因 (只训练人看不出来 风景也一起练你就懂了)
# 如果不是特意训练人体的模型 不建议使用 STYLE_PLUS 如果只靠output层实在还原不了画风再考虑SYTLE_PLUS
# 尽可能只使用output层来还原画风 这样的模型在人物以外的prompt也能保证基础模型的水平(前提你不是用的垃圾模型来合并)
# STYLE_PLUS0 实际上对构图影响大的只有 input7 和 input8 所以在只使用output层就能很好还原画风的情况下 用这个可以保留更多画风细节?
# STYLE_PLUS1 发现于PVC手办模型的情况 忽略input7 和 input8不能很好还原手办质感所以作为一种预设 这个选项救不了过拟合 只能减弱过拟合

BLOCKS=["encoder",
"diffusion_model_input_blocks_0_",
"diffusion_model_input_blocks_1_",
"diffusion_model_input_blocks_2_",
"diffusion_model_input_blocks_3_",
"diffusion_model_input_blocks_4_",
"diffusion_model_input_blocks_5_",
"diffusion_model_input_blocks_6_",
"diffusion_model_input_blocks_7_",
"diffusion_model_input_blocks_8_",
"diffusion_model_input_blocks_9_",
"diffusion_model_input_blocks_10_",
"diffusion_model_input_blocks_11_",
"diffusion_model_middle_block_",
"diffusion_model_output_blocks_0_",
"diffusion_model_output_blocks_1_",
"diffusion_model_output_blocks_2_",
"diffusion_model_output_blocks_3_",
"diffusion_model_output_blocks_4_",
"diffusion_model_output_blocks_5_",
"diffusion_model_output_blocks_6_",
"diffusion_model_output_blocks_7_",
"diffusion_model_output_blocks_8_",
"diffusion_model_output_blocks_9_",
"diffusion_model_output_blocks_10_",
"diffusion_model_output_blocks_11_",
"embedders"]

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

re_digits = re.compile(r"\d+")

re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")

re_unet_down_blocks_res = re.compile(r"lora_unet_down_blocks_(\d+)_resnets_(\d+)_(.+)")
re_unet_mid_blocks_res = re.compile(r"lora_unet_mid_block_resnets_(\d+)_(.+)")
re_unet_up_blocks_res = re.compile(r"lora_unet_up_blocks_(\d+)_resnets_(\d+)_(.+)")

re_unet_downsample = re.compile(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv(.+)")
re_unet_upsample = re.compile(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv(.+)")

re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")

re_inherited_weight = re.compile(r"X([+-])?([\d.]+)?")

def convert_diffusers_name_to_compvis(key):
    def match(match_list, regex):
        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, re_unet_down_blocks):
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_mid_blocks):
        return f"diffusion_model_middle_block_1_{m[1]}"

    if match(m, re_unet_up_blocks):
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_down_blocks_res):
        block = f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_mid_blocks_res):
        block = f"diffusion_model_middle_block_{m[0]*2}_"
        if m[1].startswith('conv1'):
            return f"{block}in_layers_2{m[1][len('conv1'):]}"
        elif m[1].startswith('conv2'):
            return f"{block}out_layers_3{m[1][len('conv2'):]}"
        elif m[1].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[1][len('time_emb_proj'):]}"
        elif m[1].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[1][len('conv_shortcut'):]}"

    if match(m, re_unet_up_blocks_res):
        block = f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_downsample):
        return f"diffusion_model_input_blocks_{m[0]*3+3}_0_op{m[1]}"

    if match(m, re_unet_upsample):
        return f"diffusion_model_output_blocks_{m[0]*3 + 2}_{1+(m[0]!=0)}_conv{m[1]}"

    if match(m, re_text_block):
        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key

def expand_ratios(ratios: list[float]) -> list[float]:
    if len(ratios) == 17:
        return [ratios[0]] + [1] + ratios[1:3] + [1] + ratios[3:5] + [1] + ratios[5:7] + [1, 1, 1] + [ratios[7]] + [1, 1, 1] + ratios[8:]
    elif len(ratios) == 26:
        return ratios
    else:
        raise ValueError("权重长度错误.")

def filter_layers(lora: dict[str, torch.Tensor], lwei: list[float]) -> dict[str, torch.Tensor]:
    filtered_state_dict = {}
    for layer_name, weight in lora.items():
        fullkey = convert_diffusers_name_to_compvis(layer_name)
        key, lora_key = fullkey.split(".", 1)
        for i,block in enumerate(BLOCKS):
            if block in key:
                if i == 26:
                    i = 0
                ratio = lwei[i] 
                if ratio != 0:
                    filtered_state_dict[layer_name] = weight * math.sqrt(abs(ratio))
    return filtered_state_dict

def parse_weight_input(ratios):
    if ratios in DEF_WEIGHT_PRESET:
        ratios = DEF_WEIGHT_PRESET.split(ratios + ':')[1].split('\n')[0]
    return [float(r) for r in ratios.split(",")]

def main(input: str, output: str, ratios: str):
    input = Path(input)
    if not output.endswith(".safetensors"):
        output += ".ckpt"
    output = Path(output)

    input_lora = load_model(input, "cpu")
    
    expanded_ratios = expand_ratios(parse_weight_input(ratios))
    filtered_state_dict = filter_layers(input_lora, expanded_ratios)

    format = output.suffix[1:] # Remove the leading dot
    save_state_dict(filtered_state_dict, output, format)
    print(f"Saved to {output.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to input file. Must be a .safetensors or .ckpt file.")
    parser.add_argument("output", type=str, help="Path to output file. Must be a .safetensors or .ckpt file.")
    parser.add_argument("weight", type=str, help="The weight ratios for each block.")
    args = parser.parse_args()

    if not args.input or not args.output:
        parser.print_help()
        exit()

    main(args.input, args.output, args.weight)

