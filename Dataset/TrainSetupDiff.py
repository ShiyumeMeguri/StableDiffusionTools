import argparse
import configparser
import glob
import json
import os
import subprocess
import shutil
import math
from PIL import Image
from pathlib import Path

# 获取脚本的实际路径
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(script_dir, 'TrainSetupDiffConfig.ini')
# 创建配置解析器并读取配置文件
config = configparser.ConfigParser()
config.read(config_file_path)

base_model						=	config.get('DEFAULT', 'base_model')
dataset_root_path				=	config.get('DEFAULT', 'dataset_root_path')
save_model_as					=	config.get('DEFAULT', 'save_model_as')
lr_scheduler					=	config.get('DEFAULT', 'lr_scheduler')

resize_lora_path 				=	config.get('DEFAULT', 'resize_lora_path')
sd_scripts_path					=	config.get('DEFAULT', 'sd_scripts_path')
	
lora_unet_lr					=	config.get('DEFAULT', 'lora_unet_lr')
lora_text_encoder_lr			=	config.get('DEFAULT', 'lora_text_encoder_lr')
lora_prior_loss_weight			=	config.get('DEFAULT', 'lora_prior_loss_weight')
lora_batch_size					=	config.get('DEFAULT', 'lora_batch_size')
lora_train_step					=	config.get('DEFAULT', 'lora_train_step')
lora_network_dim				=	config.get('DEFAULT', 'lora_network_dim')
lora_conv_dim					=	config.get('DEFAULT', 'lora_conv_dim')
lora_resolution					=	config.get('DEFAULT', 'lora_resolution')

def create_folder_structure(dataset_root_path, dataset_path, folder_name):
    target_folder = Path(dataset_root_path) / folder_name

    for folder in ['img', 'reg', 'model']:
        folder_path = os.path.join(target_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

    image_path = target_folder / 'img' / dataset_path.name
    shutil.move(str(dataset_path), str(image_path))

    return image_path
    
def run_scripts(image_path):
    txt_files = glob.glob(os.path.join(image_path, '*.txt'))
    
    if not txt_files:
        print('找不到txt文件, 使用AI生成prompt')
        subprocess.run(f'{sd_scripts_path}finetune/tag_images_by_wd14_tagger.py "{image_path}"', shell=True)
    else:
        print('找到txt文件, 跳过 tag_images_by_wd14_tagger.py')
    
def merge_captions(image_path, prompt_json_path):
    # 检查文件是否存在，如果存在则删除
    if os.path.exists(prompt_json_path):
        os.remove(prompt_json_path)
        
    subprocess.run(f'{sd_scripts_path}finetune/merge_captions_to_metadata.py --caption_extension=.txt --full_path "{image_path}" {prompt_json_path}', shell=True)

def create_config(path, data, params=None):
    with open(path, 'w') as f:
        if params:
            f.write(data.format_map(params))
        else:
            f.write(data)

finetune_toml_config = """[general]
enable_bucket = false
shuffle_caption = false
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = {batch_size}

  [[datasets.subsets]]
  image_dir = '{image_path}'
  metadata_file = '{prompt_json_path}'
"""

#训练通用配置
base_batch_config = """
{sd_scripts_path}{train_script}.py --pretrained_model_name_or_path={base_model} --dataset_config="{toml_path}" --output_dir={output_dir} --output_name={output_name} --save_model_as={save_model_as} --max_train_steps={train_step} --optimizer_type AdamW8bit --xformers --mixed_precision=fp16 --cache_latents --gradient_checkpointing --save_every_n_steps={save_every_n_steps} --lr_scheduler="{lr_scheduler}" --seed 1234 """

lora_batch_config = """--unet_lr={unet_lr} --text_encoder_lr={text_encoder_lr} --network_module={network_module} --network_dim {network_dim} --network_alpha 1 --network_args "down_lr_weight=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0" "up_lr_weight=1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0" "mid_lr_weight=0.01" "conv_dim={conv_dim}" "conv_alpha=1" "algo=lora" --network_train_unet_only --persistent_data_loader_workers --prior_loss_weight={prior_loss_weight} """

def process_dataset(path):
    dataset_path = Path(path)
    folder_name = args.name
    base_path = f'{dataset_root_path}{folder_name}'
    
    image_path = create_folder_structure(dataset_root_path, dataset_path, folder_name) # 创建文件夹构造
    num_images = len([f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    run_scripts(image_path)
    
    image_name = image_path.name
    # tag合并
    prompt_json_path = f'{base_path}/meta_cap_{image_name}.json'
    # 合并提示
    merge_captions(image_path, prompt_json_path)
    
    #训练配置生成
    training_type = "LoRA"
    base_train_path = f'{base_path}/{folder_name}_{image_name}_{training_type}'
    #生成Toml配置
    toml_path = f'{base_train_path}.toml'
    
    toml_params = {} 
    toml_params["resolution"] = globals()[f"{training_type.lower()}_resolution"]
    toml_params["batch_size"] = globals()[f"{training_type.lower()}_batch_size"]
    toml_params["image_path"] = image_path
    toml_params["prompt_json_path"] = prompt_json_path
    
    toml_config = finetune_toml_config
    create_config(toml_path, toml_config, toml_params)
    
    batch_size = globals()[f"{training_type.lower()}_batch_size"]
        
    bat_config = base_batch_config
    if training_type == "LoRA":
        train_script = "train_network"
        network_module = "networks.lora"
        bat_config += lora_batch_config
        
    model_output_dir = f"{base_path}/model/{image_name}"
    #基本配置参数
    base_output_name = f"{folder_name}_{image_name}_{training_type}_{lr_scheduler}"
    bat_params = {} 
    bat_params["sd_scripts_path"] = sd_scripts_path
    bat_params["train_script"] = train_script
    bat_params["base_model"] = base_model
    bat_params["output_dir"] = model_output_dir
    bat_params["output_name"] = base_output_name
    bat_params["folder_name"] = folder_name
    bat_params["image_name"] = image_name
    bat_params["training_type"] = training_type
    bat_params["lr_scheduler"] = lr_scheduler
    bat_params["toml_path"] = toml_path
    bat_params["save_model_as"] = save_model_as
    base_train_step = int(globals()[f"{training_type.lower()}_train_step"])
    bat_params["train_step"] = base_train_step
    bat_params["save_every_n_steps"] = max(50, round((math.log(num_images + 1, 10) * 100) / 50) * 50)
    #添加LoRA参数
    bat_params["unet_lr"] = globals()[f"{training_type.lower()}_unet_lr"]
    bat_params["text_encoder_lr"] = globals()[f"{training_type.lower()}_text_encoder_lr"]
    bat_params["prior_loss_weight"] = globals()[f"{training_type.lower()}_prior_loss_weight"]
    bat_params["network_dim"] = globals()[f"{training_type.lower()}_network_dim"]
    bat_params["conv_dim"] = globals()[f"{training_type.lower()}_conv_dim"]
    bat_params["network_module"] = network_module
            
    if args.noise_offset:
        bat_config += f"--noise_offset {args.noise_offset} "
    
    model_path = f"{model_output_dir}/{base_output_name}.{save_model_as}"
    bat_config = bat_config.format_map(bat_params)
    return bat_config, model_path

def main():
    output_name = Path(args.diff).name
    model_output_dir = f"{dataset_root_path}{args.name}/model/{output_name}"
    bat_config = ""
    bat_config1, model_path1 = process_dataset(args.path)
    bat_config2, model_path2 = process_dataset(args.diff)
    bat_config += bat_config1
    bat_config += bat_config2
    bat_config += f"""
{sd_scripts_path}networks/svd_merge_lora.py --models {model_path2} {model_path1} --ratios 1 -1 --new_rank {lora_network_dim} --new_conv_rank {lora_conv_dim} --device cuda --save_to {model_output_dir}/{lora_network_dim}x{lora_conv_dim}_{output_name}.ckpt"""
    
    folder_name = args.name
    batch_path = f'{dataset_root_path}{folder_name}/{folder_name}_{Path(args.diff).name}_DiffTrain_LoRA.bat'
    create_config(batch_path, bat_config)
    
# 解析命令行参数
parser = argparse.ArgumentParser(description='差异训练法自动化配置数据集.')
parser.add_argument('path', type=str, help='原图效果路径')
parser.add_argument('diff', type=str, help='差异路径填目标效果的数据集路径')
parser.add_argument('name', type=str, help='folder parent name')

parser.add_argument('--noise_offset', type=str, default='', help='noise offset')
args = parser.parse_args()

if __name__ == "__main__":
    main()
