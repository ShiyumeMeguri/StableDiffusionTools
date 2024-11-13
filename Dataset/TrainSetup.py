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
config_file_path = os.path.join(script_dir, 'TrainSetupConfig.ini')
# 创建配置解析器并读取配置文件
config = configparser.ConfigParser()
config.read(config_file_path)

base_model						=	config.get('DEFAULT', 'base_model')
dataset_root_path				=	config.get('DEFAULT', 'dataset_root_path')
save_model_as					=	config.get('DEFAULT', 'save_model_as')
lr_scheduler					=	config.get('DEFAULT', 'lr_scheduler')

resize_lora_path 				=	config.get('DEFAULT', 'resize_lora_path')
sd_scripts_path					=	config.get('DEFAULT', 'sd_scripts_path')
sample_prompts					=	config.get('DEFAULT', 'sample_prompts')

weight_decay					=	config.get('DEFAULT', 'weight_decay')
noise_offset					=	config.get('DEFAULT', 'noise_offset')
gradient_accumulation_steps     =	config.get('DEFAULT', 'gradient_accumulation_steps')
	
finetune_lr						=	config.get('DEFAULT', 'finetune_lr')
finetune_batch_size				=	config.get('DEFAULT', 'finetune_batch_size')
finetune_train_step				=	config.get('DEFAULT', 'finetune_train_step')
finetune_resolution				=	config.get('DEFAULT', 'finetune_resolution')
	
dreambooth_lr					=	config.get('DEFAULT', 'dreambooth_lr')
dreambooth_batch_size			=	config.get('DEFAULT', 'dreambooth_batch_size')
dreambooth_train_step			=	config.get('DEFAULT', 'dreambooth_train_step')
dreambooth_resolution			=	config.get('DEFAULT', 'dreambooth_resolution')

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
        subprocess.run(f'{sd_scripts_path}finetune/tag_images_by_wd14_tagger.py --batch_size 1 --caption_extension .txt --caption_separator ,  --debug --frequency_tags --max_data_loader_n_workers 2 --onnx --remove_underscore --repo_id SmilingWolf/wd-v1-4-convnextv2-tagger-v2 "{image_path}"', shell=True)
    else:
        print('找到txt文件, 跳过 tag_images_by_wd14_tagger.py')

def merge_captions(image_path, prompt_json_path):
    # 检查文件是否存在，如果存在则删除
    if os.path.exists(prompt_json_path):
        os.remove(prompt_json_path)
    print(f'{sd_scripts_path}finetune/merge_captions_to_metadata.py --caption_extension=.txt --full_path "{image_path}" {prompt_json_path}')
    subprocess.run(f'{sd_scripts_path}finetune/merge_captions_to_metadata.py --caption_extension=.txt --full_path "{image_path}" {prompt_json_path}', shell=True)

def create_config(path, params, data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data.format_map(params))
    
finetune_toml_config = """[general]
enable_bucket = true
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = {batch_size}

  [[datasets.subsets]]
  image_dir = '{image_path}'
  metadata_file = '{prompt_json_path}'
"""

dreambooth_toml_config = """[general]
enable_bucket = true
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = {batch_size}

  [[datasets.subsets]]
  image_dir = '{image_path}'
  class_tokens = '{class_tokens}'
  caption_extension = '.txt'
  
  [[datasets.subsets]]
  is_reg = {is_reg}
  image_dir = '{reg_path}'
  class_tokens = '{class_tokens}'
  num_repeats = 1
"""

lora_toml_config = """[general]
enable_bucket = true
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = {batch_size}

  [[datasets.subsets]]
  image_dir = '{image_path}'
  #metadata_file = '{prompt_json_path}'
  class_tokens = '{class_tokens}'
  caption_extension = '.txt'            # キャプションファイルの拡張子　.txt を使う場合には書き換える
  #caption_prefix = ''
  #caption_suffix = ''
  
  #[[datasets.subsets]]
  #is_reg = true
  #image_dir = '{reg_path}'                      # 正則化画像を入れたフォルダを指定
  #class_tokens = '{class_tokens}'                     # class を指定
  #num_repeats = 1                           # 正則化画像の繰り返し回数、基本的には1でよい
"""

#训练通用配置
base_batch_config = """
{sd_scripts_path}{train_script}.py --pretrained_model_name_or_path={base_model} --dataset_config="{toml_path}" --output_dir={output_dir} --output_name={output_name} --save_model_as={save_model_as} --max_train_steps={train_step} --optimizer_type Lion8bit --xformers --mixed_precision=fp16 --full_fp16 --fp8_base --save_every_n_steps={save_every_n_steps} --lr_scheduler="{lr_scheduler}" """
# v_pred_like_loss 0.1 越高细节学习越好 冲突不要了
# zero_terminal_snr 增强纯噪声还原能力 并避免伪噪声污染和--debiased_estimation_loss --ip_noise_gamma 0.1冲突 会变紫
#--cache_latents 恢复 为了更多的batch
finetune_batch_config = """--learning_rate={lr} """

dreambooth_batch_config = """--learning_rate={lr} """
lora_batch_config = """--unet_lr={unet_lr} --network_module={network_module} --network_dim {network_dim} --network_alpha 1 --network_args "conv_dim={conv_dim}" "conv_alpha=1" "algo=lora" --network_train_unet_only --persistent_data_loader_workers --prior_loss_weight={prior_loss_weight} """

def main():
    dataset_path = Path(args.path)
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
    training_types = ["DreamBooth", "FineTune", "LoRA"]

    for training_type in training_types:
        base_train_path = f'{base_path}/{folder_name}_{image_name}_{training_type}'
        #生成Toml配置
        toml_path = f'{base_train_path}'
        temp_resolution = int(globals()[f"{training_type.lower()}_resolution"])
        toml_path_new = f"{toml_path}_{temp_resolution}.toml"
        
        toml_params = {} 
        toml_params["resolution"] = globals()[f"{training_type.lower()}_resolution"]
        batch_size = int(globals()[f"{training_type.lower()}_batch_size"])
        toml_params["batch_size"] = batch_size
        toml_params["image_path"] = image_path
        toml_params["prompt_json_path"] = prompt_json_path
        toml_params["reg_path"] = args.reg_dir
        toml_params["class_tokens"] = args.reg_tokens
        toml_params["is_reg"] = str(args.reg_dir != '').lower()
        
        # 根据训练类型选择对应的 toml 配置
        if training_type == "DreamBooth" or args.reg_dir:
            toml_config = dreambooth_toml_config
        elif training_type == "LoRA":
            toml_config = lora_toml_config
        else:
            toml_config = finetune_toml_config

        create_config(toml_path_new, toml_params, toml_config)
        
        #生成bat配置
        batch_path = f'{base_train_path}.bat'
        
            
        bat_config = base_batch_config
        if training_type == "LoRA":
            train_script = "sdxl_train_network"
            network_module = "networks.lora"
            bat_config += lora_batch_config
        elif training_type == "FineTune":
            train_script = "fine_tune"
            bat_config += finetune_batch_config
            bat_config +=  f"""--stop_text_encoder_training 0"""
        elif training_type == "DreamBooth":
            train_script = "sdxl_train"
            bat_config += dreambooth_batch_config
            
        #                                                                                          --ip_noise_gamma越大学得越平滑  # 学习细节用--min_snr_gamma 5 和 debiased_estimation_loss二选一 两个一起没啥意义 不能使用 --flip_aug --random_crop  --color_aug
        bat_config += f"""--gradient_checkpointing --loss_type l2 --optimizer_args betas=0.9,0.95 --debiased_estimation_loss --ip_noise_gamma 0.1 --gradient_accumulation_steps={gradient_accumulation_steps} """ # --face_crop_aug_range 1.0,3.0 --cache_text_encoder_outputs weight_decay={weight_decay}   
        if noise_offset: # 开启零终端snr就不用
            bat_config += f"""--noise_offset {noise_offset} """
        
        lr = dreambooth_lr if training_type == "DreamBooth" else finetune_lr
        model_output_dir = f"{base_path}/model/{folder_name}_{image_name}"
        #基本配置参数
        base_output_name = f"{folder_name}_{image_name}_{training_type}_{lr_scheduler}"
        bat_params = {} 
        bat_params["sd_scripts_path"] = sd_scripts_path
        bat_params["train_script"] = train_script
        bat_params["sample_prompts"] = sample_prompts
        bat_params["base_model"] = base_model
        bat_params["output_dir"] = model_output_dir
        bat_params["output_name"] = base_output_name
        bat_params["folder_name"] = folder_name
        bat_params["image_name"] = image_name
        bat_params["training_type"] = training_type
        bat_params["lr_scheduler"] = lr_scheduler
        bat_params["toml_path"] = toml_path_new
        bat_params["save_model_as"] = save_model_as
        base_train_step = int(globals()[f"{training_type.lower()}_train_step"])
        bat_params["train_step"] = base_train_step
        bat_params["lr"] = lr
        bat_params["save_every_n_steps"] = int(max(50, round((math.log(num_images + 1, 10) * 100) / 50) * 50) / (int(batch_size) * int(gradient_accumulation_steps)))
        #添加LoRA参数
        use_lora = training_type == "LoRA"
        if use_lora:
            lora_count = 0
            bat_params["output_name"] = f"{lora_count}_{base_output_name}"
            bat_params["unet_lr"] = globals()[f"{training_type.lower()}_unet_lr"]
            bat_params["text_encoder_lr"] = globals()[f"{training_type.lower()}_text_encoder_lr"]
            bat_params["prior_loss_weight"] = globals()[f"{training_type.lower()}_prior_loss_weight"]
            bat_params["network_dim"] = globals()[f"{training_type.lower()}_network_dim"]
            bat_params["conv_dim"] = globals()[f"{training_type.lower()}_conv_dim"]
            bat_params["network_module"] = network_module

        create_config(batch_path, bat_params, bat_config)
        
# 解析命令行参数
parser = argparse.ArgumentParser(description='自动化配置数据集.')
parser.add_argument('path', type=str, help='the folder path to process')
parser.add_argument('name', type=str, help='folder parent name')

parser.add_argument('--reg_dir', type=str, default='', help='the folder path to process')
parser.add_argument('--reg_tokens', type=str, default='girl', help='reg prompt')
args = parser.parse_args()

if __name__ == "__main__":
    main()
