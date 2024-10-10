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
	
finetune_lr						=	config.get('DEFAULT', 'finetune_lr')
finetune_batch_size				=	config.get('DEFAULT', 'finetune_batch_size')
finetune_train_step				=	config.get('DEFAULT', 'finetune_train_step')
finetune_resolution				=	config.get('DEFAULT', 'finetune_resolution')
	
dreambooth_lr					=	config.get('DEFAULT', 'dreambooth_lr')
dreambooth_batch_size			=	config.get('DEFAULT', 'dreambooth_batch_size')
dreambooth_train_step			=	config.get('DEFAULT', 'dreambooth_train_step')
dreambooth_resolution			=	config.get('DEFAULT', 'dreambooth_resolution')
	
chara_down_lr_weight			=	config.get('DEFAULT', 'chara_down_lr_weight')
chara_mid_lr_weight		    	=	config.get('DEFAULT', 'chara_mid_lr_weight')
chara_up_lr_weight				=	config.get('DEFAULT', 'chara_up_lr_weight')

style_down_lr_weight			=	config.get('DEFAULT', 'style_down_lr_weight')
style_mid_lr_weight			    =	config.get('DEFAULT', 'style_mid_lr_weight')
style_up_lr_weight				=	config.get('DEFAULT', 'style_up_lr_weight')

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
    with open(path, 'w') as f:
        f.write(data.format_map(params))

def process_chara_json_file(file_path, tags):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    tags_array = tags.split(',')
    
    # 初始化一个新的字典来存放处理后的数据
    new_data = {}
    
    # 遍历json文件中的所有节点
    for key, value in data.items():
        # 检查caption是否以数组第一个字符串为开头
        if value["caption"].startswith(tags_array[0]):
            # 如果是，那么查找里面有没有剩余的tag
            remaining_tags = [tag for tag in tags_array[1:] if tag in value["caption"]]
            
            # 如果有，就只保留这些tag，其他的tag全部删除
            if remaining_tags:
                new_data[key] = {"caption": ", ".join(remaining_tags)}
        
    output_file_path = file_path.replace('.json', '_CharaPrompt.json')
    with open(output_file_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    return output_file_path
    
finetune_toml_config = """[general]
enable_bucket = true
shuffle_caption = false
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
shuffle_caption = false
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = {batch_size}

  [[datasets.subsets]]
  image_dir = '{image_path}'
  caption_extension = '.txt'
  
  [[datasets.subsets]]
  is_reg = {is_reg}
  image_dir = '{reg_path}'
  class_tokens = '{class_tokens}'
  num_repeats = 1
"""
#训练通用配置
base_batch_config = """
{sd_scripts_path}{train_script}.py --pretrained_model_name_or_path={base_model} --dataset_config="{toml_path}" --output_dir={output_dir} --output_name={output_name} --save_model_as={save_model_as} --max_train_steps={train_step} --optimizer_type AdamW8bit --xformers --mixed_precision=fp16 --full_fp16 --gradient_checkpointing --save_every_n_epochs={save_every_n_epochs} --lr_scheduler="{lr_scheduler}" --seed 1234 """# --sample_prompts {sample_prompts} --sample_sampler ddim --sample_every_n_epochs {save_every_n_epochs} """
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
        #if num_images > 200:
        #    batch_size = int(batch_size * (num_images / 200))
        #if batch_size >= 32:
        #    batch_size = 32
        toml_params["batch_size"] = batch_size
        toml_params["image_path"] = image_path
        toml_params["prompt_json_path"] = prompt_json_path
        toml_params["reg_path"] = args.reg_dir
        toml_params["class_tokens"] = args.reg_tokens
        toml_params["is_reg"] = str(args.reg_dir != '').lower()
        
        toml_config = dreambooth_toml_config if args.reg_dir or training_type == "DreamBooth" else finetune_toml_config
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
            
        if not args.chara:
            bat_config += f"""--cache_text_encoder_outputs --flip_aug""" # --face_crop_aug_range 1.0,3.0 --optimizer_args weight_decay={weight_decay} betas=.9,.999 --color_aug 
        if args.noise_offset:
            bat_config += f"""--noise_offset {args.noise_offset} """
        
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
        #if num_images > 500:
        #    base_train_step = int(base_train_step * (num_images / 500))
        bat_params["train_step"] = base_train_step
        bat_params["lr"] = lr
        bat_params["save_every_n_epochs"] = math.ceil(16 / (num_images / 16))
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
            #if args.chara:
            #    bat_params["down_lr_weight"] = chara_down_lr_weight
            #    bat_params["mid_lr_weight"] = chara_mid_lr_weight
            #    bat_params["up_lr_weight"] = chara_up_lr_weight
            #else:
            #    bat_params["down_lr_weight"] = style_down_lr_weight
            #    bat_params["mid_lr_weight"] = style_mid_lr_weight
            #    bat_params["up_lr_weight"] = style_up_lr_weight
            
            #bat_config_list = ""
            #style_up_lr_weight_base = [0.0001]*12
            #
            #for index in range(3, 12):  # index 3 to 11
            #    output_layer = style_up_lr_weight_base.copy()
            #    output_layer[index] = 1.0
            #    output_layer = ",".join(str(x) for x in output_layer)
            #    
            #    bat_params["up_lr_weight"] = output_layer  
            #    bat_params["output_name"] = f"{lora_count}_{base_output_name}"
            #    
            #    temp_bat_config = bat_config.format_map(bat_params)
            #    count = lora_count-1
            #    if count >= 0:
            #        new_bat_config = f"""{temp_bat_config} --network_weights {model_output_dir}/{lora_count-1}_{base_output_name}.{save_model_as}"""
            #    else:
            #        new_bat_config = f"""{temp_bat_config}"""
            #    lora_count += 1
            #    
            #    bat_config_list += new_bat_config
            #
            ## Join all the bat_configs
            #bat_config = bat_config_list

        create_config(batch_path, bat_params, bat_config)
        
# 解析命令行参数
parser = argparse.ArgumentParser(description='自动化配置数据集.')
parser.add_argument('path', type=str, help='the folder path to process')
parser.add_argument('name', type=str, help='folder parent name')

parser.add_argument('--chara', type=str, default='', help='chara prompt')
parser.add_argument('--noise_offset', type=str, default='', help='noise offset')

parser.add_argument('--reg_dir', type=str, default='', help='the folder path to process')
parser.add_argument('--reg_tokens', type=str, default='', help='reg prompt')
args = parser.parse_args()

if __name__ == "__main__":
    main()
