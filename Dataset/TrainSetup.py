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

config = configparser.ConfigParser()
config.read('TrainSetupConfig.ini')

base_model						=	config.get('DEFAULT', 'base_model')
dataset_root_path				=	config.get('DEFAULT', 'dataset_root_path')
save_model_as					=	config.get('DEFAULT', 'save_model_as')
lr_scheduler					=	config.get('DEFAULT', 'lr_scheduler')
	
lora_pruneder					=	config.get('DEFAULT', 'lora_pruneder')
sd_scripts_path					=	config.get('DEFAULT', 'sd_scripts_path')
	
finetune_lr						=	config.get('DEFAULT', 'finetune_lr')
finetune_batch_size				=	config.get('DEFAULT', 'finetune_batch_size')
finetune_train_step				=	config.get('DEFAULT', 'finetune_train_step')
finetune_resolution				=	config.get('DEFAULT', 'finetune_resolution')
	
lora_unet_lr					=	config.get('DEFAULT', 'lora_unet_lr')
lora_text_encoder_lr			=	config.get('DEFAULT', 'lora_text_encoder_lr')
lora_prior_loss_weight			=	config.get('DEFAULT', 'lora_prior_loss_weight')
lora_batch_size					=	config.get('DEFAULT', 'lora_batch_size')
lora_train_step					=	config.get('DEFAULT', 'lora_train_step')
lora_network_dim				=	config.get('DEFAULT', 'lora_network_dim')
lora_conv_dim					=	config.get('DEFAULT', 'lora_conv_dim')
lora_resolution					=	config.get('DEFAULT', 'lora_resolution')
lora__text_encoder_lr					=	config.get('DEFAULT', 'lora_resolution')
	
lycoris_unet_lr					=	config.get('DEFAULT', 'lycoris_unet_lr')
lycoris_text_encoder_lr			=	config.get('DEFAULT', 'lycoris_text_encoder_lr')
lycoris_prior_loss_weight		=	config.get('DEFAULT', 'lycoris_prior_loss_weight')
lycoris_batch_size				=	config.get('DEFAULT', 'lycoris_batch_size')
lycoris_train_step				=	config.get('DEFAULT', 'lycoris_train_step')
lycoris_network_dim				=	config.get('DEFAULT', 'lycoris_network_dim')
lycoris_conv_dim				=	config.get('DEFAULT', 'lycoris_conv_dim')
lycoris_resolution				=	config.get('DEFAULT', 'lycoris_resolution')
lycoris_resolution				=	config.get('DEFAULT', 'lycoris_resolution')

def create_folder_structure(dataset_root_path, dataset_path, folder_name):
    target_folder = Path(dataset_root_path) / folder_name

    for folder in ['img', 'reg', 'model']:
        (target_folder / folder).mkdir(exist_ok=True)

    image_path = target_folder / 'img' / dataset_path.name
    shutil.move(str(dataset_path), str(image_path))

    return image_path
  
def flip_images(image_path):
    for file in os.listdir(image_path):
        if file.lower().endswith((".jpg", ".png", ".bmp")):
            flipped_img = Image.open(os.path.join(image_path, file)).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
            flipped_img.save(os.path.join(image_path, os.path.splitext(file)[0] + "_flip.png"))

            txt_file = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(image_path, txt_file)
            if os.path.isfile(txt_path):
                flipped_txt_path = os.path.join(image_path, os.path.splitext(file)[0] + "_flip.txt")
                with open(txt_path, "r") as f_in, open(flipped_txt_path, "w") as f_out:
                    f_out.write(f_in.read())
    print("翻转图片完成.")  
    
def run_scripts(image_path, folder_name):
    txt_files = glob.glob(os.path.join(image_path, '*.txt'))
    
    if not txt_files:
        print('找不到txt文件, 使用AI生成prompt')
        subprocess.run(f'{sd_scripts_path}finetune/tag_images_by_wd14_tagger.py "{image_path}"', shell=True)
    else:
        print('找到txt文件, 跳过 tag_images_by_wd14_tagger.py')

def data_augmentation(image_path, num_images):
    has_flipped_images = any('_flip' in f for f in os.listdir(image_path))
    if has_flipped_images:
        print('文件夹已包含_flipped结尾的图片，跳过数据增强')
    else:
        flip_images(image_path)
        return True
    return False
    
def merge_captions(image_path, json_path):
    # 检查文件是否存在，如果存在则删除
    if os.path.exists(json_path):
        os.remove(json_path)
    subprocess.run(f'{sd_scripts_path}finetune/merge_captions_to_metadata.py --caption_extension=.txt --full_path "{image_path}" {json_path}', shell=True)

def create_config(path, data, params):
    with open(path, 'w') as f:
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
  metadata_file = '{json_path}'
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
  caption_extension = '.txt'
  
  [[datasets.subsets]]
  is_reg = true
  image_dir = '{reg_path}'
  class_tokens = '{class_tokens}'
  num_repeats = 1
"""

batch_config = """
{sd_scripts_path}{train_script}.py --pretrained_model_name_or_path={base_model} --dataset_config="{toml_path}" --output_dir={output_dir} --output_name={output_name} --save_model_as={save_model_as} --max_train_steps={train_step} --optimizer_type AdamW8bit --xformers --mixed_precision=fp16 --cache_latents --gradient_checkpointing --save_every_n_epochs={save_every_n_epochs} --lr_scheduler="{lr_scheduler}" """

finetune_batch_config = """--learning_rate={lr} """

lora_batch_config = """--unet_lr={unet_lr} --text_encoder_lr={text_encoder_lr} --network_module={network_module} --network_dim {network_dim} --network_alpha 1 --network_args "conv_dim={conv_dim}" "conv_alpha=1" "algo=lora" --network_train_unet_only --persistent_data_loader_workers --prior_loss_weight={prior_loss_weight} """

def main():
    dataset_path = Path(args.path)
    folder_name = args.name
    base_path = f'{dataset_root_path}{folder_name}'
    
    image_path = create_folder_structure(dataset_root_path, dataset_path, folder_name) # 创建文件夹构造
    num_images = len([f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    run_scripts(image_path, folder_name)
    
    if not args.chara:
        flipped_images = data_augmentation(image_path, num_images)
        if flipped_images:
            num_images *= 2
    
    image_name = image_path.name
    # tag合并
    json_path = f'{base_path}/meta_cap_{image_name}.json'
    # 合并提示
    merge_captions(image_path, json_path)
    
    training_types = ["FineTune", "LoRA", "LyCORIS"]

    for training_type in training_types:
        toml_path = f'{base_path}/{folder_name}_{image_name}_{training_type}.toml'
        toml_config = dreambooth_toml_config if args.use_reg else finetune_toml_config
        
        toml_params = {} 
        toml_params["resolution"] = globals()[f"{training_type.lower()}_resolution"]
        toml_params["batch_size"] = globals()[f"{training_type.lower()}_batch_size"]
        toml_params["image_path"] = image_path
        toml_params["json_path"] = json_path
        toml_params["reg_path"] = args.reg_dir
        toml_params["class_tokens"] = args.reg_tokens
        
        create_config(toml_path, toml_config, toml_params)
        
        if training_type == "LoRA":
            train_script = "train_network"
            network_module = "networks.lora"
        elif training_type == "LyCORIS":
            train_script = "train_network"
            network_module = "lycoris.kohya"
        elif training_type == "FineTune":
            train_script = "fine_tune"

        batch_path = f'{base_path}/{folder_name}_{image_name}_{training_type}.bat'
        bat_config = batch_config
        if training_type == "LoRA" or training_type == "LyCORIS":
            bat_config += lora_batch_config
        elif training_type == "FineTune":
            bat_config += finetune_batch_config
            
        batch_size = globals()[f"{training_type.lower()}_batch_size"]
        temp_batch_size = int(num_images) if int(num_images) < int(batch_size) else int(batch_size)
            
        bat_params = {} 
        bat_params["sd_scripts_path"] = sd_scripts_path
        bat_params["train_script"] = train_script
        bat_params["base_model"] = base_model
        bat_params["output_dir"] = f"{base_path}/model/{image_name}"
        bat_params["output_name"] = f"{folder_name}_{image_name}_{training_type}_{lr_scheduler}"
        bat_params["folder_name"] = folder_name
        bat_params["image_name"] = image_name
        bat_params["training_type"] = training_type
        bat_params["lr_scheduler"] = lr_scheduler
        bat_params["toml_path"] = toml_path
        bat_params["save_model_as"] = save_model_as
        bat_params["train_step"] = globals()[f"{training_type.lower()}_train_step"]
        bat_params["lr"] = finetune_lr
        bat_params["save_every_n_epochs"] = math.ceil(temp_batch_size / (num_images / temp_batch_size))
        
        lora_count = 0
        if training_type == "LoRA" or training_type == "LyCORIS":
            bat_params["output_name"] = f"{lora_count}_{folder_name}_{image_name}_{training_type}_{lr_scheduler}"
            bat_params["unet_lr"] = globals()[f"{training_type.lower()}_unet_lr"]
            bat_params["text_encoder_lr"] = globals()[f"{training_type.lower()}_text_encoder_lr"]
            bat_params["prior_loss_weight"] = globals()[f"{training_type.lower()}_prior_loss_weight"]
            bat_params["network_dim"] = globals()[f"{training_type.lower()}_network_dim"]
            bat_params["conv_dim"] = globals()[f"{training_type.lower()}_conv_dim"]
            bat_params["network_module"] = network_module
            
        def process_json_file(file_path, tags):
            with open(file_path, 'r') as f:
                data = json.load(f)

            tags_array = tags.split(',')

            # 初始化两个新的字典来存放处理后的数据
            included_data = {}
            excluded_data = {}

            # 遍历 JSON 文件中的所有节点
            for key, value in data.items():
                caption = value["caption"]

                # 检查 caption 是否以数组第一个字符串为开头
                if caption.startswith(tags_array[0]):
                    # 如果是，那么查找里面有没有剩余的 tag
                    remaining_tags = [tag for tag in tags_array[1:] if tag in caption]
                    if remaining_tags:
                        included_data[key] = {"caption": ", ".join(remaining_tags)}
                    
                    remaining_tags = [tag for tag in caption.split(', ') if tag.strip() not in tags_array]
                    if remaining_tags:
                        excluded_data[key] = {"caption": ", ".join(remaining_tags)}

            # 保存匹配的标签到文件
            included_output_file_path = file_path.replace('.json', '_Included_CharaPrompt.json')
            with open(included_output_file_path, 'w') as f:
                json.dump(included_data, f, indent=2)

            # 保存排除的标签到文件
            excluded_output_file_path = file_path.replace('.json', '_Excluded_CharaPrompt.json')
            with open(excluded_output_file_path, 'w') as f:
                json.dump(excluded_data, f, indent=2)

            return included_output_file_path, excluded_output_file_path
        
        batch_size = globals()[f"{training_type.lower()}_batch_size"]
        if int(num_images) < int(batch_size):
            temp_batch_size = int(num_images)
        else:
            temp_batch_size = int(batch_size)
        bat_params["save_every_n_epochs"] = math.ceil(temp_batch_size / (num_images / temp_batch_size))
        
        #if training_type == "LoRA" or training_type == "LyCORIS":
        #   charaPrompt = args.chara
        #   if image_name.lower() not in charaPrompt:
        #       charaPrompt += f", {image_name.lower()}"
        #       
        #   included_chara_json, excluded_chara_json = process_json_file(json_path, args.chara)
        #   toml_path = f'{base_path}/{folder_name}_{image_name}_{training_type}_Excluded_CharaPrompt.toml'
        #   lora_count = 1
        #   bat_config += batch_config
        #   if training_type == "LoRA" or training_type == "LyCORIS":
        #       bat_config += lora_batch_config
        #   elif training_type == "FineTune":
        #       bat_config += finetune_batch_config
        #   bat_params["output_name"] = f"{lora_count}_{folder_name}_{image_name}_{training_type}_{lr_scheduler}"
        #   bat_params["toml_path"] = toml_path
        #   toml_params["json_path"] = excluded_chara_json
        #   create_config(toml_path, toml_config, toml_params)
        #   
        #   toml_path = f'{base_path}/{folder_name}_{image_name}_{training_type}_Included_CharaPrompt.toml'
        #   lora_count = 2
        #   bat_config += batch_config
        #   if training_type == "LoRA" or training_type == "LyCORIS":
        #       bat_config += lora_batch_config
        #   elif training_type == "FineTune":
        #       bat_config += finetune_batch_config
        #   bat_params["output_name"] = f"{lora_count}_{folder_name}_{image_name}_{training_type}_{lr_scheduler}"
        #   bat_params["toml_path"] = toml_path
        #   toml_params["json_path"] = included_chara_json
        #   create_config(toml_path, toml_config, toml_params)
                
        create_config(batch_path, bat_config, bat_params)
        
    
# 解析命令行参数
parser = argparse.ArgumentParser(description='自动化配置数据集.')
parser.add_argument('path', type=str, help='the folder path to process')
parser.add_argument('name', type=str, help='folder parent name')

parser.add_argument('--chara', type=str, default='', help='chara prompt')
parser.add_argument('--use_reg', action='store_true', help='use reg train')
parser.add_argument('--reg_dir', type=str, default='', help='the folder path to process')
parser.add_argument('--reg_tokens', type=str, default='', help='chara prompt')
args = parser.parse_args()

if __name__ == "__main__":
    main()
