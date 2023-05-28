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
	
finetune_lr						=	config.get('DEFAULT', 'finetune_lr')
finetune_batch_size				=	config.get('DEFAULT', 'finetune_batch_size')
finetune_train_step				=	config.get('DEFAULT', 'finetune_train_step')
finetune_resolution				=	config.get('DEFAULT', 'finetune_resolution')
	
chara_down_lr_weight			=	config.get('DEFAULT', 'chara_down_lr_weight')
chara_up_lr_weight				=	config.get('DEFAULT', 'chara_up_lr_weight')

style_down_lr_weight			=	config.get('DEFAULT', 'style_down_lr_weight')
style_up_lr_weight				=	config.get('DEFAULT', 'style_up_lr_weight')

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
        folder_path = os.path.join(target_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

    image_path = target_folder / 'img' / dataset_path.name
    shutil.move(str(dataset_path), str(image_path))

    return image_path
  
def flip_images(image_path):
    for file in os.listdir(image_path):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".bmp"):
            img = Image.open(os.path.join(image_path, file)).convert("RGB")
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_img.save(os.path.join(image_path, os.path.splitext(file)[0] + "_flip.jpg"), quality=90)

            # 查找同名的文本文件
            txt_file = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(image_path, txt_file)
            if os.path.isfile(txt_path):
                # 如果存在同名的文本文件，则复制一份到翻转后的图片的同一目录下
                flipped_txt_path = os.path.join(image_path, os.path.splitext(file)[0] + "_flip.txt")
                with open(txt_path, "r") as f_in, open(flipped_txt_path, "w") as f_out:
                    f_out.write(f_in.read())
    print("翻转图片完成.")
    
def run_scripts(image_path):
    txt_files = glob.glob(os.path.join(image_path, '*.txt'))
    
    if not txt_files:
        print('找不到txt文件, 使用AI生成prompt')
        subprocess.run(f'{sd_scripts_path}finetune/tag_images_by_wd14_tagger.py "{image_path}"', shell=True)
    else:
        print('找到txt文件, 跳过 tag_images_by_wd14_tagger.py')

def data_augmentation(image_path):
    has_flipped_images = any('_flip' in f for f in os.listdir(image_path))
    if has_flipped_images:
        print('文件夹已包含_flipped结尾的图片，跳过数据增强')
    else:
        flip_images(image_path)
        return True
    return False
    
def merge_captions(image_path, prompt_json_path):
    # 检查文件是否存在，如果存在则删除
    if os.path.exists(prompt_json_path):
        os.remove(prompt_json_path)
        
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
  caption_extension = '.txt'
  
  [[datasets.subsets]]
  is_reg = true
  image_dir = '{reg_path}'
  class_tokens = '{class_tokens}'
  num_repeats = 1
"""
#训练通用配置
base_batch_config = """
{sd_scripts_path}{train_script}.py --pretrained_model_name_or_path={base_model} --dataset_config="{toml_path}" --output_dir={output_dir} --output_name={output_name} --save_model_as={save_model_as} --max_train_steps={train_step} --optimizer_type AdamW8bit --xformers --mixed_precision=fp16 --cache_latents --gradient_checkpointing --save_every_n_epochs={save_every_n_epochs} --lr_scheduler="{lr_scheduler}" """

finetune_batch_config = """--learning_rate={lr} """

lora_batch_config = """--unet_lr={unet_lr} --text_encoder_lr={text_encoder_lr} --network_module={network_module} --network_dim {network_dim} --network_alpha 1 --network_args "block_lr_zero_threshold=0.1" "down_lr_weight={down_lr_weight}" "up_lr_weight={up_lr_weight}" "conv_dim={conv_dim}" "conv_alpha=1" "algo=lora" --network_train_unet_only --persistent_data_loader_workers --prior_loss_weight={prior_loss_weight} """

def main():
    dataset_path = Path(args.path)
    folder_name = args.name
    base_path = f'{dataset_root_path}{folder_name}'
    
    image_path = create_folder_structure(dataset_root_path, dataset_path, folder_name) # 创建文件夹构造
    num_images = len([f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    run_scripts(image_path)
    
    if not args.chara:
        flipped_images = data_augmentation(image_path)
        if flipped_images:
            num_images *= 2
    
    image_name = image_path.name
    # tag合并
    prompt_json_path = f'{base_path}/meta_cap_{image_name}.json'
    # 合并提示
    merge_captions(image_path, prompt_json_path)
    
    #训练配置生成
    training_types = ["FineTune", "LoRA", "LyCORIS"]

    for training_type in training_types:
        base_train_path = f'{base_path}/{folder_name}_{image_name}_{training_type}'
        #生成Toml配置
        toml_path = f'{base_train_path}.toml'
        
        toml_params = {} 
        toml_params["resolution"] = globals()[f"{training_type.lower()}_resolution"]
        toml_params["batch_size"] = globals()[f"{training_type.lower()}_batch_size"]
        toml_params["image_path"] = image_path
        toml_params["prompt_json_path"] = prompt_json_path
        toml_params["reg_path"] = args.reg_dir
        toml_params["class_tokens"] = args.reg_tokens
        
        toml_config = dreambooth_toml_config if args.use_reg else finetune_toml_config
        create_config(toml_path, toml_params, toml_config)
        
        #生成bat配置
        batch_path = f'{base_train_path}.bat'
        
        batch_size = globals()[f"{training_type.lower()}_batch_size"]
            
        bat_config = base_batch_config
        if training_type == "LoRA":
            train_script = "train_network"
            network_module = "networks.lora"
            bat_config += lora_batch_config
        elif training_type == "LyCORIS":
            train_script = "train_network"
            network_module = "lycoris.kohya"
            bat_config += lora_batch_config
        elif training_type == "FineTune":
            train_script = "fine_tune"
            bat_config += finetune_batch_config
            
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
        bat_params["train_step"] = globals()[f"{training_type.lower()}_train_step"] * int(num_images / 500) if num_images > 500 else globals()[f"{training_type.lower()}_train_step"]
        bat_params["lr"] = finetune_lr
        bat_params["save_every_n_epochs"] = math.ceil(32 / (num_images / 32))
        
        #添加LoRA参数
        use_lora = training_type == "LoRA" or training_type == "LyCORIS"
        if use_lora:
            bat_params["output_name"] = f"0_{base_output_name}"
            bat_params["unet_lr"] = globals()[f"{training_type.lower()}_unet_lr"]
            bat_params["text_encoder_lr"] = globals()[f"{training_type.lower()}_text_encoder_lr"]
            bat_params["prior_loss_weight"] = globals()[f"{training_type.lower()}_prior_loss_weight"]
            bat_params["network_dim"] = globals()[f"{training_type.lower()}_network_dim"]
            bat_params["conv_dim"] = globals()[f"{training_type.lower()}_conv_dim"]
            bat_params["network_module"] = network_module
            if args.chara:
                bat_params["down_lr_weight"] = chara_down_lr_weight
                bat_params["up_lr_weight"] = chara_up_lr_weight
                #强制格式化一次 不然output_name的名字会被覆盖
                bat_config = bat_config.format_map(bat_params)
                bat_params["output_name"] = f"1_{base_output_name}"
                bat_params["toml_path"] = toml_path
                #第一轮训练全tag 第二轮强化训练角色名提示
                chara_json = process_chara_json_file(prompt_json_path, args.chara)
                toml_path = f'{base_train_path}_CharaPrompt.toml'
                #新建一轮训练
                bat_config += base_batch_config + lora_batch_config
                bat_config += f"""
{resize_lora_path} --new_rank 2 --save_to {model_output_dir}/rank2_{base_output_name}.ckpt --model {model_output_dir}/1_{base_output_name}.ckpt --device cuda"""
                toml_params["prompt_json_path"] = chara_json
                create_config(toml_path, toml_params, toml_config)
            else:
                bat_params["down_lr_weight"] = style_down_lr_weight
                bat_params["up_lr_weight"] = style_up_lr_weight
                
                   
        create_config(batch_path, bat_params, bat_config)
        
    
# 解析命令行参数
parser = argparse.ArgumentParser(description='自动化配置数据集.')
parser.add_argument('path', type=str, help='the folder path to process')
parser.add_argument('name', type=str, help='folder parent name')

parser.add_argument('--chara', type=str, default='', help='chara prompt')
parser.add_argument('--chara_weight', type=str, default='', help='chara weight')

parser.add_argument('--use_reg', action='store_true', help='use reg train')
parser.add_argument('--reg_dir', type=str, default='', help='the folder path to process')
parser.add_argument('--reg_tokens', type=str, default='', help='reg prompt')
args = parser.parse_args()

if __name__ == "__main__":
    main()
