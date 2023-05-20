import argparse
import os
import shutil
import glob
import json
from pathlib import Path
from PIL import Image
import subprocess
import configparser

config = configparser.ConfigParser()
config.read('TrainSetupConfig.ini')


base_model = config.get('DEFAULT', 'model_path')
lora_pruneder = config.get('DEFAULT', 'lora_pruneder')
sd_scripts_path = config.get('DEFAULT', 'sd_scripts_path')
dataset_root_path = config.get('DEFAULT', 'dataset_root_path')
lr_scheduler = config.get('DEFAULT', 'lr_scheduler')
learning_rate_finetune = config.get('DEFAULT', 'learning_rate_finetune')
train_step_finetune = config.get('DEFAULT', 'train_step_finetune')
learning_rate_lora = config.get('DEFAULT', 'learning_rate_lora')
resolution_finetune = config.get('DEFAULT', 'resolution_finetune')
batch_size_lora_high = config.get('DEFAULT', 'batch_size_lora_high')
batch_size_lora_low = config.get('DEFAULT', 'batch_size_lora_low')
network_dim_lora = config.get('DEFAULT', 'network_dim_lora')
network_dim_lycoris = config.get('DEFAULT', 'network_dim_lycoris')
resolution_lora_low = config.get('DEFAULT', 'resolution_lora_low')
resolution_lora_high = config.get('DEFAULT', 'resolution_lora_high')
save_model_as = config.get('DEFAULT', 'save_model_as')
use_blip = config.getboolean('DEFAULT', 'use_blip')
#缓存潜在空间
#python prepare_buckets_latents.py --full_path D:\DataSet\BlueArchiveAnime\img\Style D:\DataSet\BlueArchiveAnime\meta_cap_Style.json D:\DataSet\BlueArchiveAnime\meta_cap_Style_prepare_buckets_latents.json D:/stable-diffusion-webui/models/_TempModel/NovelAI/animefull-latest.ckpt --batch_size 4 --max_resolution 512 --mixed_precision fp16

def create_folder_structure(dataset_root_path, dataset_path, folder_name):
    base_path = Path(dataset_root_path)
    target_folder = base_path / folder_name
    img_folder = target_folder / 'img'
    model_folder = target_folder / 'model'

    target_folder.mkdir(exist_ok=True)
    img_folder.mkdir(exist_ok=True)
    model_folder.mkdir(exist_ok=True)

    img_dst = img_folder / dataset_path.name
    shutil.move(str(dataset_path), str(img_dst))

    return img_dst

def run_scripts(img_dst, num_images, folder_name):
    img_dst_str = str(img_dst)

    blip_prompt = False
    txt_files = glob.glob(os.path.join(img_dst_str, '*.txt'))
    if not txt_files:
        print('找不到txt文件, 使用AI生成prompt')
        if (use_blip):
            blip_prompt = True
            subprocess.run(f'{sd_scripts_path}finetune/make_captions.py --caption_extention .txt "{img_dst_str}"', shell=True)
        else:
            subprocess.run(f'{sd_scripts_path}finetune/tag_images_by_wd14_tagger.py "{img_dst_str}"', shell=True)

    else:
        print('找到txt文件, 跳过 tag_images_by_wd14_tagger.py')

    return blip_prompt
    
def flip_images(img_dst):
    for file in os.listdir(img_dst):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".bmp"):
            img = Image.open(os.path.join(img_dst, file)).convert("RGB")
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_img.save(os.path.join(img_dst, os.path.splitext(file)[0] + "_flip.jpg"), quality=90)

            # 查找同名的文本文件
            txt_file = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(img_dst, txt_file)
            if os.path.isfile(txt_path):
                # 如果存在同名的文本文件，则复制一份到翻转后的图片的同一目录下
                flipped_txt_path = os.path.join(img_dst, os.path.splitext(file)[0] + "_flip.txt")
                with open(txt_path, "r") as f_in, open(flipped_txt_path, "w") as f_out:
                    f_out.write(f_in.read())
    print("翻转图片完成.")
    
def data_augmentation(img_dst, num_images):
    has_flipped_images = any('_flip' in f for f in os.listdir(img_dst))
    if num_images < 1000 and not has_flipped_images:
        print(f'图片少于1000,总数量: "{num_images}" 开启数据增强')
        flip_images(img_dst)
        return True
    elif has_flipped_images:
        print('文件夹已包含_flipped结尾的图片，跳过数据增强')
    else:
        print(f'图片数量为 "{num_images}"，不需要进行数据增强')

    return False

def merge_captions(img_dst, json_path, blip_prompt):
    subprocess.run(f'{sd_scripts_path}finetune/merge_captions_to_metadata.py --caption_extension=.txt --full_path "{img_dst}" {json_path}', shell=True)

    if (blip_prompt):
        print('清理错误Prompt中')
        subprocess.run(f'{sd_scripts_path}finetune/clean_captions_and_tags.py {json_path} {json_path}', shell=True)


def create_toml_config(img_dst, json_path, folder_name, resolution, batch_size, training_type, customName=""):
    toml_config = f"""[general]
enable_bucket = true
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = {batch_size}

  [[datasets.subsets]]
  image_dir = '{img_dst}'
  metadata_file = '{json_path}'
"""

    toml_file = f'{dataset_root_path}{folder_name}/{folder_name}{img_dst.name}_{training_type}{customName}.toml'
    with open(toml_file, 'w') as f:
        f.write(toml_config)

    return toml_file
        
def create_batch_file(img_dst, json_path, toml_file1024, toml_file512, folder_name, num_images, training_type, lr, train_step, network_dim=1, conv_dim=1):
    # Compute the number of steps per epoch
    if num_images < batch_size:
        temp_batch_size = num_images
    else:
        temp_batch_size = batch_size
    save_every_n_epochs = math.ceil(1024 / (num_images * temp_batch_size))

    network_module = None
    
    if training_type == "LoRA":
        train_script = "train_network"
        network_module = "networks.lora"
    elif training_type == "LyCORIS":
        train_script = "train_network"
        network_module = "lycoris.kohya"
    elif training_type == "FineTune":
        train_script = "fine_tune"
        
    
    def process_json_file(file_path, tags):
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
                remaining_tags = [tags_array[0]] + [tag for tag in tags_array[1:] if tag in value["caption"]]
                
                # 如果有，就只保留这些tag，其他的tag全部删除
                if remaining_tags:
                    new_data[key] = {"caption": ", ".join(remaining_tags)}
            
        output_file_path = file_path.replace('.json', '_CharaPrompt.json')
        with open(output_file_path, 'w') as f:
            json.dump(new_data, f, indent=2)

    
    batch_content = ""
    batch_add_content = ""
    count = 1
    if "FineTune" in training_type:
        batch_content = f"""{sd_scripts_path}{train_script}.py --pretrained_model_name_or_path={base_model} --output_dir="{dataset_root_path}{folder_name}/model" --output_name={folder_name}{img_dst.name}_{training_type}{lr_scheduler} --dataset_config="{toml_file1024}" --save_model_as={save_model_as} --learning_rate={lr} --max_train_steps={train_step} --optimizer_type AdamW8bit --xformers --gradient_checkpointing --mixed_precision=fp16 --save_every_n_epochs={save_every_n_epochs} --clip_skip=2 --cache_latents --lr_scheduler="{lr_scheduler}" """
    else:
        for i in range(4):
            if i > 1:
                lr = 0.0001
                train_step = 800
                save_every_n_epochs = 4
            if i > 7: #暂时不用
                lr = 0.0001
                train_step = 2000
                process_json_file(json_path, args.chara)
                batch_add_content = "--network_train_text_encoder_only"
                
            if count % 2 == 1:
                current_toml_file = toml_file512
            else:
                current_toml_file = toml_file1024

            batch_content += f"""{sd_scripts_path}{train_script}.py --pretrained_model_name_or_path={base_model} --output_dir="{dataset_root_path}{folder_name}/model" --output_name={count}_{folder_name}_{img_dst.name}_{training_type}{lr_scheduler} --dataset_config="{current_toml_file}" --save_model_as={save_model_as} --learning_rate={lr} --max_train_steps={train_step} --optimizer_type AdamW8bit --xformers --gradient_checkpointing --mixed_precision=fp16 --save_every_n_epochs={save_every_n_epochs} --clip_skip=2 --cache_latents --lr_scheduler="{lr_scheduler}" """
            if count -1 > 0:
                batch_content += f"""--network_weights model/{count-1}_{folder_name}_{img_dst.name}_{training_type}{lr_scheduler}.{save_model_as} """
            if training_type == "LoRA" or "LyCORIS":
                batch_content += f"""--network_module={network_module} --network_dim {network_dim} --network_alpha 1 --network_args "conv_dim={conv_dim}" "conv_alpha=1" "algo=lora" {batch_add_content} 
"""
            if os.path.isfile(lora_pruneder):
                batch_content += f"""{lora_pruneder} model/{count}_{folder_name}_{img_dst.name}_{training_type}{lr_scheduler}.{save_model_as} model/pruned_{count}_{folder_name}_{img_dst.name}_{training_type}{lr_scheduler}.{save_model_as} ALL
"""
            count += 1
            #--network_train_text_encoder_only
    
    batch_file = f'{dataset_root_path}{folder_name}/{folder_name}{img_dst.name}_{training_type}.bat'
    with open(batch_file, 'w') as f:
        f.write(batch_content)

def main():
    dataset_path = args.path
    folder_name = args.name
    
    dataset_path = Path(dataset_path)
    img_dst = create_folder_structure(dataset_root_path, dataset_path, folder_name)
    num_images = len([f for f in os.listdir(img_dst) if os.path.isfile(os.path.join(img_dst, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    blip_prompt = run_scripts(img_dst, num_images, folder_name)
    has_flipped_images = data_augmentation(img_dst, num_images)
    if has_flipped_images:
        num_images *= 2

    json_path = f'{dataset_root_path}{folder_name}/meta_cap_{img_dst.name}.json'

    # 检查文件是否存在，如果存在则删除
    if os.path.exists(json_path):
        os.remove(json_path)

    # 合并提示
    merge_captions(img_dst, json_path, blip_prompt)

    conv_dim = 4
    
    use_type = "FineTune"
    fine_tune_toml_file = create_toml_config(img_dst, json_path, folder_name, resolution=resolution_finetune, batch_size=1, training_type=use_type)
    create_batch_file(img_dst, json_path, fine_tune_toml_file, fine_tune_toml_file, folder_name, num_images, training_type=use_type, lr=learning_rate_finetune, train_step=train_step_finetune)
    
    use_type = "LoRA"
    lora_toml_file1024 = create_toml_config(img_dst, json_path, folder_name, resolution=1024, batch_size=batch_size_lora_high, training_type=use_type, customName="_HighDiffuse1024")
    lora_toml_file512 = create_toml_config(img_dst, json_path, folder_name, resolution=512, batch_size=batch_size_lora_low, training_type=use_type, customName="_HighDiffuse512")
    create_batch_file(img_dst, json_path, lora_toml_file1024, lora_toml_file512, folder_name, num_images, training_type=use_type, lr=1e-3, train_step=num_images, network_dim=network_dim_lora, conv_dim=conv_dim)
    
    use_type = "LyCORIS"
    lora_toml_file1024 = create_toml_config(img_dst, json_path, folder_name, resolution=1024, batch_size=batch_size_lora_high, training_type=use_type, customName="_HighDiffuse1024")
    lora_toml_file512 = create_toml_config(img_dst, json_path, folder_name, resolution=512, batch_size=batch_size_lora_low, training_type=use_type, customName="_HighDiffuse512")
    create_batch_file(img_dst, json_path, lora_toml_file1024, lora_toml_file512, folder_name, num_images, training_type=use_type, lr=1e-3, train_step=num_images, network_dim=network_dim_lycoris, conv_dim=conv_dim)

# 解析命令行参数
parser = argparse.ArgumentParser(description='自动化配置数据集.')
parser.add_argument('path', type=str, help='the folder path to process')
parser.add_argument('name', type=str, help='folder parent name')
parser.add_argument('--chara', type=str, help='chara prompt')

args = parser.parse_args()

if __name__ == "__main__":
    main()
