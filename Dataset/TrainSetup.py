import argparse
import os
import shutil
import glob
from pathlib import Path
import subprocess

def create_folder_structure(dataset_path, folder_name):
    base_path = Path("D:/DataSet")
    target_folder = base_path / folder_name
    img_folder = target_folder / 'img'
    model_folder = target_folder / 'model'

    target_folder.mkdir(exist_ok=True)
    img_folder.mkdir(exist_ok=True)
    model_folder.mkdir(exist_ok=True)

    img_dst = img_folder / dataset_path.name
    shutil.move(str(dataset_path), str(img_dst))

    return img_dst

def run_scripts(img_dst, num_images, folder_name, use_blip):
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

def data_augmentation(img_dst, num_images):
    has_flipped_images = any('_flip' in f for f in os.listdir(img_dst))
    if num_images < 1000 and not has_flipped_images:
        print(f'图片少于1000,总数量: "{num_images}" 开启数据增强')
        subprocess.run(f'D:/DataSet/ImageFlipTool.py --path "{img_dst}"', shell=True)
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


def create_toml_config(img_dst, json_path, folder_name, resolution, batch_size, training_type):
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

    toml_file = f'D:/DataSet/{folder_name}/{folder_name}{img_dst.name}_{training_type}.toml'
    with open(toml_file, 'w') as f:
        f.write(toml_config)

    return toml_file

def create_batch_file(img_dst, toml_file, folder_name, num_images, training_type, num_cpu, lr, train_step, network_dim=8, conv_dim=8):
    save_every_n_epochs = max(1, min(20, 1000 // num_images))
    
    if training_type == "LoRA":
        train_script = "train_network"
        network_module = "networks.lora"
    elif training_type == "LyCORIS":
        train_script = "train_network"
        network_module = "lycoris.kohya"
    elif training_type == "FineTune":
        train_script = "fine_tune"
        
    batch_content = f"""D:/sd-scripts/venv/Scripts/activate.bat && accelerate launch --num_cpu_threads_per_process {num_cpu} {sd_scripts_path}{train_script}.py --pretrained_model_name_or_path={base_model} --output_dir="D:/DataSet/{folder_name}/model" --output_name={folder_name}{img_dst.name}_{training_type}cosine --dataset_config="{toml_file}" --save_model_as=ckpt --learning_rate={lr} --max_train_steps={train_step} --optimizer_type Lion --xformers --gradient_checkpointing --mixed_precision=fp16 --save_every_n_epochs={save_every_n_epochs} --clip_skip=2 --cache_latents --lr_scheduler="cosine" --sample_every_n_epochs 1 --sample_prompts "D:\DataSet\SamplePrompt.txt" --sample_sampler ddim """
    #学习率动态调整方法有 linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
    if training_type == "LoRA" or "LyCORIS":
        batch_content += f"""--network_module=networks.lora --network_train_unet_only --network_dim {network_dim} --network_alpha 1 --network_args "conv_dim={conv_dim}" "conv_alpha=1" "algo=lora" """

    batch_file = f'D:/DataSet/{folder_name}/{folder_name}{img_dst.name}_{training_type}.bat'
    with open(batch_file, 'w') as f:
        f.write(batch_content)

def main(dataset_path, folder_name, use_blip):
    dataset_path = Path(dataset_path)
    img_dst = create_folder_structure(dataset_path, folder_name)
    num_images = len([f for f in os.listdir(img_dst) if os.path.isfile(os.path.join(img_dst, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    blip_prompt = run_scripts(img_dst, num_images, folder_name, use_blip)
    has_flipped_images = data_augmentation(img_dst, num_images)
    if has_flipped_images:
        num_images *= 2

    json_path = f'D:/DataSet/{folder_name}/meta_cap_{img_dst.name}.json'

    # 检查文件是否存在，如果存在则删除
    if os.path.exists(json_path):
        os.remove(json_path)

    # 合并提示
    merge_captions(img_dst, json_path, blip_prompt)

    use_type = "FineTune"
    fine_tune_toml_file = create_toml_config(img_dst, json_path, folder_name, resolution=512, batch_size=1, training_type=use_type)
    create_batch_file(img_dst, fine_tune_toml_file, folder_name, num_images, training_type=use_type, num_cpu=1, lr=2e-6, train_step=10000)
    
    #if folder_name == "Chara":
    #    conv_dim = 1
    #    network_dim = 1
    #elif folder_name == "Style" or "Background" or "Object":
    #    conv_dim = 8
    #    network_dim = 8
    #elif folder_name == "Full":
    #    conv_dim = 16
    #    network_dim = 16
    
    conv_dim = 64
    network_dim = 64
    
    use_type = "LoRA"
    lora_toml_file = create_toml_config(img_dst, json_path, folder_name, resolution=512, batch_size=4, training_type=use_type)
    create_batch_file(img_dst, lora_toml_file, folder_name, num_images, training_type=use_type, num_cpu=1, lr=1e-4, train_step=5000, network_dim=network_dim, conv_dim=conv_dim)
    
    use_type = "LyCORIS"
    lora_toml_file = create_toml_config(img_dst, json_path, folder_name, resolution=512, batch_size=4, training_type=use_type)
    create_batch_file(img_dst, lora_toml_file, folder_name, num_images, training_type=use_type, num_cpu=1, lr=1e-4, train_step=2500, network_dim=network_dim, conv_dim=conv_dim)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='自动化配置数据集.')
    parser.add_argument('path', type=str, help='the folder path to process')
    parser.add_argument('name', type=str, help='folder parent name')
    parser.add_argument('--blip', action='store_true', help='使用 blip 生成prompt')
    args = parser.parse_args()

    base_model = "D:/stable-diffusion-webui/models/_TempModel/NovelAI/animefull-latest.ckpt"
    sd_scripts_path = "D:/sd-scripts/"

    dataset_path = args.path
    folder_name = args.name
    use_blip = args.blip
    main(dataset_path, folder_name, use_blip)
