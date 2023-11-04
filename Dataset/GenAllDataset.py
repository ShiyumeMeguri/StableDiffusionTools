import os

# 获取当前工作目录的绝对路径
current_dir = os.path.abspath(os.path.dirname(__file__))

# 遍历当前文件夹内不以数字开头的文件夹
folder_names = [folder_name for folder_name in os.listdir(current_dir) if not folder_name[0].isdigit() and os.path.isdir(folder_name)]

# 遍历找到的文件夹
bat_commands = []
for folder_name in folder_names:
    img_folder_path = os.path.join(folder_name, 'img')
    if os.path.exists(img_folder_path):
        # 遍历img文件夹内的文件夹
        for sub_folder_name in os.listdir(img_folder_path):
            sub_folder_path = os.path.join(img_folder_path, sub_folder_name)
            if os.path.isdir(sub_folder_path):
                sub_folder_path = os.path.abspath(sub_folder_path)

                batch_add_content = ""
                chara_tag = os.path.basename(sub_folder_path.lower())
                chara_tags_set = set([chara_tag])
                if folder_name.endswith("_Chara"):
                    # 增加的功能：检查子文件夹中所有的txt文件并统计包含"1girl"和"1boy"的文件数
                    girl_count = 0
                    boy_count = 0
                    for file_name in os.listdir(sub_folder_path):
                        if file_name.endswith('.txt'):
                            # 查找以"chara_tag"开头到第一个"_"符号位置的字母
                            tag_start = chara_tag.split('_')[0] + "_"

                            with open(os.path.join(sub_folder_path, file_name), 'r', encoding='utf8') as file:
                                content = file.read()
                                tags_array = content.split(', ')

                                for tag in tags_array:
                                    if tag.startswith(tag_start):
                                        chara_tags_set.add(tag)  # 将符合条件的内容添加到批量内容中

                                if '1girl' in tags_array:
                                    girl_count += 1
                                if '1boy' in tags_array:
                                    boy_count += 1

                    # 选择数量较多的关键词
                    key_word = '1girl' if girl_count >= boy_count else '1boy'
                    batch_add_content = f" --chara {key_word}," + ','.join(chara_tags_set)
                # 生成bat命令
                bat_commands.append(f'D:\StableDiffusionTools\Dataset\TrainSetup.py "{sub_folder_path}" {folder_name}{batch_add_content}')

# 将bat命令写入文件
with open('GenAllDataset.bat', 'w') as f:
    f.write('\n'.join(bat_commands))
