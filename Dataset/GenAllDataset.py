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
            print(sub_folder_name)
            sub_folder_path = os.path.join(img_folder_path, sub_folder_name)
            if os.path.isdir(sub_folder_path):
                sub_folder_path = os.path.abspath(sub_folder_path)

                # 生成bat命令
                bat_commands.append(f'TrainSetup.py "{sub_folder_path}" {folder_name}')
                

# 将bat命令写入文件
with open('GenAllDataset.bat', 'w') as f:
    f.write('\n'.join(bat_commands))
