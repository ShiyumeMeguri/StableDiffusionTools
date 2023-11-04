import os
import random
import sys

if len(sys.argv) != 2:
    print("请提供文件夹路径作为参数。例如：python script.py /path/to/your/folder")
    sys.exit(1)

# 获取命令行参数中的文件夹路径
folder_path = sys.argv[1]

# 获取该目录下的所有文件
file_names = os.listdir(folder_path)

for file_name in file_names:
    # 获取文件的扩展名
    extension = os.path.splitext(file_name)[1]
    
    # 生成新的文件名，这里我们使用10位随机数字
    new_file_name = str(random.randint(10**9, 10**10 - 1)) + extension
    
    # 构造完整的文件或路径名
    old_file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # 如果新的文件名与现有文件冲突，则重新生成
    while os.path.exists(new_file_path):
        new_file_name = str(random.randint(10**9, 10**10 - 1)) + extension
        new_file_path = os.path.join(folder_path, new_file_name)
    
    # 重命名文件
    os.rename(old_file_path, new_file_path)

print("所有文件已成功重命名。")
