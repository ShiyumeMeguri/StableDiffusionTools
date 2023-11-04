import argparse
from PIL import Image
import os
import sys

# 等比缩放工具
# 解析命令行参数
parser = argparse.ArgumentParser(description='Resize images in a folder.')
parser.add_argument('--path', metavar='path', type=str, help='the folder path to process')
parser.add_argument('--size', metavar='size', type=int, default=1024, help='the maximum size of the longer side (default: 1024)')
args = parser.parse_args()

if not args.path:
    print('Please specify the folder path to process using --path.')
    sys.exit()

# 遍历文件夹路径下的所有文件
for file in os.listdir(args.path):
    # 如果是图片文件，则处理
    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".bmp"):
        # 打开正方形图片
        img = Image.open(os.path.join(args.path, file))
        # 获取正方形图片的宽度和高度
        width, height = img.size
        # 计算缩放后的宽度和高度
        if width >= height:
            new_width = args.size
            new_height = int(height * new_width / width)
        else:
            new_height = args.size
            new_width = int(width * new_height / height)
        # 缩放图片
        resized_img = img.resize((new_width, new_height))
        # 将缩放后的图片保存到指定路径下，并保持原始文件名不变
        resized_img.save(os.path.join(args.path, file))
