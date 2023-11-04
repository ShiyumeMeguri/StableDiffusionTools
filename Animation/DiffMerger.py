from PIL import Image
import os

input_dir = "input"
output_dir = "Chara_output"
merge_dir = "merger"

# 遍历input文件夹中所有jpg文件
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        # 获取对应的输出文件名
        output_filename = os.path.join(output_dir, filename)
        # 如果输出文件也存在且是一个jpg文件，就进行合并
        if os.path.isfile(output_filename) and output_filename.endswith(".jpg"):
            input_image = Image.open(os.path.join(input_dir, filename))
            output_image = Image.open(output_filename)
            # 缩放input图片到output图片的大小
            if input_image.size != output_image.size:
                ratio = output_image.size[0] / input_image.size[0]
                input_image = input_image.resize((output_image.size[0], int(input_image.size[1] * ratio)))
            # 获取两张图片的尺寸
            input_width, input_height = input_image.size
            output_width, output_height = output_image.size
            # 创建新的合并后的图片
            merged_image = Image.new("RGB", (input_width + output_width, max(input_height, output_height)))
            merged_image.paste(input_image, (0, 0))
            merged_image.paste(output_image, (input_width, 0))
            # 保存合并后的图片
            merge_filename = os.path.join(merge_dir, filename)
            merged_image.save(merge_filename)
