import os
import cv2
import numpy as np

# 输入1：已经计算好的mask文件夹
mask_folder = "mask_output"

# 输入2：图片序列帧文件夹
image_folder = "ai"

# 输出文件夹
output_folder = "aiFix"

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取mask文件名列表
mask_files = sorted(os.listdir(mask_folder))

# 获取图片文件名列表
image_files = sorted(os.listdir(image_folder))

# 读取所有mask
masks = [cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE) for mask_file in mask_files]

# 初始化同步点矩阵
sync_matrix = np.zeros((len(mask_files), masks[0].shape[0], masks[0].shape[1]), dtype=bool)

# 像素块大小
block_size = 192

# 查找mask中的同步点
for row in range(0, masks[0].shape[0], block_size):
    for col in range(0, masks[0].shape[1], block_size):
        for i in range(1, len(mask_files)):
            if not np.array_equal(masks[i - 1][row:row + block_size, col:col + block_size],
                                  masks[i][row:row + block_size, col:col + block_size]):
                sync_matrix[i, row:row + block_size, col:col + block_size] = True

# 遍历图片文件
reference_image = cv2.imread(os.path.join(image_folder, image_files[0]))
for i, image_file in enumerate(image_files):
    # 读取图片
    image = cv2.imread(os.path.join(image_folder, image_file))

    # 处理图片
    if i > 0:
        for row in range(0, image.shape[0], block_size):
            for col in range(0, image.shape[1], block_size):
                if not sync_matrix[i, row, col]:
                    image[row:row + block_size, col:col + block_size] = reference_image[row:row + block_size, col:col + block_size]

    # 保存处理后的图片
    cv2.imwrite(os.path.join(output_folder, image_file), image)

print("处理完成，结果已保存到", output_folder)
