import os
from PIL import Image, ImageEnhance
import numpy as np
import argparse

def adjust_hue(image, hue_delta):
    """
    Adjust the hue of an image.
    :param image: PIL.Image object.
    :param hue_delta: Hue adjustment value, should be between -180 and 180.
    :return: PIL.Image object with adjusted hue.
    """
    # Convert hue_delta to a scale of 0-1
    hue_delta = hue_delta / 360.0
    img = image.convert('HSV')
    np_img = np.array(img, dtype=np.uint8)
    np_img = np_img.astype('int16')
    np_img[..., 0] += int(hue_delta * 255)
    np_img[..., 0] %= 255
    np_img = np_img.astype('uint8')
    img = Image.fromarray(np_img, 'HSV').convert('RGB')
    return img

def process_images_in_folder(folder_path):
    """
    Process all images in the specified folder, applying hue adjustments and saving the results.
    :param folder_path: Path to the folder containing images.
    """
    output_folder = f"{folder_path}OUTPUT"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            for hue in range(-90, 91, 45):
                adjusted_image = adjust_hue(image, hue)
                # Construct new file name based on the hue adjustment
                new_file_name = f"{os.path.splitext(file_name)[0]}_hue_{hue}.png"
                save_path = os.path.join(output_folder, new_file_name)
                adjusted_image.save(save_path)
                print(f"Saved {save_path}")

parser = argparse.ArgumentParser(description='Copy txt and image files to destination folder.')
parser.add_argument('input', type=str, help='输入数据集路径')

args = parser.parse_args()

process_images_in_folder(args.input)
