import cv2
import os
import sys
import numpy as np

def rotate_image(image, angle):
    # 旋转图像而不填充黑边
    (h, w) = image.shape[:2]
    # 计算旋转的中心点
    center = (w / 2, h / 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 计算绕中心旋转后图像所需的最大边框
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算新边界的宽度和高度
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动（平移）部分以考虑到旋转后的平移
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # 执行实际的旋转和平移操作
    rotated_image = cv2.warpAffine(image, M, (nW, nH))
    return rotated_image

def detect_and_crop(filename, output_dir, scale_factor=1, cascade_file="../lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Failed to open", filename)
        return False

    original_image = image.copy()
    rotation_attempts = 0
    faces_detected = 0

    while rotation_attempts < 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(400, 400))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                center_x, center_y = x + w // 2, y + h // 2
                new_size = int(max(w, h) * scale_factor)
                left = max(center_x - new_size // 2, 0)
                top = max(center_y - new_size // 2, 0)
                right = min(center_x + new_size // 2, original_image.shape[1])
                bottom = min(center_y + new_size // 2, original_image.shape[0])
                cropped_image = original_image[top:bottom, left:right]

                if cropped_image.size != 0:
                    save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filename))[0]}x{scale_factor}_face{faces_detected}.png")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, cropped_image)
                    faces_detected += 1
            break  # Exit the loop after processing all faces in the current orientation
        else:
            # Rotate the image for the next attempt
            image = rotate_image(original_image, -90 * (rotation_attempts + 1))
            rotation_attempts += 1

    if faces_detected == 0:
        # If no faces were detected, save the unmodified image in a specific folder
        save_path = os.path.join(output_dir, "Fail", os.path.basename(filename))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, original_image)

    return faces_detected > 0

def process_folder(folder, output_dir):
    image_files = [os.path.join(root, file) for root, dirs, files in os.walk(folder) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    processed_files = 0
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 在调用detect_and_crop函数之前打印当前进度
                processed_files += 1
                print(f"Processing {file} ({processed_files}/{total_images})...")
                detect_and_crop(os.path.join(root, file), output_dir)

if len(sys.argv) != 2:
    sys.stderr.write("usage: detect.py <folder>\n")
    sys.exit(-1)

input_dir = sys.argv[1]
output_dir = input_dir + "Output"
process_folder(input_dir, output_dir)
