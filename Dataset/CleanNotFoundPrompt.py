import os
import argparse
import send2trash
import shutil

def find_and_move_txt_without_img(folder_path):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                txt_path = os.path.join(root, file)
                txt_name_without_ext = os.path.splitext(file)[0]

                img_found = False
                for ext in img_extensions:
                    img_path = os.path.join(root, txt_name_without_ext + ext)
                    if os.path.exists(img_path):
                        img_found = True
                        break

                if not img_found:
                    # Move file to the recycle bin
                    try:
                        send2trash.send2trash(txt_path) # Use the full file path
                        send2trash.send2trash(txt_name_without_ext + ".npz") # Use the full file path
                        print(f'Moved {file} to the recycle bin')
                    except Exception as e:
                        print(f'Failed to move {file} to the recycle bin:', str(e))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find and move txt files without an image with the same name to the recycle bin.')
    parser.add_argument('path', type=str, help='Path to the folder to be processed')
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print('The specified path is not a directory. Please provide a valid folder path.')
    else:
        find_and_move_txt_without_img(args.path)
