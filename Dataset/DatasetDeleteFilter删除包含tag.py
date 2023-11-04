import os
import argparse
import send2trash
import glob

def check_tags(file_path, tags):
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.read()
        # Split the tags by commas, and check if any are in the file
        return any(tag in contents for tag in tags.split(','))

def handle_files_in_dir(path, tags):
    for txt_file in glob.glob(os.path.join(path, '*.txt')):
        if check_tags(txt_file, tags):  # 文件中存在任何一个tag，就删除
            base = os.path.splitext(txt_file)[0]
            for ext in ['.jpg', '.png']:
                img_file = base + ext
                if os.path.exists(img_file):
                    send2trash.send2trash(img_file)
                    print(f"已移除{img_file}")
            send2trash.send2trash(txt_file)
            print(f"已移除{txt_file}")

def handle_directory(directory, tags):
    has_subdir = False
    for root, dirs, files in os.walk(directory):
        if 'img' in dirs:
            has_subdir = True
            img_path = os.path.join(root, 'img')
            handle_files_in_dir(img_path, tags)
    if not has_subdir:  # 当前目录没有子目录
        handle_files_in_dir(directory, tags)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('directory', type=str, help='A directory to process')
    parser.add_argument('tags', type=str, help='A comma separated list of tags')

    args = parser.parse_args()

    handle_directory(args.directory, args.tags)
