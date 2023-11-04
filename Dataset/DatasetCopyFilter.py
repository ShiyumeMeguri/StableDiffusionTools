import argparse
import shutil
import re
import os

def copy_files(input_file, dest_folder):
    # Updated regex pattern to match any .txt file path
    pattern = r"(.*\.txt)"

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(pattern, line)
            if match:
                # strip spaces from the beginning and the end of the file path
                filepath = match.group(1).strip()
                filename = os.path.basename(filepath)
                name, _ = os.path.splitext(filename)
                # If dest_folder argument is not given, set it as the parent directory name of the .txt file
                if dest_folder is None:
                    folder = os.path.basename(os.path.dirname(filepath))
                else:
                    folder = dest_folder
                if not os.path.exists(folder):
                    os.makedirs(folder)
                for ext in ['.txt', '.png', '.jpg']:
                    dest_path = os.path.join(folder, name+ext)
                    if filepath.replace('.txt', ext) != dest_path:  # check if source and destination are not the same
                        try:
                            shutil.copy(filepath.replace('.txt', ext), dest_path)
                        except FileNotFoundError:
                            continue

# Create argument parser and parse command line arguments
parser = argparse.ArgumentParser(description='Copy txt and image files to destination folder.')
parser.add_argument('input_file', type=str, help='输入包含路径的txt.')
parser.add_argument('dest_folder', type=str, nargs='?', default=None, help='Optional destination folder to copy files to.')

args = parser.parse_args()

copy_files(args.input_file, args.dest_folder)
