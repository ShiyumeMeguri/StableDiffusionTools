import os
import argparse

def generate_bat_files(path, weight):
    commands = []
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if name.endswith(".ckpt") or name.endswith(".safetensors"):
                if path in name:
                    # Generate the command to be written in the bat file
                    cmd = f"LoRA_Pruneder.py {name} {name[:-5]}_Pruned {weight}"
                    commands.append(cmd)
                    print(f"{name[:-5]} command generated!")
    # Write all commands to a bat file
    with open("run_pruning.bat", "w") as bat_file:
        bat_file.write("\n".join(commands))
    print("run_pruning.bat generated!")

def main(args):
    # Call the function with the search string specified in the command line argument
    generate_bat_files(args.path, args.weight)

if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("weight", help="Lora权重")
    parser.add_argument("-p", "--path", default=".", help="path to the directory containing the model files")
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
