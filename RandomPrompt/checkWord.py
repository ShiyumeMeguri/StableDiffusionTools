import json


def load_vocab(filename):
    """
    从 JSON 文件中加载词汇表
    返回一个包含所有单词的列表
    """
    with open(filename) as f:
        vocab = json.load(f)
    words = [key[:-4] if key.endswith('</w>') else key for key in vocab.keys()]
    return words


def check_hairstyles(hairstyles, key):
    """
    检查 hairstyles 中的单词是否在 key 中存在
    如果单词包含空格，会将其分割为多个单词进行检查
    返回一个列表，其中包含所有存在于 key 中的单词
    """
    result = []
    for hairstyle in hairstyles:
        # 将单词分割为多个单词
        words = hairstyle.split()
        # 检查每个单词是否在 key 中存在
        for word in words:
            if word not in key:
                break
        else:
            result.append(hairstyle)
    return result


hairstyles = ['hairpin', 'hair band', 'hair tie', 'hair clip', 'headband', 'hair comb', 'hair brush', 'bobby pin', 'scrunchie', 'barrette', 'chignon pin', 'flower clip', 'feather clip', 'bow', 'ribbon', 'scarf', 'bandana', 'hair fork', 'hair sticks', 'hair spiral', 'hair beads', 'hair rings', 'hair chains', 'hair cuffs', 'hair bar', 'hair claw', 'hair stick pin', 'hair coil', 'hair vine', 'hair fork comb', 'hair fascinator', 'hair feathers', 'hair twist', 'hair wrap', 'hair pick', 'hair bun holder', 'hair elastic', 'hair snood', 'hair jewel', 'hair wreath', 'hair halo', 'hair wire', 'hair wrap scarf', 'hair tie cuff', 'hair braid', 'hair twisties', 'hair barrette clip', 'hair clip bow', 'hair clasp', 'hair ponytail holder', 'hair stick fork', 'hair bow tie', 'hair tie ribbon', 'hair twist tie', 'hair pins clips', 'hair jewelry chain', 'hair clip set', 'hair clip claw', 'hair clamps', 'hair clipper', 'hair comb set', 'hair accessory set', 'hair net', 'hair donut', 'hair foam', 'hair dryer brush', 'hair dryer diffuser', 'hair dryer holder', 'hair dryer nozzle', 'hair dryer organizer', 'hair dryer stand', 'hair dryer with comb', 'hair dryer with diffuser', 'hair dryer with nozzle', 'hair extension brush', 'hair extension clips', 'hair extension glue', 'hair extension kit', 'hair extension pliers', 'hair extension pull', 'hair extension remover', 'hair extension tape', 'hair extension thread', 'hair extension tools', 'hair extensions beads', 'hair extensions clip in', 'hair extensions for braids', 'hair extensions for men', 'hair extensions for short hair', 'hair extensions for women', 'hair extensions glue in', 'hair extensions human hair', 'hair extensions micro beads', 'hair extensions remy', 'hair extensions tape in', 'hair extensions wig', 'hair falls', 'hair filler powder', 'hair french twist', 'hair gems', 'hair glitter', 'hair grips', 'hair headpiece', 'hair highlights kit', 'hair hoop', 'hair hook', 'hair horn', 'hair hot rollers', 'hair iron', 'hair iron flat', 'hair iron holder', 'hair iron straightener', 'hair ironing board', 'hair jewels', 'hair jump rings', 'hair kanzashi', 'hair keratin glue', 'hair kit', 'hair lace', 'hair loc jewelry', 'hair lock beads', 'hair lock jewelry', 'hair locks', 'hair loop', 'hair marley twists', 'hair mask', 'hair metal beads', 'hair metal cuffs', 'hair metal rings', 'hair metal spirals', 'hair micro beads', 'hair micro rings', 'hair mousse', 'hair needle', 'hair net cap', 'hair ornament', 'hair painting brush', 'hair pin accessory', 'hair pin flower', 'hair pin jewelry', 'hair pin stick', 'hair plait', 'hair pleat', 'hair pomade', 'hair ponytail', 'hair ribbon clip', 'hair ring', 'hair roller', 'hair root powder', 'hair scrunchies']

# 加载词汇表
vocab_file = "VocabStrip.json"
key = load_vocab(vocab_file)

# 检查 hairstyles 中的单词是否在 key 中存在
result = check_hairstyles(hairstyles, key)

# 输出有效单词到文件
with open("output.txt", "w") as f:
    for word in result:
        f.write(word + "\n")

print("已将有效单词输出到 output.txt 文件中。")
