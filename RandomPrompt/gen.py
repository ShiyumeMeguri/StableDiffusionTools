import json
import re
import random
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Generate prompt from JSON vocab')
parser.add_argument('-count', type=int, default=100, help='number of prompts to generate')
parser.add_argument('-l', type=int, default=10, help='length of each prompt')
args = parser.parse_args()

# 打开JSON文件
with open('VocabStrip.json') as f:
    vocab = json.load(f)

# 用于匹配单词的正则表达式
word_pattern = re.compile(r'^[a-zA-Z0-9]{3,}$')

# 筛选出单词key
word_keys = [key[:-4] if key.endswith('</w>') else key for key in vocab.keys() if word_pattern.match(key)]

# 发型单词列表
hairstyles = ['french braid pigtails','a-line bob','afro','angled layers','angled lob','asymmetric cut','asymmetrical bangs','bald','banana clip','bangs','bantu knots','beach waves','beachy braids','beachy layers','beachy waves','beveled bangs','big barrel curls','blunt bangs','bob haircut','boho braids','braid crown','braid out','braided bangs','braided bun','braided crown','braided headband','braided low bun','braided messy bun','braided pigtails','braided ponytail','braided space buns','braided top knot','braided updo','braids','brush up','bubble ponytail','bun','buzz cut','chinese staircase braid','choppy bangs','choppy layers','comb over','corkscrew curls','corset braid','crew cut','crochet braids','crossover ponytail','crown braid updo','crown braid','crown twist','crown','curled lob','curly afro','curly bangs','curly bob','curly hair','curly layers','curtain bangs','double braids','dutch braid','dutch crown braid','face-framing layers','fade','faded undercut','faux hawk','feathered bangs','feathered hair','feathered layers','finger coils','finger waves','fishtail braid','fishtail crown braid','five-strand braid','flat twist out','flower crown','four-strand braid','french braid','french bun','french twist','fringe bangs','fringe','gibson tuck','goddess braid','graduated bob','greek goddess updo','hair bow','hair buns','hair clip','hair tie','half-up bun','half-up ponytail','headband','high braided ponytail','high bun','high ponytail','hime cut','hollywood waves','inside-out braid','inverted bob','inverted curly bob','ivy league','knotted ponytail','lace braid','layered bangs','layered hair','layered shag','lob','long bangs','long hair','loose waves','low braided ponytail','low chignon','low ponytail','low rolled bun','low twist bun','mermaid braid','messy braid','messy bun updo','messy bun','messy chignon','messy ponytail','messy top knot','messy updo','micro bangs','micro braids','middle-parted bob','mohawk','pin curls','pineapple updo','pixie bob','pixie cut','ponytail','princess curls','razored bangs','razored layers','rod set','rolled updo','rope braid','s waves','scrunched ponytail','shag cut','shaggy bangs','shaggy layers','short bangs','short hair','shoulder-length curls','side ponytail','side swept curls','side-parted bob','six-strand braid','sleek bob','sleek layers','sleek low bun','sleek ponytail','slicked back','slicked-back bob','slicked-back hair','slicked-back ponytail','snake braid','space buns','spiky bangs','spiky hair','spiral braid','spiral curls','spiral ponytail','straight hair','straight-across bangs','swoop bangs','textured bangs','textured layers','tiara','top knot bun','top knot','twist and curl','twist braid','twist out','twisted braids','twisted bun','twisted chignon','twisted headband','twisted pigtails','twisted updo','vintage curls','voluminous blowout','voluminous curls','voluminous layers','wash and go','waterfall braid','waterfall twist','wavy hair','wavy lob','wispy bangs','wispy layers','wrap-around braided bun','wrapped ponytail']

# 发饰单词列表
accessories = ['hair band','hair tie','hair clip','headband','hair comb','hair brush','bobby pin','flower clip','feather clip','bow','ribbon','scarf','bandana','hair fork','hair sticks','hair spiral','hair beads','hair rings','hair chains','hair cuffs','hair bar','hair claw','hair stick pin','hair coil','hair vine','hair fork comb','hair feathers','hair twist','hair wrap','hair pick','hair bun holder','hair elastic','hair jewel','hair wreath','hair halo','hair wire','hair wrap scarf','hair tie cuff','hair braid','hair clip bow','hair ponytail holder','hair stick fork','hair bow tie','hair tie ribbon','hair twist tie','hair pins clips','hair jewelry chain','hair clip set','hair clip claw','hair clipper','hair comb set','hair accessory set','hair net','hair donut','hair foam','hair dryer brush','hair dryer diffuser','hair dryer holder','hair dryer organizer','hair dryer stand','hair dryer with comb','hair dryer with diffuser','hair extension brush','hair extension clips','hair extension glue','hair extension kit','hair extension pull','hair extension remover','hair extension tape','hair extension thread','hair extension tools','hair extensions beads','hair extensions clip in','hair extensions for braids','hair extensions for men','hair extensions for short hair','hair extensions for women','hair extensions glue in','hair extensions human hair','hair extensions micro beads','hair extensions remy','hair extensions tape in','hair extensions wig','hair falls','hair filler powder','hair french twist','hair gems','hair glitter','hair grips','hair highlights kit','hair hoop','hair hook','hair horn','hair hot rollers','hair iron','hair iron flat','hair iron holder','hair ironing board','hair jewels','hair jump rings','hair kit','hair lace','hair loc jewelry','hair lock beads','hair lock jewelry','hair locks','hair loop','hair marley twists','hair mask','hair metal beads','hair metal cuffs','hair metal rings','hair micro beads','hair micro rings','hair mousse','hair needle','hair net cap','hair ornament','hair painting brush','hair pin accessory','hair pin flower','hair pin jewelry','hair pin stick','hair ponytail','hair ribbon clip','hair ring','hair roller','hair root powder']

# 生成prompt
count = args.count
length = args.l
prompts = []
for i in range(count):
    # 随机选择一个位置放置cute
    cute_index = random.randint(0, length - 1)
    # 随机选择发型单词并插入到单词列表中
    hairstyle = random.choice(hairstyles)
    accessorie = random.choice(accessories)
    compatible_hairstyles = [hairstyle, accessorie, 'loli', '1girl', 'cute']
    words = random.sample(list(set(word_keys) - set(compatible_hairstyles)), length - 4)
    words.insert(cute_index, 'cute')
    # 随机选择一个位置放置loli或1girl
    loli_index = random.randint(0, length - 3)
    words.insert(loli_index, random.choice(['loli', '1girl']))
    # 插入发型单词
    hairstyle_index = random.randint(0, length - 2)
    words.insert(hairstyle_index, hairstyle)
    words.insert(hairstyle_index, accessorie)
    prompt = ', '.join(words) + ', '
    prompts.append(prompt)

# 输出prompt
for prompt in prompts:
    print(prompt)
