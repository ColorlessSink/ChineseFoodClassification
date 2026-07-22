"""
任务1：中餐食物数据集构建脚本
---
从../ChineseFoodNet/release_data选取部分数据并按 训练:验证:测试 = 7:1:2 分割

输出文件目录结构：dataset_20cls/
  dataset_20cls/train/<class_idx>_<name>/<img>.jpg
  dataset_20cls/val/  ...
  dataset_20cls/test/  ...
  dataset_20cls/classes.csv       类别映射表 (idx, zh, en)
  dataset_20cls/<split>.csv       每个split的 (相对路径, 类别idx) 清单

"""
import os, random, shutil
from collections import defaultdict
import pandas as pd


SRC = os.path.join(os.path.dirname(__file__), "..", "ChineseFood Net 3", "release_data")     # 划分的数据来源于上级的ChineseFoodNet目录
SRC = os.path.normpath(SRC)
OUT = os.path.join(os.path.dirname(__file__), "dataset_20cls")

PER_CLASS_TRAIN, PER_CLASS_VAL, PER_CLASS_TEST = 70, 10, 20     # 7:1:2
SEED = 42
SEL = [
    (0,   "麻婆豆腐",   "Mapo Tofu"),
    (71,  "宫保鸡丁",   "Kung Pao Chicken"),
    (84,  "回锅肉",     "Double Cooked Pork"),
    (97,  "鱼香肉丝",   "Yu-Shiang Shredded Pork"),
    (112, "水煮鱼",     "Fish in Hot Chili Oil"),
    (11,  "鱼香茄子",   "Yu-Shiang Eggplant"),
    (5,   "酸辣土豆丝", "Hot and Sour Potato"),
    (50,  "西红柿炒蛋", "Tomato and Egg"),
    (9,   "地三鲜",     "Di San Xian"),
    (18,  "蚝油生菜",   "Oyster Sauce Lettuce"),
    (77,  "红烧肉",     "Braised Pork"),
    (58,  "糖醋排骨",   "Sweet and Sour Spareribs"),
    (83,  "梅菜扣肉",   "Pork with Salted Vegetable"),
    (92,  "京酱肉丝",   "Sweet Bean Pork"),
    (159, "饺子",       "Dumplings"),
    (145, "包子",       "Steamed Stuffed Bun"),
    (130, "扬州炒饭",   "Yangzhou Fried Rice"),
    (149, "炸酱面",     "Zhajiang Noodles"),
    (104, "葱爆羊肉",   "Scallion Lamb"),
    (118, "香辣小龙虾", "Spicy Crayfish"),
]


# ----------------------------------------------------------------------
# 1. 读取原始三个 list 文件
# ----------------------------------------------------------------------
def read_split(path, has_label=True):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img = parts[0]
            lab = int(parts[1]) if has_label else None
            pairs.append((img, lab))
    return pairs

print("读取原始 list 文件 ...")
train_pairs = read_split(os.path.join(SRC, "train_list.txt"))          # (000/000000.jpg, lab)
val_pairs   = read_split(os.path.join(SRC, "val_list.txt"))            # (000/000000.jpg, lab)
test_pairs  = read_split(os.path.join(SRC, "test_list.txt"), has_label=False)  # 只有文件名
test_truth  = read_split(os.path.join(SRC, "test_truth_list.txt"))     # (000000.jpg, lab)

# 把原始三份合并成 {原始类id: [所有图片相对路径]}
# train / val 的图片路径形如 "000/000000.jpg"，对应 train/<path>
# test 的图片是平铺的 "000000.jpg"，对应 test/<文件名>
by_class = defaultdict(list)  # 原始类id -> [(相对文件系统路径, 所在split)]

for img, lab in train_pairs:
    by_class[lab].append(("train/" + img, "train"))
for img, lab in val_pairs:
    by_class[lab].append(("train/" + img, "train"))   # val 也复用 train 目录(ChineseFoodNet 的 val/ 命名见下)
for img, lab in test_truth:
    by_class[lab].append(("test/" + img, "test"))

# 注意：ChineseFoodNet 的 val 图片实际放在 train/<类>/<图>.jpg 下
# （val_list 里的路径和 train_list 同构），test 图片单独放在 test/ 下平铺。

# ----------------------------------------------------------------------
# 2. 清空并创建输出目录
# ----------------------------------------------------------------------
if os.path.exists(OUT):
    shutil.rmtree(OUT)
for split in ("train", "val", "test"):
    os.makedirs(os.path.join(OUT, split), exist_ok=True)

# ----------------------------------------------------------------------
# 3. 按类抽样并复制
# ----------------------------------------------------------------------
random.seed(SEED)

records = []          # (split, rel_path, new_idx)
classes_rows = []     # (new_idx, zh, en, orig_id)

for new_idx, (orig_id, zh, en) in enumerate(SEL):
    all_imgs = by_class[orig_id]                       # [(fs_rel, split)]
    random.shuffle(all_imgs)
    # 优先取 test/ 里的图作为测试集（这些是真实“未见”图），其余从 train 里抽
    test_pool  = [x for x in all_imgs if x[1] == "test"]
    train_pool = [x for x in all_imgs if x[1] == "train"]

    n_train = PER_CLASS_TRAIN
    n_val   = PER_CLASS_VAL
    n_test  = PER_CLASS_TEST

    chosen = []  # (fs_rel, target_split)
    # 先填 test
    chosen += [(p, "test") for p, _ in test_pool[:n_test]]
    if len([c for c in chosen if c[1] == "test"]) < n_test:
        need = n_test - len([c for c in chosen if c[1] == "test"])
        chosen += [(p, "test") for p, _ in train_pool[:need]]
        train_pool = train_pool[need:]
    # 再填 train / val
    val_sel = train_pool[:n_val];   train_pool = train_pool[n_val:]
    train_sel = train_pool[:n_train]
    chosen += [(p, "val")   for p, _ in val_sel]
    chosen += [(p, "train") for p, _ in train_sel]

    # 复制文件到新结构: dataset_20cls/<split>/<idx>_<name>/<原文件名>
    folder_name = f"{new_idx:02d}_{zh}"
    for fs_rel, target_split in chosen:
        src_path = os.path.join(SRC, fs_rel)
        dst_dir  = os.path.join(OUT, target_split, folder_name)
        os.makedirs(dst_dir, exist_ok=True)
        # 用全局唯一文件名避免不同 split 同名覆盖
        fname = f"{new_idx:02d}_{os.path.basename(fs_rel)}"
        dst_path = os.path.join(dst_dir, fname)
        shutil.copyfile(src_path, dst_path)
        rel_out = os.path.join(target_split, folder_name, fname)
        records.append((target_split, rel_out, new_idx))

    classes_rows.append((new_idx, zh, en, orig_id))
    print(f"[{new_idx:02d}] {zh:<10} -> train {n_train} / val {n_val} / test {n_test}  (源类id={orig_id})")

# ----------------------------------------------------------------------
# 4. 写类别表 & 各 split 清单 csv
# ----------------------------------------------------------------------
pd.DataFrame(classes_rows, columns=["idx", "zh", "en", "orig_id"]).to_csv(
    os.path.join(OUT, "classes.csv"), index=False, encoding="utf-8-sig")

for split in ("train", "val", "test"):
    sub = [(s, p, i) for (s, p, i) in records if s == split]
    pd.DataFrame(sub, columns=["split", "path", "label"]).to_csv(
        os.path.join(OUT, f"{split}.csv"), index=False, encoding="utf-8-sig")

total = len(records)
print(f"\n完成。输出目录: {OUT}")
print(f"总图片数: {total}  (期望 {20*(n_train+n_val+n_test)})")
print(f"  train: {sum(1 for r in records if r[0]=='train')}")
print(f"  val:   {sum(1 for r in records if r[0]=='val')}")
print(f"  test:  {sum(1 for r in records if r[0]=='test')}")
