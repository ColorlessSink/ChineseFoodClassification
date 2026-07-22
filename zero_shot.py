'''
任务2：VLM 零样本分类基线 (Chinese-CLIP)
========================================
流程：
  1. 读取 dataset_20cls/classes.csv，拿到 20 个中文菜名
  2. 用单一 prompt 模板 "一张{菜名}的照片" 把菜名变成文本，过文本编码器得到 [20, D]
  3. 读取 test/ 下每张图片，过图像编码器得到 [N, D]
  4. 计算图片与 20 个文本的余弦相似度，取 argmax → Top-1；取前5 → Top-5
  5. 输出 Top-1 / Top-5 准确率，并保存逐图预测到 results/

注：本脚本只跑单一模板（按当前目标）。模板对比留待后续任务。
'''

import os, time
import torch
from PIL import Image
import pandas as pd
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# ----------------------------------------------------------------------
# 0. 配置
# ----------------------------------------------------------------------
ROOT      = "D:/study/University/Course/EagleLab/小作业/高扬"
DATA_DIR  = os.path.join(ROOT, "code", "dataset_20cls")
RESULT_DIR = os.path.join(ROOT, "code", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
# 本地缓存快照（避免每次联网校验导致超时）
_LOCAL_SNAP = r"C:\Users\Bill\.cache\huggingface\hub\models--OFA-Sys--chinese-clip-vit-base-patch16\snapshots\f4a64596bbcf9a2a94591b74b9dc39b2e4e77e3e"
import os as _os
if _os.path.isdir(_LOCAL_SNAP):
    MODEL_NAME = _LOCAL_SNAP
TEMPLATE   = "一张{c}的照片"     # 单一模板（本阶段不做对比）
SPLIT      = "test"             # 评估哪个 split (test / val)
BATCH_SIZE = 16

# 设备：当前环境 torch 是 CPU 版，强制用 CPU
import os as _os
_os.environ["HF_HUB_OFFLINE"] = "1"   # 完全离线，避免联网校验超时
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[环境] torch={torch.__version__}  device={device}")

# ----------------------------------------------------------------------
# 1. 加载模型
# ----------------------------------------------------------------------
print("[1/5] 加载 Chinese-CLIP 模型 ...")
model = ChineseCLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)
print(f"   模型路径: {MODEL_NAME}")

# ----------------------------------------------------------------------
# 2. 读取类别表，构造文本 prompt，编码文本特征
# ----------------------------------------------------------------------
print("[2/5] 读取类别表 & 编码文本特征 ...")
classes_df = pd.read_csv(os.path.join(DATA_DIR, "classes.csv"))
names_zh   = classes_df["zh"].tolist()
class_idx  = classes_df["idx"].tolist()   # 0..19

# 单一模板：把每个菜名套进模板
texts = [TEMPLATE.format(c=n) for n in names_zh]
print("   候选文本示例:")
for t in texts[:5]:
    print(f"     - {t}")

# 文本只需编码一次（与图片无关）。Chinese-CLIP 的 get_text_features 在本版本
# 只返回隐藏状态(未过投影)，需要手动取 [CLS] token 再过 text_projection。
with torch.no_grad():
    t_inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)
    text_out = model.get_text_features(**t_inputs)
    if hasattr(text_out, "text_embeds") and text_out.text_embeds is not None:
        text_feats = text_out.text_embeds
    else:
        # 旧版/异常返回：手动取 [CLS] (last_hidden_state[:,0,:]) 再投影
        cls = text_out.last_hidden_state[:, 0, :]
        text_feats = model.text_projection(cls)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # L2 归一化
print(f"   文本特征 shape: {text_feats.shape}")

# ----------------------------------------------------------------------
# 3. 收集测试集图片路径与真实标签
#    目录结构: test/<idx>_<name>/<img>.jpg  → 真实标签即文件夹名里的 idx
# ----------------------------------------------------------------------
print(f"[3/5] 收集 {SPLIT} 集图片 ...")
split_dir = os.path.join(DATA_DIR, SPLIT)
img_paths, labels = [], []
for folder in sorted(os.listdir(split_dir)):
    fpath = os.path.join(split_dir, folder)
    if not os.path.isdir(fpath):
        continue
    idx = int(folder.split("_")[0])     # 文件夹名 "07_西红柿炒蛋" → 7
    for fn in os.listdir(fpath):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            img_paths.append(os.path.join(fpath, fn))
            labels.append(idx)
print(f"   共 {len(img_paths)} 张测试图, {len(set(labels))} 类")

# ----------------------------------------------------------------------
# 4. 批量编码图片特征
# ----------------------------------------------------------------------
print(f"[4/5] 批量编码图片特征 (batch={BATCH_SIZE}) ...")
all_img_feats = []
t0 = time.time()
for i in range(0, len(img_paths), BATCH_SIZE):
    batch_paths = img_paths[i:i+BATCH_SIZE]
    imgs = [Image.open(p).convert("RGB") for p in batch_paths]
    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        img_out = model.get_image_features(**inputs)
        if hasattr(img_out, "image_embeds") and img_out.image_embeds is not None:
            feats = img_out.image_embeds
        else:
            cls = img_out.last_hidden_state[:, 0, :]
            feats = model.visual_projection(cls)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_img_feats.append(feats)
    if (i // BATCH_SIZE) % 5 == 0:
        print(f"   {i+len(batch_paths)}/{len(img_paths)}  用时 {time.time()-t0:.0f}s")
img_feats = torch.cat(all_img_feats, dim=0)               # [N, 512]
print(f"   图片特征 shape: {img_feats.shape}  总用时 {time.time()-t0:.0f}s")

# ----------------------------------------------------------------------
# 5. 计算相似度 → Top-1 / Top-5
# ----------------------------------------------------------------------
print("[5/5] 计算相似度 & 准确率 ...")
# logits = img_feats @ text_feats.T   shape [N, 20]
logits = 100.0 * img_feats @ text_feats.T     # 乘 100 是 CLIP 惯例的温度缩放，对 argmax 无影响
labels_t = torch.tensor(labels)

top1 = logits.argmax(dim=-1)
top1_acc = (top1 == labels_t).float().mean().item()

top5 = logits.topk(5, dim=-1).indices
top5_acc = sum(labels_t[i].item() in top5[i].tolist() for i in range(len(labels_t))) / len(labels_t)

print("\n========== 结果 ==========")
print(f"模板: {TEMPLATE}")
print(f"模型: {MODEL_NAME}")
print(f"评估集: {SPLIT} ({len(img_paths)} 张)")
print(f"Top-1 准确率: {top1_acc*100:.2f}%")
print(f"Top-5 准确率: {top5_acc*100:.2f}%")
print("==========================")

# ----------------------------------------------------------------------
# 6. 保存逐图预测，供后续混淆矩阵/失败案例分析
# ----------------------------------------------------------------------
pred_rows = []
for i in range(len(img_paths)):
    pred = top1[i].item()
    pred_rows.append({
        "path": os.path.relpath(img_paths[i], DATA_DIR),
        "true": labels[i],
        "pred": pred,
        "true_name": names_zh[labels[i]],
        "pred_name": names_zh[pred],
        "correct": int(labels[i] == pred),
    })
pd.DataFrame(pred_rows).to_csv(
    os.path.join(RESULT_DIR, f"zeroshot_{SPLIT}_preds.csv"), index=False, encoding="utf-8-sig")

summary = {
    "model": MODEL_NAME, "template": TEMPLATE, "split": SPLIT,
    "n_images": len(img_paths), "n_classes": len(names_zh),
    "top1": round(top1_acc*100, 2), "top5": round(top5_acc*100, 2),
}
import json
with open(os.path.join(RESULT_DIR, f"zeroshot_{SPLIT}_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n预测明细已保存: results/zeroshot_{SPLIT}_preds.csv")
print(f"结果摘要已保存: results/zeroshot_{SPLIT}_summary.json")
