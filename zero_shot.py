'''
任务2：VLM 零样本分类基线 (选取的VLM是Chinese-CLIP)
---
流程：
  1. 读取 dataset_20cls/classes.csv，拿到 20 个中文菜名
  2. 用单一 prompt 模板 "一张{菜名}的照片" 把菜名变成文本，过文本编码器得到 [20, D]
  3. 读取 test/ 下每张图片，过图像编码器得到 [N, D]
  4. 计算图片与 20 个文本的余弦相似度，取 argmax → Top-1；取前5 → Top-5
  5. 输出 Top-1 / Top-5 准确率，并保存逐图预测到 results/

注：本脚本暂时只跑了单一模板
'''

import os, time, json
import torch
from PIL import Image
import pandas as pd
from transformers import ChineseCLIPProcessor, ChineseCLIPModel


ROOT = os.getcwd()     # 数据根路径默认为当前路径
DATA_DIR = os.path.join(ROOT, "dataset_20cls")
RESULT_DIR = os.path.join(ROOT, "results")

MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
_LOCAL_SNAP = r"C:\Users\Bill\.cache\huggingface\hub\models--OFA-Sys--chinese-clip-vit-base-patch16\snapshots\f4a64596bbcf9a2a94591b74b9dc39b2e4e77e3e"
MODEL_NAME = _LOCAL_SNAP if _LOCAL_SNAP else MODEL_NAME  # 本地缓存快照（避免每次联网校验导致超时）

TEMPLATE = ["一张{c}的照片", "{c}", "一张盘子上有{c}的照片","一张近距离拍摄{c}的照片","一张中餐菜品{c}的照片","一张盘子上摆着中餐菜品{c}的照片"]     # 模板
SPLIT = "test"     # 评估哪个数据划分(test/val)
BATCH_SIZE = 16

device = "cuda"      # 在 gpu 上运行


class Main(object):
    # 主程序类
    def __init__(self, model, data_dir):
        # 加载模型
        self.model = ChineseCLIPModel.from_pretrained(model).to(device)
        self.model.eval()
        self.processor = ChineseCLIPProcessor.from_pretrained(model)

        # 加载类别表
        self.data_dir = data_dir
        self.classes_df = pd.read_csv(os.path.join(self.data_dir, "classes.csv"))
        self.names_zh = self.classes_df["zh"].tolist()
        self.class_idx = self.classes_df["idx"].tolist()

    def generate_prompts(self, template):
        # 将菜名套进模版
        return [template.format(c=n) for n in self.names_zh]

    def encode_texts(self, texts):
        # 编码文本特征，返回文本向量
        with torch.no_grad():
            t_inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(device)
            text_out = self.model.get_text_features(**t_inputs)

            if hasattr(text_out, "text_embeds") and text_out.text_embeds is not None:
                text_feats = text_out.text_embeds
            else:
                # 旧版/异常返回：手动取 [CLS] (last_hidden_state[:,0,:]) 再投影
                cls = text_out.last_hidden_state[:, 0, :]
                text_feats = self.model.text_projection(cls)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # 归一化

            return text_feats

    def encode_pic(self, split, batch_size):
        # 收集测试集图片路径
        self.split = split
        split_dir = os.path.join(self.data_dir, split)
        self.img_paths, self.labels = [], []
        for i in sorted(os.listdir(split_dir)):
            fpath = os.path.join(split_dir, i)
            if not os.path.isdir(fpath):
                continue
            index = int(i.split("_")[0])
            for j in os.listdir(fpath):
                if j.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.img_paths.append(os.path.join(fpath, j))
                    self.labels.append(index)

        # 批量编码图片特征
        all_img_feats = []
        self.t0 = time.time()
        for i in range(0, len(self.img_paths), batch_size):
            batch_paths = self.img_paths[i:i+batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            
            with torch.no_grad():
                inputs = self.processor(images=imgs, return_tensors="pt").to(device)
                img_out = self.model.get_image_features(**inputs)
                
                if hasattr(img_out, "image_embeds") and img_out.image_embeds is not None:
                    feats = img_out.image_embeds
                else:
                    cls = img_out.last_hidden_state[:, 0, :]
                    feats = self.model.visual_projection(cls)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_img_feats.append(feats)
            if (i // BATCH_SIZE) % 5 == 0:
                print(f"   {i+len(batch_paths)}/{len(self.img_paths)}  用时 {time.time()-self.t0:.0f}s")
        
        img_feats = torch.cat(all_img_feats, dim=0)     # [N, 512]
        return img_feats

    def calc_simularity(self, text_feats, pic_feats, output=''):
        # 计算相似度，output为预测明细csv文件保存路径（留空为不保存）
        logits = 100.0 * pic_feats @ text_feats.T     # 乘100是CLIP惯例的温度缩放，对argmax无影响

        if output != '':
            pred_rows = []
            for i in range(len(self.img_paths)):
                pred = int(logits[i].argmax())
                pred_rows.append({
                    "path": os.path.relpath(self.img_paths[i], DATA_DIR),
                    "true": self.labels[i],
                    "pred": pred,
                    "true_name": self.names_zh[self.labels[i]],
                    "pred_name": self.names_zh[pred],
                    "correct": int(self.labels[i] == pred),
                })
            pd.DataFrame(pred_rows).to_csv(
                os.path.join(output, f"zeroshot_{self.split}_preds.csv"), index=False, encoding="utf-8-sig")

        return logits

    def calc_accuracy(self, logits, ans, topk):
        # 计算准确率，logits（相似度）为pt类型，ans（答案）是普通数组类型
        labels_t = torch.tensor(ans).to(device)

        top = logits.topk(topk, dim=-1).indices
        top_acc = sum(labels_t[i].item() in top[i].tolist() for i in range(len(labels_t))) / len(labels_t)
        return top_acc



if __name__ == "__main__":
    print(f"环境： torch={torch.__version__}  device={device}")
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("1. 加载 Chinese-CLIP 模型：")
    mod = Main(MODEL_NAME, DATA_DIR)
    print(f"   模型路径: {mod.model}")

    texts = []
    text_feats = []
    print("2. 编码文本特征：")
    for i in range(len(TEMPLATE)):
        texts.append(mod.generate_prompts(TEMPLATE[i]))
        print("提示模版%d： ", i+1)
        print("   候选文本示例:")
        for t in texts[i][:5]:
            print(f"     - {t}")
        text_feats.append(mod.encode_texts(texts[i]))
        print(f"   文本特征 shape: {text_feats[i].shape}")

    print(f"3. 批量编码图片特征 (batch={BATCH_SIZE})：")
    pic_feats = mod.encode_pic(SPLIT, BATCH_SIZE)
    print(f"   共 {len(mod.img_paths)} 张测试图, {len(set(mod.labels))} 类")
    print(f"   图片特征 shape: {pic_feats.shape}  总用时 {time.time()-mod.t0:.0f}s")

    print(f"4. 计算相似度")
    simularity = []
    for i in range(len(TEMPLATE)):
        simularity.append(mod.calc_simularity(text_feats[i], pic_feats, RESULT_DIR))
    
    print(f"5. 计算准确率")

    top1_acc = []
    top5_acc = []
    for i in range(len(TEMPLATE)):
        top1_acc.append(mod.calc_accuracy(simularity[i], mod.labels, 1))
        top5_acc.append(mod.calc_accuracy(simularity[i], mod.labels, 5))
    
    print("\n========== 结果 ==========")
    print(f"模型: {MODEL_NAME}")
    print(f"评估集: {SPLIT} ({len(mod.img_paths)} 张)\n")
    for i in range(len(TEMPLATE)):
        print(f"模板: {TEMPLATE[i]}")
        print(f"Top-1 准确率: {top1_acc[i]*100:.2f}%")
        print(f"Top-5 准确率: {top5_acc[i]*100:.2f}%\n")
    print("==========================")

    summary = {
        "model": MODEL_NAME, "template": TEMPLATE[0], "split": SPLIT,
        "n_images": len(mod.img_paths), "n_classes": len(mod.names_zh),
        "top1": round(top1_acc[0]*100, 2), "top5": round(top5_acc[0]*100, 2),
    }
    with open(os.path.join(RESULT_DIR, f"zeroshot_{SPLIT}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n预测明细已保存: results/zeroshot_{SPLIT}_preds.csv")
    print(f"结果摘要已保存: results/zeroshot_{SPLIT}_summary.json")
