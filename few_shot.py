'''
任务3-1：Few-shot学习性能提升
---
流程：
  1. 读取 dataset_20cls/classes.csv，拿到 20 个中文菜名
  2. 每类从 train 选 K 张图片作为支持集
  3. 用图像编码器提取支持集特征
  4. 每类特征求平均，得到类别 prototype
  5. 测试图像与 prototype 计算预先相似度
  5. 输出 Top-1 / Top-5 准确率，与 baseline 对比并保存结果到 results/
'''

import os, time, json, random
import torch
from PIL import Image
import pandas as pd
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

from zero_shot import ZeroShot


ROOT = os.getcwd()     # 数据根路径默认为当前路径
DATA_DIR = os.path.join(ROOT, "dataset_20cls")
RESULT_DIR = os.path.join(ROOT, "results")

MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
_LOCAL_SNAP = r"C:\Users\Bill\.cache\huggingface\hub\models--OFA-Sys--chinese-clip-vit-base-patch16\snapshots\f4a64596bbcf9a2a94591b74b9dc39b2e4e77e3e"
MODEL_NAME = _LOCAL_SNAP if _LOCAL_SNAP else MODEL_NAME  # 本地缓存快照（避免每次联网校验导致超时）

SPLIT = "test"     # 评估哪个数据划分(test/val)
BATCH_SIZE = 16
K_SHOT = 10     # 每类菜抽取多少个支持集样本
SEED = 67     # 随机种子
BASELINE_TOP1 = 78.75
BASELINE_TOP5 = 92.75

device = "cuda"      # 在 gpu 上运行


class FewShot(ZeroShot):
    # 少样本学习主程序类，继承自ZeroShot
    def __init__(self, model, data_dir):
        super().__init__(model, data_dir)

    def sample_support_set(self, k, seed):
        random.seed(seed)

        support_path = []
        support_labels = []
        food_dir = sorted(os.listdir(os.path.join(self.data_dir, "train")))
        for i in range(len(food_dir)):
            selected = random.sample(sorted(os.listdir(os.path.join(self.data_dir, "train", food_dir[i]))), k)
            for l in selected:
                support_path.append(os.path.join(self.data_dir, "train", food_dir[i], l))
                support_labels.append(i)
        return support_path, support_labels

    def build_prototypes(self, paths, labels, batch_size):
        # 编码 few-shot 支持集图片
        support_feats = self.encode_pic(paths, batch_size)
        support_labels_t = torch.tensor(labels).to(device)

        prototypes = []
        for i in range(len(self.names_zh)):
            class_feats = support_feats[support_labels_t == i]
            proto = class_feats.mean(dim=0)
            proto = proto / proto.norm(dim=-1, keepdim=True)      # 归一化
            prototypes.append(proto)

        prototypes = torch.stack(prototypes, dim=0)
        return prototypes

    def calc_sim_prototypes(self, prototypes, pic_feats, output=''):
        # 计算测试图像与 prototype 的相似度
        logits = 100.0 * pic_feats @ prototypes.T

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
                output, index=False, encoding="utf-8-sig")

        return logits



if __name__ == "__main__":
    print("1. 加载 Chinese-CLIP 模型")
    mod = FewShot(MODEL_NAME, DATA_DIR)

    print(f"2. 随机选取 {K_SHOT} 张支持集图片：")
    print(f"    随机种子：{SEED}")
    support_paths, support_labels = mod.sample_support_set(K_SHOT, SEED)
    for i in range(len(support_labels)):
        print("     "+mod.names_zh[support_labels[i]]+": "+support_paths[i])

    print("3. 编码 few-shot 支持集图片：")
    prototypes = mod.build_prototypes(support_paths, support_labels, BATCH_SIZE)
    print(f"   prototype 特征 shape: {prototypes.shape}")

    print(f"4. 计算 {SPLIT} 集图像与 prototype 的相似度：")
    test_paths, test_labels = mod.gather_pic(SPLIT)
    test_feats = mod.encode_pic(test_paths, BATCH_SIZE)
    logits = mod.calc_sim_prototypes(prototypes, test_feats, os.path.join(RESULT_DIR, "fewshot_"+SPLIT+"_preds.csv"))
    top1_acc = mod.calc_accuracy(logits, test_labels, 1)
    top5_acc = mod.calc_accuracy(logits, test_labels, 5)

    print("\n========== 结果 ==========")
    print(f"模型: {MODEL_NAME}")
    print(f"评估集: {SPLIT} ({len(mod.img_paths)} 张)\n")
    print(f"Baseline Top-1 准确率: {BASELINE_TOP1}%")
    print(f"Baseline Top-5 准确率: {BASELINE_TOP5}%\n")
    print(f"Top-1 准确率: {top1_acc*100:.2f}%")
    print(f"Top-5 准确率: {top5_acc*100:.2f}%\n")
    print("==========================")

    summary = {
        "model": MODEL_NAME, "split": SPLIT,
        "n_images": len(mod.img_paths), "n_classes": len(mod.names_zh),
        "k_shot": K_SHOT, "top1": round(top1_acc*100, 2), "top5": round(top5_acc*100, 2)
    }
    with open(os.path.join(RESULT_DIR, f"fewshot_{SPLIT}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n预测明细已保存: results/fewshot_{SPLIT}_preds.csv")
    print(f"结果摘要已保存: results/fewshot_{SPLIT}_summary.json")
