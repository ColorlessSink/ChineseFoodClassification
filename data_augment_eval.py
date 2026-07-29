import os, json
import pandas as pd
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import peft as pf
import torch
from lora_train import LoRA

ROOT = os.getcwd()     # 数据根路径默认为当前路径
DATA_DIR = os.path.join(ROOT, "dataset_20cls")
RESULT_DIR = os.path.join(ROOT, "results")
ADAPTER_DIR = os.path.join(RESULT_DIR, "lora_adapter_aug")

MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
_LOCAL_SNAP = r"C:\Users\Bill\.cache\huggingface\hub\models--OFA-Sys--chinese-clip-vit-base-patch16\snapshots\f4a64596bbcf9a2a94591b74b9dc39b2e4e77e3e"
MODEL_NAME = _LOCAL_SNAP if _LOCAL_SNAP else MODEL_NAME  # 本地缓存快照（避免每次联网校验导致超时）

TEMPLATE = "一张{c}的照片"     # 模板
SPLIT = "test"     # 评估哪个数据划分(test/val)
BATCH_SIZE = 16
BASELINE_TOP1 = 92.5
BASELINE_TOP5 = 99.25

device = "cuda"      # 在 gpu 上运行

if __name__ == "__main__":
    print(f"环境： torch={torch.__version__}  device={device}")
    print("1. 加载 base 模型和 adapter：")
    base = ChineseCLIPModel.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True).to(device)
    model = pf.PeftModel.from_pretrained(base, ADAPTER_DIR)
    processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)
    print(f"    模型路径:{base}\n   adapter 路径：{ADAPTER_DIR}")

    t_eval = LoRA(model, processor, DATA_DIR)
    print(f"2. 计算相似度")
    text_feats = t_eval.encode_texts(t_eval. generate_prompts(TEMPLATE))
    pic_feats = t_eval.encode_pic(t_eval.gather_pic(SPLIT)[0], BATCH_SIZE)
    simularity = t_eval.calc_simularity(text_feats, pic_feats, os.path.join(RESULT_DIR, "lora_aug_"+SPLIT+"_preds.csv"))
    
    print(f"3. 计算准确率")
    top1_acc = t_eval.calc_accuracy(simularity, t_eval.labels, 1)
    top5_acc = t_eval.calc_accuracy(simularity, t_eval.labels, 5)
    
    print("\n==========结果==========")
    print(f"模型: {MODEL_NAME}")
    print(f"评估集: {SPLIT} ({len(t_eval.img_paths)} 张)")
    print(f"模板: {TEMPLATE}\n")

    print("Baseline:")
    print(f"Top-1 准确率: {BASELINE_TOP1}%")
    print(f"Top-5 准确率: {BASELINE_TOP5}%\n")

    print(f"Adapter 路径：{ADAPTER_DIR}")
    print(f"Top-1 准确率: {top1_acc*100:.2f}%")
    print(f"Top-5 准确率: {top5_acc*100:.2f}%\n")
    print("==========================")

    summary = {
        "model": MODEL_NAME, "method": "LoRA r=8 alpha=16", "split": SPLIT,
        "n_images": len(t_eval.img_paths), "n_classes": len(t_eval.names_zh),
        "results": {"template": TEMPLATE, "top1": round(top1_acc*100, 2), "top5": round(top5_acc*100, 2)}
    }
    with open(os.path.join(RESULT_DIR, f"lora_aug_{SPLIT}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n预测明细已保存: results/lora_aug_{SPLIT}_preds.csv")
    print(f"结果摘要已保存: results/lora_aug_{SPLIT}_summary.json")

    