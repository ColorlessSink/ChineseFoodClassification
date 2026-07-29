import os, time, json, random
from PIL import Image
import pandas as pd
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from torch.utils.data import Dataset, DataLoader
import peft as pf
import torch

from zero_shot import ZeroShot


TEMPLATE = "一张{c}的照片"     # 模版
ROOT = os.getcwd()     # 数据根路径默认为当前路径
DATA_DIR = os.path.join(ROOT, "dataset_20cls")
RESULT_DIR = os.path.join(ROOT, "results")
SPLIT = "train"     # 训练数据集

EPOCH = 10  
BATCH_SIZE = 16

MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
_LOCAL_SNAP = r"C:\Users\Bill\.cache\huggingface\hub\models--OFA-Sys--chinese-clip-vit-base-patch16\snapshots\f4a64596bbcf9a2a94591b74b9dc39b2e4e77e3e"
MODEL_NAME = _LOCAL_SNAP if _LOCAL_SNAP else MODEL_NAME  # 本地缓存快照（避免每次联网校验导致超时）

device = "cuda"

def collate_fn(BATCH, proc):
    # 传入dataloader的自定义collate_fn函数
    pics = [i[0] for i in BATCH]
    texts = [i[1] for i in BATCH]

    labels = torch.tensor([i[2] for i in BATCH], dtype=torch.long)
    procs = proc(text=texts, images=pics, padding=True, return_tensors='pt')
    return procs, labels


class FoodDataset(Dataset):
    def __init__(self, data_dir, split, transform):
        super().__init__()
        # 加载类别表
        self.data_dir = data_dir
        self.classes_df = pd.read_csv(os.path.join(self.data_dir, "classes.csv"))
        self.names_zh = self.classes_df["zh"].tolist()
        self.class_idx = self.classes_df["idx"].tolist()

        # 将菜名套进模版
        self.texts = []
        for i in self.names_zh:
            self.texts.append(TEMPLATE.format(c=i))

        # 遍历图片路径，格式化存进samples
        self.samples = []
        for i in os.listdir(os.path.join(self.data_dir, split)):
            fpath = os.path.join(self.data_dir, split, i)
            if os.path.isdir(fpath):
                for j in os.listdir(fpath):
                    if j.lower().endswith((".jpg", ".jpeg", ".png")):
                        picpath = os.path.join(fpath, j)
                        label = int(j.split('_')[0])
                        text = self.texts[label]
                        self.samples.append((picpath, text, label))

        self.transform = transform      # 传入方法，用于图像增强

    def __len__(self):
        # Dataset需要此方法，计数
        return len(self.samples)
    
    def __getitem__(self, i):
        # Dataset需要此方法，取数据
        pic = Image.open(self.samples[i][0]).convert('RGB')
        if self.transform is not None:
            pic = self.transform(pic)
        return pic, self.samples[i][1], self.samples[i][2]


class LoRA(ZeroShot):
    # 供Lora调参过后的模型进行分类，继承自ZeroShot
    def __init__(self, model, processor, data_dir):
        # 加载模型
        self.model = model
        self.processor = processor
        self.model.eval()

        # 加载类别表
        self.data_dir = data_dir
        self.classes_df = pd.read_csv(os.path.join(self.data_dir, "classes.csv"))
        self.names_zh = self.classes_df["zh"].tolist()
        self.class_idx = self.classes_df["idx"].tolist()


if __name__ == "__main__":
    # 准备好processor和dataset
    processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)
    dataset = FoodDataset(DATA_DIR, split=SPLIT, transform=None)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, 
                            collate_fn=lambda BATCH: collate_fn(BATCH, proc=processor), num_workers=0)

    # 加载模型，加载过程中曾出现内存不足的报错，所以加上了low_cpu_mem_usage
    model = ChineseCLIPModel.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True).to(device)
    model.requires_grad_(False)     # 冻结参数

    config = pf.LoraConfig(r=8,      # 低秩矩阵的秩
                           lora_alpha=16,      # 缩放系数
                           lora_dropout=0.05,      # lora层的dropout
                           target_modules=["query", "value", "q_proj", "v_proj"])     # ChineseCLIP里文本塔和视觉塔的Q、V层名称
    model = pf.get_peft_model(model, config)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4, weight_decay=0.01)

    # 进行epoch轮训练
    best_acc = 0
    pre_loss = []
    pre_acc = []
    for i in range(EPOCH):
        model.train()
        epoch_loss = 0
        for (inputs, label) in dataloader:
            output = model(**{k: v.to(device) for k, v in inputs.items()})
  
            img_feats = output.image_embeds     # 文本描述向量
            txt_feats = output.text_embeds     # 图片描述向量
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)     # 归一化
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits = logit_scale * (img_feats @ txt_feats.T)

            labels = torch.arange(logits.shape[0]).to(device)
            
            # 分别按图和文本算双向loss
            loss_img = torch.nn.functional.cross_entropy(logits, labels)
            loss_txt = torch.nn.functional.cross_entropy(logits.T, labels)
            loss = (loss_img + loss_txt)/2

            optimizer.zero_grad()
            loss.backward()     # 反向传播
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        pre_loss.append(avg_loss)

        # 计算val集准确率
        test_m = LoRA(model, processor, DATA_DIR)
        text_feats = test_m.encode_texts(test_m.generate_prompts(TEMPLATE))
        pic_feats = test_m.encode_pic(test_m.gather_pic("val")[0], BATCH_SIZE)
        val_sim = test_m.calc_simularity(text_feats, pic_feats)
        val_acc = test_m.calc_accuracy(val_sim, test_m.labels, 1)
        pre_acc.append(val_acc)
        print(f"Epoch {i+1}: train_loss={avg_loss:.4f}, val_acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            # 更新并保存
            best_acc = val_acc
            model.save_pretrained(os.path.join(RESULT_DIR, "lora_adapter")) 
            print("新的最佳已保存")



