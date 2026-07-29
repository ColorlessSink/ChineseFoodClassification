import os, json, random
import numpy as np
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from torch.utils.data import Dataset, DataLoader
import cv2
from lora_train import *


class FoodAugment(object):
    def __init__(self, p=0.5):
        self.p = p    # 进行增强处理的概率
    
    def __call__(self, pic):
        # pic传入PIL的Image对象
        pic_bgr = cv2.cvtColor(np.array(pic), cv2.COLOR_RGB2BGR)

        if random.random() < self.p:
            pic_bgr = self.rotate(pic_bgr)
        if random.random() < self.p:
            pic_bgr = self.crop(pic_bgr)
        if random.random() < self.p:
            pic_bgr = self.jitter(pic_bgr)

        pic_bgr = cv2.cvtColor(pic_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(pic_bgr)
    
    def rotate(self, pic, max_angle=15):
        # 随机旋转，pic传入BGR格式的numpy数组
        ang = random.uniform(-max_angle, max_angle)
        mat = cv2.getRotationMatrix2D((pic.shape[1]/2, pic.shape[0]/2), ang, 1.0)
        rotated = cv2.warpAffine(pic, mat, (pic.shape[1], pic.shape[0]))
        return rotated
    
    def crop(self, pic, min_ratio=0.8):
        # 随机裁剪，pic传入BGR格式的numpy数组
        ratio = random.uniform(min_ratio, 1.0)
        new_h, new_w = int(pic.shape[0] * ratio), int(pic.shape[1] * ratio)
        # 裁切左上角
        top = random.randint(0, pic.shape[0] - new_h)
        left = random.randint(0, pic.shape[1] - new_w)
        cropped = pic[top:top+new_h, left:left+new_w]
        cropped = cv2.resize(cropped, (pic.shape[0], pic.shape[1]))     # resize 回原尺寸
        return cropped
    
    def jitter(self, pic, jit=10):
        # 随机颜色抖动，pic传入BGR格式的numpy数组
        hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)     # 转 HSV
        h, s, v = cv2.split(hsv)

        h = np.clip(h.astype(np.int16) + random.randint(-jit, jit), 0, 179).astype(np.uint8)
        s = np.clip(s.astype(np.int16) + random.randint(-jit, jit), 0, 255).astype(np.uint8)
        v = np.clip(v.astype(np.int16) + random.randint(-jit, jit), 0, 255).astype(np.uint8)

        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



if __name__ == "__main__":
    # 准备好processor和dataset
    processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)
    dataset = FoodDataset(DATA_DIR, split=SPLIT, transform=FoodAugment())

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, 
                            collate_fn=lambda BATCH: collate_fn(BATCH, proc=processor), num_workers=0)

    # 加载模型，加载过程中曾出现内存不足的报错，所以加上了low_cpu_mem_usage
    model = ChineseCLIPModel.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True).to(device)
    
    loraTrain(model, processor, dataloader, RESULT_DIR, "lora_adapter_aug")

