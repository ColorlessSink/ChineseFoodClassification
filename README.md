# [中餐食物分类（Chinese-CLIP）](https://github.com/ColorlessSink/ChineseFoodClassification)

## 作业概述

基于视觉语言模型的中餐食物零样本 / 少样本 / LoRA 微调分类实验。从 ChineseFoodNet 中抽取 20 类中餐菜品，用 Chinese-CLIP 实现零样本分类基线，并通过 Few-shot prototype、LoRA 微调、OpenCV 数据增强三种方式提升分类性能，最后对结果进行系统分析。

## 结果汇总

test 集 400 张、20 类的最终结果：

| 方法 | 改权重 | Top-1 | Top-5 | 对比零样本基线 |
|---|---|---|---|---|
| 零样本基线（`一张{c}的照片`） | 否 | 78.75% | 92.75% | — |
| Few-shot 10-shot prototype | 否 | 90.50% | 98.50% | +11.75 / +5.75 |
| **LoRA 微调 (r=8)** | **是 (0.31%)** | **92.50%** | **99.25%** | **+13.75 / +6.50** |
| 数据增强版 LoRA | 是 | 91.50% | 99.25% | −1.0 / 0 |

## 作业环境

- Python 3.13
- torch（CUDA 版，本机为 RTX 4050 Laptop 6GB 显存）
- transformers、peft、accelerate、datasets
- opencv-python、scikit-learn、matplotlib、seaborn、pandas、Pillow

```bash
pip install -r requirements.txt
```

Chinese-CLIP 模型 `OFA-Sys/chinese-clip-vit-base-patch16` 需提前下载到本地 HuggingFace 缓存，或修改各脚本中的 `_LOCAL_SNAP` 路径指向自己的快照位置。

## 目录结构

```
.
├── README.md                       本文件
├── requirements.txt                Python 依赖列表
├── 基于VLM的食物卡路里识别.md        原题目要求
├── ChineseFood Net 3/              数据源（需自行放置，不入库）
├── build_dataset.py                任务1：构建 20 类数据集（7:1:2 划分）
├── zero_shot.py                    任务2：零样本分类 + 6 种 prompt 模板对比
├── few_shot.py                     任务3-1：10-shot prototype 少样本分类
├── lora_train.py                   任务3-2：LoRA 微调训练
├── lora_eval.py                    任务3-2：加载 adapter 评估
├── data_augment_train.py           任务3-3：OpenCV 数据增强版 LoRA 训练
├── data_augment_eval.py            任务3-3：增强版评估
├── scripts/
│   ├── inspect_model.py            辅助：探查模型结构找 LoRA target_modules
│   └── plot_confusion.py           辅助：绘制混淆矩阵
├── report/
│   └── 王铭翔 - 小作业实验报告.pdf   实验报告
├── dataset_20cls/                  构建好的数据集（由 build_dataset.py 生成）
└── results/                        预测明细、adapter 权重、混淆矩阵图
    ├── lora_adapter/               LoRA 训练权重（2.4MB）
    ├── lora_adapter_aug/           增强版 LoRA 权重
    ├── *_test_preds.csv            各方法的逐图预测
    ├── *_test_summary.json         各方法的结果摘要
    ├── confusion_matrix.png        LoRA 混淆矩阵
    └── confusion_matrix_2.png      零样本混淆矩阵
```

## 数据集

- **来源**：ChineseFoodNet（208 类），从中选取 20 类常见中餐菜品
- **规模**：每类 100 张，共 2000 张，按 7:1:2 划分为训练 70 / 验证 10 / 测试 20
- **20 类菜品**：

| idx | 菜名 | idx | 菜名 |
|---|---|---|---|
| 0 | 麻婆豆腐 | 10 | 红烧肉 |
| 1 | 宫保鸡丁 | 11 | 糖醋排骨 |
| 2 | 回锅肉 | 12 | 梅菜扣肉 |
| 3 | 鱼香肉丝 | 13 | 京酱肉丝 |
| 4 | 水煮鱼 | 14 | 饺子 |
| 5 | 鱼香茄子 | 15 | 包子 |
| 6 | 酸辣土豆丝 | 16 | 扬州炒饭 |
| 7 | 西红柿炒蛋 | 17 | 炸酱面 |
| 8 | 地三鲜 | 18 | 葱爆羊肉 |
| 9 | 蚝油生菜 | 19 | 香辣小龙虾 |

## Quick Start

所有脚本均需在**项目根目录**运行（脚本内用 `os.getcwd()` 定位 `dataset_20cls/` 和 `results/`）。

#### 1. 构建数据集

确保 `ChineseFood Net 3/release_data/` 原始数据存在，执行：

```bash
python build_dataset.py     # 输出到 dataset_20cls/
```

#### 2. 零样本分类基线

```bash
python zero_shot.py     # 对比6种 prompt 模板，结果存入 results/zeroshot_*
```

#### 3. Few-shot 少样本学习

```bash
python few_shot.py     # 10-shot prototype 分类，结果存入 results/fewshot_*
```

#### 4. LoRA 微调

```bash
python lora_train.py    # 训练，adapter 存入 results/lora_adapter/
python lora_eval.py     # 评估，结果存入 results/lora_test_*
```

#### 5. 数据增强版 LoRA

```bash
python data_augment_train.py    # 训练，adapter 存入 results/lora_adapter_aug/
python data_augment_eval.py     # 评估，结果存入 results/lora_aug_test_*
```

#### 6. 绘制混淆矩阵

```bash
python scripts/plot_confusion.py     # 图片输出至 results/confusion_matrix.png
```

## 关键实现说明

- **对比**：所有提升实验均与零样本基线模板 `一张{c}的照片`（78.75%）对比。
- **LoRA 注入位置**：Chinese-CLIP 文本塔（`query`/`value`）和视觉塔（`q_proj`/`v_proj`）命名不一致，详见 `scripts/inspect_model.py`。
- **InfoNCE 损失**：CLIP 式双向交叉熵，对角线为正样本，batch 内其余为负样本。
- **过拟合防止**：每 epoch 在 val 集评估，只保存 val 最优的 adapter，不保存最后一个 epoch。
- **数据增强**：OpenCV 实现旋转/裁剪/颜色抖动。

## 备注

- `ChineseFood Net 3/`、`dataset_20cls/`、`.claude/` 已在 `.gitignore` 中忽略，不入库。
- 各脚本中的 `_LOCAL_SNAP` 是本机 HuggingFace 缓存快照路径，换机器运行时需改为自己的路径或留空使用在线模型名。
