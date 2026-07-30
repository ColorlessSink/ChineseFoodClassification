from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

# 路径以本文件所在目录为基准，可从任意位置运行
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根(顶层)
RESULT_DIR = os.path.join(ROOT_DIR, "results")
DATA_DIR = os.path.join(ROOT_DIR, "dataset_20cls")

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]        # 指定用"黑体"
matplotlib.rcParams["axes.unicode_minus"] = False          # 修负号显示问题

# 读取结果csv文件
df = pd.read_csv(os.path.join(RESULT_DIR, "lora_test_preds.csv"))
names = pd.read_csv(os.path.join(DATA_DIR, "classes.csv"))["zh"].tolist()
# 生成混淆矩阵
cm = confusion_matrix(df["true"], df["pred"], labels=range(20))

# 绘图
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=names, yticklabels=names,
            cmap="Blues")
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("LoRA 混淆矩阵")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=150)

# 输出错误率前10的菜品
np.fill_diagonal(cm, 0)   # 屏蔽对角线，只看错误
pairs = []
for i in range(20):
    for j in range(20):
        if i != j and cm[i][j] > 0:
            pairs.append((cm[i][j], i, j))
pairs.sort(reverse=True)
for i in pairs:
    print(f"{names[i[1]]} --> {names[i[2]]}: {i[0].item()}")

