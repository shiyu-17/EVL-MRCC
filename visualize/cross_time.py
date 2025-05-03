import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D

# 设置全局字体为Times New Roman，符合学术论文规范
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 读取四个JSON文件
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 假设您有四个JSON文件的路径
file_paths = [
    'data/ARKitScenes_cross_time.json',  # 替换为您的文件路径
    'data/7Scenes_cross_time.json',
    'data/TUM_cross_time.json',
    'data/Cambridge_cross_time.json'
]

# 您可以为每个数据集指定名称
dataset_names = [
    'ARKitScenes',  # 替换为您想要的数据集名称
    '7Scenes',
    'TUM',
    'Cambridge'
]

# 颜色和标记样式，选择了辨识度高且符合学术美观的配色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']
linestyles = ['-', '--', '-.', ':']

# 加载数据
datasets = []
for path in file_paths:
    try:
        datasets.append(load_json_file(path))
    except FileNotFoundError:
        print(f"警告: 文件 {path} 未找到。使用示例数据代替。")
        # 如果文件不存在，使用示例数据（您提供的那个）
        with open('paste.txt', 'r') as f:
            datasets.append(json.load(f))

# 创建四个子图的图表布局
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten()

# 设置图表标题和整体标题
fig.suptitle('Performance Metrics Across Different Temporal Interval Threshold', fontsize=20, y=0.98)

# 图1: Recall@k for k=1
ax = axs[0]
x_values = [1, 5, 10, 30, 60]  # 采样数量
x_labels = ['1', '5', '10', '30', '60']

for i, (data, name) in enumerate(zip(datasets, dataset_names)):
    y_values = [data[str(x)]["recall_at_k"]["1"] for x in x_values]
    ax.plot(range(len(x_values)), y_values, color=colors[i], marker=markers[i], 
            linestyle=linestyles[i], linewidth=2, markersize=8, label=name)

ax.set_title('Recall@1 vs. Temporal Interval Threshold', fontsize=16)
ax.set_xlabel('Threshold', fontsize=14)
ax.set_ylabel('Recall@1', fontsize=14)
ax.set_xticks(range(len(x_values)))
ax.set_xticklabels(x_labels)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_ylim(0.75, 1.05)
ax.legend(loc='lower right', fontsize=12)

# 图2: Precision@k for k=1
ax = axs[1]
for i, (data, name) in enumerate(zip(datasets, dataset_names)):
    y_values = [data[str(x)]["precision_at_k"]["1"] for x in x_values]
    ax.plot(range(len(x_values)), y_values, color=colors[i], marker=markers[i], 
            linestyle=linestyles[i], linewidth=2, markersize=8, label=name)

ax.set_title('Precision@1 vs. Temporal Interval Threshold', fontsize=16)
ax.set_xlabel('Threshold', fontsize=14)
ax.set_ylabel('Precision@1', fontsize=14)
ax.set_xticks(range(len(x_values)))
ax.set_xticklabels(x_labels)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_ylim(0.75, 1.05)
ax.legend(loc='lower right', fontsize=12)

# 图3: MRR (Mean Reciprocal Rank)
ax = axs[2]
for i, (data, name) in enumerate(zip(datasets, dataset_names)):
    y_values = [data[str(x)]["mrr"] for x in x_values]
    ax.plot(range(len(x_values)), y_values, color=colors[i], marker=markers[i], 
            linestyle=linestyles[i], linewidth=2, markersize=8, label=name)

ax.set_title('MRR vs. Temporal Interval Threshold', fontsize=16)
ax.set_xlabel('Threshold', fontsize=14)
ax.set_ylabel('MRR', fontsize=14)
ax.set_xticks(range(len(x_values)))
ax.set_xticklabels(x_labels)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_ylim(0.85, 1.05)
ax.legend(loc='lower right', fontsize=12)

# 图4: NDCG (Normalized Discounted Cumulative Gain)
ax = axs[3]
for i, (data, name) in enumerate(zip(datasets, dataset_names)):
    y_values = [data[str(x)]["ndcg"] for x in x_values]
    ax.plot(range(len(x_values)), y_values, color=colors[i], marker=markers[i], 
            linestyle=linestyles[i], linewidth=2, markersize=8, label=name)

ax.set_title('NDCG vs. Temporal Interval Threshold', fontsize=16)
ax.set_xlabel('Threshold', fontsize=14)
ax.set_ylabel('NDCG', fontsize=14)
ax.set_xticks(range(len(x_values)))
ax.set_xticklabels(x_labels)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_ylim(0.85, 1.05)
ax.legend(loc='lower right', fontsize=12)

# 调整子图之间的间距
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# 保存图表
plt.savefig('performance_metrics_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('performance_metrics_comparison.png', format='png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()