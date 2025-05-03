import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
import os

# 设置全局字体为Times New Roman，符合学术论文规范
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

def visualize_transformation_performance(file_paths, dataset_names, output_filename='image_transformation_performance'):
    """
    可视化多个数据集上不同图像变换方法的性能
    
    参数:
    file_paths: 数据集JSON文件路径列表
    dataset_names: 对应的数据集名称列表
    output_filename: 输出文件名(不含扩展名)
    """
    # 颜色和标记样式，确保学术美观
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']
    
    # 加载数据
    datasets = []
    for path in file_paths:
        try:
            with open(path, 'r') as f:
                datasets.append(json.load(f))
        except FileNotFoundError:
            print(f"警告: 文件 {path} 未找到. 请检查文件路径.")
            return
    
    # 创建图表
    fig = plt.figure(figsize=(18, 12))
    
    # 主要指标名称列表
    metrics = ["recall_at_k", "precision_at_k", "mrr", "ndcg"]
    titles = ["Recall@1", "Precision@1", "MRR", "NDCG"]
    
    # 获取变换方法列表（从第一个数据集中提取）
    transformations = list(datasets[0].keys())
    
    # 将主图划分为2×2的子图
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 
           fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    
    # 设置图表标题
    fig.suptitle('Performance Metrics Across Different Image Transformations', fontsize=20, y=0.98)
    
    # 绘制主要指标图表
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axs[idx]
        
        # 准备数据
        x = np.arange(len(transformations))
        width = 0.8 / len(datasets)  # 柱状图宽度
        
        for i, (dataset, name) in enumerate(zip(datasets, dataset_names)):
            if metric in ["recall_at_k", "precision_at_k"]:
                # 对于recall_at_k和precision_at_k，取k=1的值
                values = [dataset[trans][metric]["1"] if trans in dataset else 0 for trans in transformations]
            else:
                # 对于mrr和ndcg，直接取值
                values = [dataset[trans][metric] if trans in dataset else 0 for trans in transformations]
            
            # 绘制柱状图
            offset = i * width - (len(datasets) - 1) * width / 2
            bars = ax.bar(x + offset, values, width * 0.9, label=name, color=colors[i], 
                         edgecolor='black', linewidth=1, alpha=0.8)
        
        # 设置图表格式
        ax.set_title(title, fontsize=16)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in transformations], rotation=45, ha='right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 设置y轴范围
        if metric in ["recall_at_k", "precision_at_k"]:
            ax.set_ylim(0.9, 1.01)  # 调整显示范围以突出差异
        else:
            ax.set_ylim(0.9, 1.01)
        
        # 只在第一个子图中添加图例
        if idx == 0:
            ax.legend(fontsize=12, loc='lower left')
    
    # 调整整体布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 保存图表
    plt.savefig(f'{output_filename}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_filename}.png', format='png', dpi=300, bbox_inches='tight')
    
    print(f"图表已保存为 {output_filename}.pdf 和 {output_filename}.png")
    
    # 创建第二个图表，展示 Recall@k 和 Precision@k 随 k 的变化
    k_values = range(1, 11)  # 只显示k=1到k=10
    
    # 选择一些代表性的变换方法进行展示
    selected_transforms = ['original', 'gaussian_blur', 'noise', 'rotate_small']
    if not all(t in transformations for t in selected_transforms):
        # 如果找不到预定的变换方法，就选择前4个
        selected_transforms = transformations[:min(4, len(transformations))]
    
    # 为每个数据集创建一个单独的图表
    for d_idx, (dataset, name) in enumerate(zip(datasets, dataset_names)):
        fig2 = plt.figure(figsize=(15, 10))
        gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        axs2 = [fig2.add_subplot(gs2[0, 0]), fig2.add_subplot(gs2[0, 1]), 
                fig2.add_subplot(gs2[1, 0]), fig2.add_subplot(gs2[1, 1])]
        
        # 设置图表标题
        fig2.suptitle(f'Recall@k and Precision@k for {name}', fontsize=20, y=0.98)
        
        # 绘制 Recall@k
        ax = axs2[0]
        for t_idx, transform in enumerate(selected_transforms):
            if transform in dataset:
                recalls = [dataset[transform]["recall_at_k"][str(k)] for k in k_values]
                ax.plot(k_values, recalls, color=colors[t_idx], marker=markers[t_idx], 
                        linestyle=linestyles[t_idx], linewidth=2, markersize=8, 
                        label=transform.replace('_', ' ').title())
        
        ax.set_title('Recall@k', fontsize=16)
        ax.set_xlabel('k', fontsize=14)
        ax.set_ylabel('Recall', fontsize=14)
        ax.set_xticks(k_values)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0.9, 1.01)
        ax.legend(loc='lower right', fontsize=12)
        
        # 绘制 Precision@k
        ax = axs2[1]
        for t_idx, transform in enumerate(selected_transforms):
            if transform in dataset:
                precisions = [dataset[transform]["precision_at_k"][str(k)] for k in k_values]
                ax.plot(k_values, precisions, color=colors[t_idx], marker=markers[t_idx], 
                        linestyle=linestyles[t_idx], linewidth=2, markersize=8, 
                        label=transform.replace('_', ' ').title())
        
        ax.set_title('Precision@k', fontsize=16)
        ax.set_xlabel('k', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_xticks(k_values)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0.9, 1.01)
        ax.legend(loc='lower left', fontsize=12)
        
        # MRR 和 NDCG 对比 (条形图)
        ax = axs2[2]
        x = np.arange(len(selected_transforms))
        width = 0.4  # 柱状图宽度
        
        mrr_values = [dataset[t]["mrr"] if t in dataset else 0 for t in selected_transforms]
        ndcg_values = [dataset[t]["ndcg"] if t in dataset else 0 for t in selected_transforms]
        
        ax.bar(x - width/2, mrr_values, width, label='MRR', color='#1f77b4', 
               edgecolor='black', linewidth=1, alpha=0.8)
        ax.bar(x + width/2, ndcg_values, width, label='NDCG', color='#ff7f0e', 
               edgecolor='black', linewidth=1, alpha=0.8)
        
        ax.set_title('MRR and NDCG Comparison', fontsize=16)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in selected_transforms], 
                         rotation=45, ha='right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        ax.set_ylim(0.9, 1.01)
        ax.legend(loc='lower left', fontsize=12)
        
        # 第四个子图：全部变换方法的MRR比较
        ax = axs2[3]
        x = np.arange(len(transformations))
        
        mrr_values = [dataset[t]["mrr"] if t in dataset else 0 for t in transformations]
        bars = ax.bar(x, mrr_values, color='#1f77b4', edgecolor='black', linewidth=1, alpha=0.8)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_title('MRR Across All Transformations', fontsize=16)
        ax.set_ylabel('MRR Score', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in transformations], 
                         rotation=45, ha='right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        ax.set_ylim(0.9, 1.01)
        
        # 调整布局并保存
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        dataset_filename = f'{output_filename}_{name.replace(" ", "_").lower()}'
        plt.savefig(f'{dataset_filename}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{dataset_filename}.png', format='png', dpi=300, bbox_inches='tight')
        
        print(f"{name}的详细图表已保存为 {dataset_filename}.pdf 和 {dataset_filename}.png")
        plt.close(fig2)
    
    # 显示第一个综合图表
    plt.figure(fig.number)
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 请替换为您的四个JSON文件路径
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
    # 生成可视化
    visualize_transformation_performance(file_paths, dataset_names, 'image_transformation_performance')