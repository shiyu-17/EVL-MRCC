import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置学术论文常用字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


def safe_get(data, field, subkey=None, default=0.0):
    """
    从 data[field] 中取值，或从 data[field][subkey] 中取值。
    如果不存在则返回 default，并打印一次警告。
    """
    if field not in data:
        print(f"⚠️ 警告：字段 '{field}' 不存在，已用默认值 {default}")
        return default if subkey is None else {}
    if subkey is None:
        return data[field]
    return data[field].get(subkey, default)


def visualize_datasets(file_paths, dataset_names, output_prefix='performance'):
    """
    分别生成并保存四张图：Recall@k, Precision@k, MRR & NDCG, Avg Query Time 或 P@1 vs R@1。
    Recall@k 和 MRR&NDCG 的 y 轴不从 0 开始，以便更直观比较。
    """
    # 加载 JSON 数据
    datasets = []
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                datasets.append(json.load(f))
        except FileNotFoundError:
            print(f"❌ 错误：找不到文件 {path}")
            return

    # 收集所有 k 值
    all_k = set()
    for d in datasets:
        all_k.update(int(k) for k in d.get("recall_at_k", {}).keys())
    k_values = sorted(all_k)
    if not k_values:
        print("❌ 错误：未在任何文件中找到 'recall_at_k' 数据，请检查 JSON 结构与路径。")
        return

    # === 图1: Recall@k ===
    fig, ax = plt.subplots(figsize=(8, 6))
    # 计算全局最小 recall，用于设置 y 轴起点
    all_recalls = []
    for data in datasets:
        all_recalls.extend([safe_get(data, 'recall_at_k', str(k), 0.0) for k in k_values])
    ymin_rec = max(min(all_recalls) - 0.05, 0.0)

    for i, (data, name) in enumerate(zip(datasets, dataset_names)):
        recalls = [safe_get(data, "recall_at_k", str(k), 0.0) for k in k_values]
        ax.plot(k_values, recalls,
                color=colors[i], marker=markers[i], linestyle=linestyles[i],
                linewidth=2, markersize=8, label=name)
    ax.set_title('Recall@k', fontsize=16)
    ax.set_xlabel('k', fontsize=14)
    ax.set_ylabel('Recall', fontsize=14)
    ax.set_xticks(k_values)
    ax.set_ylim(ymin_rec, 1.05)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_recall.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存为 {output_prefix}_recall.png")
    plt.show()

    # === 图2: Precision@k ===
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (data, name) in enumerate(zip(datasets, dataset_names)):
        precisions = [safe_get(data, "precision_at_k", str(k), 0.0) for k in k_values]
        ax.plot(k_values, precisions,
                color=colors[i], marker=markers[i], linestyle=linestyles[i],
                linewidth=2, markersize=8, label=name)
    ax.set_title('Precision@k', fontsize=16)
    ax.set_xlabel('k', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_xticks(k_values)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12, loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_precision.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存为 {output_prefix}_precision.png")
    plt.show()

    # === 图3: MRR & NDCG Comparison ===
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = ['MRR', 'NDCG']
    x = np.arange(len(metrics))
    width = 0.2
    # 计算全局最小值
    all_scores = []
    for data in datasets:
        all_scores.append(safe_get(data, 'mrr', default=0.0))
        all_scores.append(safe_get(data, 'ndcg', default=0.0))
    ymin_score = max(min(all_scores) - 0.05, 0.0)

    for i, (data, name) in enumerate(zip(datasets, dataset_names)):
        vals = [safe_get(data, 'mrr', default=0.0), safe_get(data, 'ndcg', default=0.0)]
        ax.bar(x + i*width - 0.3, vals, width,
               label=name, color=colors[i], edgecolor='black', alpha=0.8)
    ax.set_title('MRR and NDCG Comparison', fontsize=16)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.set_ylim(ymin_score, 1.05)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=12, loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_mrr_ndcg.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存为 {output_prefix}_mrr_ndcg.png")
    plt.show()

    # === 图4: Avg Query Time or P@1 vs R@1 ===
    fig, ax = plt.subplots(figsize=(8, 6))
    qtimes = [safe_get(d, 'avg_query_time', default=0.0) * 1000 for d in datasets]
    if any(qtimes):
        bars = ax.bar(dataset_names, qtimes,
                      color=colors[:len(datasets)], edgecolor='black', alpha=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f'{h:.2f}', ha='center', va='bottom', fontsize=12)
        ax.set_title('Average Query Time (ms)', fontsize=16)
        ax.set_ylabel('Time (ms)', fontsize=14)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    else:
        for i, (d, name) in enumerate(zip(datasets, dataset_names)):
            p1 = safe_get(d, 'precision_at_k', '1', 0.0)
            r1 = safe_get(d, 'recall_at_k',    '1', 0.0)
            ax.bar([f"{name}\nP@1", f"{name}\nR@1"], [p1, r1],
                   color=colors[i], edgecolor='black', alpha=0.8)
        ax.set_title('Precision@1 vs. Recall@1', fontsize=16)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_query_time.png', dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存为 {output_prefix}_query_time.png")
    plt.show()


if __name__ == "__main__":
    # 预设颜色、标记、线型（全局变量）
    colors     = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728']
    markers    = ['s',        'o',        '^',        'D'       ]
    linestyles = ['--',        '-',       '-.',       ':'       ]

    file_paths    = [
        'data/7Scenes_results.json',
        'data/ARKitScenes_results.json',
        'data/TUM_results.json',
        'data/Cambridge_results.json'
    ]
    dataset_names = ['7Scenes', 'ARKitScenes', 'TUM', 'Cambridge']

    visualize_datasets(file_paths, dataset_names, output_prefix='model_performance')
