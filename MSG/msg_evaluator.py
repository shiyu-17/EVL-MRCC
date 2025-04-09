import os
import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
import random

# 导入原始代码中的类和函数
from msg_localizer import (
    build_msg_localizer, 
    SimpleImageDataset, 
    simple_collate_fn
)

class ImageRetrievalEvaluator:
    """
    MSGLocalizer图像检索能力评估器
    实现多种评估指标来量化检索性能
    """
    def __init__(self, localizer, dataset, device="cuda:0"):
        """
        初始化评估器
        
        Args:
            localizer: MSGLocalizer实例
            dataset: 图像数据集
            device: 运行设备
        """
        self.localizer = localizer
        self.dataset = dataset
        self.device = device
        self.frame_ids = dataset.frame_ids
        self.frame2idx = localizer.frame2idx
        
        # 记录评估结果
        self.results = {
            "recall_at_k": {},
            "precision_at_k": {},
            "map": None,
            "mrr": None,
            "ndcg": None,
            "query_times": []
        }
    
    def evaluate_query(self, query_idx, topk=10, temporal_threshold=5):
        """
        评估单个查询的检索结果
        
        Args:
            query_idx: 查询图像的索引
            topk: 检索前k个结果
            temporal_threshold: 时间相近性阈值(帧)
            
        Returns:
            dict: 包含评估结果的字典
        """
        query_data = self.dataset[query_idx]
        query_image_path = query_data['frame_path']
        query_frame_id = query_data['image_idx']
        
        # 执行检索并记录时间
        start_time = time.time()
        closest_frame, sim_scores = self.localizer.localize(query_image_path)
        query_time = time.time() - start_time
        
        # 按相似度降序获取topk个结果索引
        _, indices = torch.topk(sim_scores, k=topk)
        retrieved_frames = [self.frame_ids[idx] for idx in indices]
        
        # 计算时间相近性（假设frame_id是时间戳或可以转换为数值）
        query_time_val = float(query_frame_id)
        
        # 对于每个检索结果，判断是否在时间阈值内
        relevant = []
        for frame_id in retrieved_frames:
            try:
                frame_time = float(frame_id)
                if abs(frame_time - query_time_val) <= temporal_threshold:
                    relevant.append(1)
                else:
                    relevant.append(0)
            except ValueError:
                # 如果frame_id不能转为float，认为不相关
                relevant.append(0)
        
        # 计算指标
        query_results = {
            "query_frame_id": query_frame_id,
            "retrieved_frames": retrieved_frames,
            "closest_frame": closest_frame,
            "relevance": relevant,
            "query_time": query_time,
            "raw_scores": sim_scores[indices].tolist()
        }
        
        return query_results
    
    def evaluate_batch(self, query_indices=None, batch_size=100, topk=10, temporal_threshold=5):
        """
        批量评估多个查询
        
        Args:
            query_indices: 要评估的查询索引列表，如果为None则随机抽样
            batch_size: 随机抽样的查询数量
            topk: 检索topk个结果
            temporal_threshold: 时间相近性阈值
            
        Returns:
            dict: 包含评估结果的字典
        """
        if query_indices is None:
            # 随机抽样查询
            total_size = len(self.dataset)
            query_indices = random.sample(range(total_size), min(batch_size, total_size))
        
        all_results = []
        for idx in tqdm(query_indices, desc="Evaluating queries"):
            result = self.evaluate_query(idx, topk, temporal_threshold)
            all_results.append(result)
            self.results["query_times"].append(result["query_time"])
        
        # 计算所有指标
        self._compute_retrieval_metrics(all_results, topk)
        
        return all_results
        
    def _compute_retrieval_metrics(self, results, topk):
        """
        计算整体检索评估指标
        
        Args:
            results: 所有查询结果列表
            topk: 检索topk个结果
        """
        all_relevance = [r["relevance"] for r in results]
        
        # 计算不同k值的召回率和精确率
        k_values = [1, 5, 10, 20, 50] if topk >= 50 else list(range(1, topk+1))
        
        # 计算每个k值的召回率和精确率
        for k in k_values:
            if k > topk:
                continue
                
            # Recall@k: 在前k个结果中找到相关项的比例
            recall_at_k = np.mean([1.0 if sum(rel[:k]) > 0 else 0.0 for rel in all_relevance])
            self.results["recall_at_k"][k] = recall_at_k
            
            # Precision@k: 前k个结果中相关项的比例
            precision_at_k = np.mean([sum(rel[:k])/k for rel in all_relevance])
            self.results["precision_at_k"][k] = precision_at_k
        
        # 计算MRR (Mean Reciprocal Rank)
        mrr_values = []
        for rel in all_relevance:
            # 找到第一个相关项的位置
            try:
                first_rel_pos = rel.index(1) + 1  # +1 因为索引从0开始
                mrr_values.append(1.0 / first_rel_pos)
            except ValueError:
                # 如果没有相关项
                mrr_values.append(0.0)
        
        self.results["mrr"] = np.mean(mrr_values)
        
        # 计算nDCG (Normalized Discounted Cumulative Gain)
        ndcg_values = []
        for rel in all_relevance:
            dcg = sum([(2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel)])
            # 理想情况下的DCG：所有相关项排在前面
            ideal_rel = sorted(rel, reverse=True)
            idcg = sum([(2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_rel)])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_values.append(ndcg)
        
        self.results["ndcg"] = np.mean(ndcg_values)
        
        # 计算平均查询时间
        self.results["avg_query_time"] = np.mean(self.results["query_times"])
    
    def evaluate_cross_time(self, frame_interval_thresholds=[1, 5, 10, 30, 60]):
        """
        评估不同时间间隔对检索性能的影响
        
        Args:
            frame_interval_thresholds: 要测试的时间间隔阈值列表
            
        Returns:
            dict: 不同时间间隔的评估结果
        """
        # 从数据集中选择一些查询
        query_size = min(500, len(self.dataset))
        query_indices = random.sample(range(len(self.dataset)), query_size)
        
        time_results = {}
        for threshold in frame_interval_thresholds:
            print(f"评估时间间隔阈值: {threshold}")
            batch_results = self.evaluate_batch(
                query_indices=query_indices,
                topk=20,
                temporal_threshold=threshold
            )
            
            # 保存这个阈值的结果
            time_results[threshold] = {
                "recall_at_k": self.results["recall_at_k"].copy(),
                "precision_at_k": self.results["precision_at_k"].copy(),
                "mrr": self.results["mrr"],
                "ndcg": self.results["ndcg"]
            }
            
        return time_results
    
    def evaluate_robustness(self, test_transforms):
        """
        评估模型对各种图像变换的鲁棒性
        
        Args:
            test_transforms: 字典，键为变换名称，值为变换函数
            
        Returns:
            dict: 不同变换的评估结果
        """
        # 从数据集中选择一些查询
        query_size = min(200, len(self.dataset))
        query_indices = random.sample(range(len(self.dataset)), query_size)
        
        # 获取原始图像
        original_dataset = self.dataset
        original_transform = self.dataset.transforms
        
        robustness_results = {}
        
        # 对每种变换进行评估
        for transform_name, transform_fn in test_transforms.items():
            print(f"评估变换: {transform_name}")
            
            # 应用变换到数据集
            self.dataset.transforms = transform_fn
            
            # 执行评估
            batch_results = self.evaluate_batch(
                query_indices=query_indices,
                topk=10,
                temporal_threshold=5
            )
            
            # 保存这个变换的结果
            robustness_results[transform_name] = {
                "recall_at_k": self.results["recall_at_k"].copy(),
                "precision_at_k": self.results["precision_at_k"].copy(),
                "mrr": self.results["mrr"],
                "ndcg": self.results["ndcg"]
            }
            
        # 恢复原始变换
        self.dataset.transforms = original_transform
        
        return robustness_results
            
    def plot_metrics(self, save_path=None):
        """
        绘制评估指标图表
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
        """
        plt.figure(figsize=(15, 10))
        
        # 绘制Recall@k和Precision@k
        plt.subplot(2, 2, 1)
        k_values = sorted(self.results["recall_at_k"].keys())
        recall_values = [self.results["recall_at_k"][k] for k in k_values]
        precision_values = [self.results["precision_at_k"][k] for k in k_values]
        
        plt.plot(k_values, recall_values, 'o-', label='Recall@k')
        plt.plot(k_values, precision_values, 's-', label='Precision@k')
        plt.xlabel('k')
        plt.ylabel('Value')
        plt.title('Recall@k and Precision@k')
        plt.legend()
        plt.grid(True)
        
        # 绘制查询时间分布
        plt.subplot(2, 2, 2)
        plt.hist(self.results["query_times"], bins=20)
        plt.xlabel('Query Time (s)')
        plt.ylabel('Frequency')
        plt.title(f'Query Time Distribution (Avg: {self.results["avg_query_time"]:.4f}s)')
        plt.grid(True)
        
        # 绘制MRR和NDCG
        plt.subplot(2, 2, 3)
        metrics = ['MRR', 'NDCG']
        values = [self.results["mrr"], self.results["ndcg"]]
        plt.bar(metrics, values)
        plt.ylim(0, 1.0)
        plt.title('MRR and NDCG Metrics')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()

    def save_results(self, output_path):
        """
        保存评估结果到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        import json
        
        # 将结果转换为可序列化的格式
        serializable_results = {
            "recall_at_k": {str(k): v for k, v in self.results["recall_at_k"].items()},
            "precision_at_k": {str(k): v for k, v in self.results["precision_at_k"].items()},
            "mrr": float(self.results["mrr"]) if self.results["mrr"] is not None else None,
            "ndcg": float(self.results["ndcg"]) if self.results["ndcg"] is not None else None,
            "avg_query_time": float(self.results["avg_query_time"]) if "avg_query_time" in self.results else None,
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"评估结果已保存至: {output_path}")


def create_transform_variations(base_transform):
    """
    创建多种图像变换来测试模型鲁棒性
    
    Args:
        base_transform: 基础变换
        
    Returns:
        dict: 变换名称到变换函数的映射
    """
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from functools import partial
    
    # 创建变换字典
    transforms_dict = {
        "original": base_transform,
    }
    
    # 亮度变化
    def brightness_transform(img, factor):
        if isinstance(img, torch.Tensor):
            return TF.adjust_brightness(img, factor)
        return img
    
    transforms_dict["brightness_low"] = lambda img: brightness_transform(base_transform(img), 0.7)
    transforms_dict["brightness_high"] = lambda img: brightness_transform(base_transform(img), 1.3)
    
    # 对比度变化
    def contrast_transform(img, factor):
        if isinstance(img, torch.Tensor):
            return TF.adjust_contrast(img, factor)
        return img
    
    transforms_dict["contrast_low"] = lambda img: contrast_transform(base_transform(img), 0.7)
    transforms_dict["contrast_high"] = lambda img: contrast_transform(base_transform(img), 1.3)
    
    # 高斯模糊
    def gaussian_blur(img, kernel_size):
        if isinstance(img, torch.Tensor):
            blur = T.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
            return blur(img)
        return img
    
    transforms_dict["gaussian_blur"] = lambda img: gaussian_blur(base_transform(img), 5)
    
    # 随机裁剪
    def random_crop(img, scale):
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
            new_h, new_w = int(h * scale), int(w * scale)
            crop = T.RandomCrop((new_h, new_w))
            resize = T.Resize((h, w), antialias=True)
            return resize(crop(img))
        return img
    
    transforms_dict["random_crop"] = lambda img: random_crop(base_transform(img), 0.9)
    
    # 噪声
    def add_noise(img, std):
        if isinstance(img, torch.Tensor):
            noise = torch.randn_like(img) * std
            return torch.clamp(img + noise, 0, 1)
        return img
    
    transforms_dict["noise"] = lambda img: add_noise(base_transform(img), 0.05)
    
    # 旋转
    def rotate(img, angle):
        if isinstance(img, torch.Tensor):
            return TF.rotate(img, angle)
        return img
    
    transforms_dict["rotate_small"] = lambda img: rotate(base_transform(img), 5)
    
    return transforms_dict


def main():
    parser = argparse.ArgumentParser(description="评估MSGLocalizer的图像检索泛化能力")
    parser.add_argument("--video_id", type=str, required=True, help="要评估的视频ID")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--split", type=str, default="test", help="数据分割(test, train, val)")
    parser.add_argument("--device", type=int, default=0, help="GPU设备ID")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=200, help="评估的查询批量大小")
    parser.add_argument("--topk", type=int, default=20, help="检索topk个结果")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"开始评估视频 {args.video_id} 的图像检索模型")
    
    # 构建MSGLocalizer
    localizer = build_msg_localizer(
        video_id=args.video_id,
        model_path=args.model_path,
        experiment_mode="localize",
        device=args.device,
        split=args.split
    )
    
    print("MSGLocalizer初始化完成")
    
    # 获取原始数据集和变换
    dataset = localizer.data
    original_transform = dataset.transforms
    
    # 创建评估器
    evaluator = ImageRetrievalEvaluator(
        localizer=localizer, 
        dataset=dataset,
        device=f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"开始基本检索性能评估...")
    evaluator.evaluate_batch(batch_size=args.batch_size, topk=args.topk)
    
    # 保存并绘制基本评估结果
    base_output_path = os.path.join(args.output_dir, f"{args.video_id}_base_results.json")
    plot_path = os.path.join(args.output_dir, f"{args.video_id}_metrics.png")
    evaluator.save_results(base_output_path)
    evaluator.plot_metrics(save_path=plot_path)
    
    print("评估不同时间间隔阈值下的性能...")
    time_results = evaluator.evaluate_cross_time()
    
    # 保存时间间隔评估结果
    time_output_path = os.path.join(args.output_dir, f"{args.video_id}_time_threshold_results.json")
    import json
    with open(time_output_path, 'w') as f:
        serializable_results = {}
        for threshold, results in time_results.items():
            serializable_results[str(threshold)] = {
                "recall_at_k": {str(k): float(v) for k, v in results["recall_at_k"].items()},
                "precision_at_k": {str(k): float(v) for k, v in results["precision_at_k"].items()},
                "mrr": float(results["mrr"]),
                "ndcg": float(results["ndcg"])
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"时间间隔评估结果已保存至: {time_output_path}")
    
    # 创建变换变体并评估鲁棒性
    print("评估对不同图像变换的鲁棒性...")
    transform_variations = create_transform_variations(original_transform)
    robustness_results = evaluator.evaluate_robustness(transform_variations)
    
    # 保存鲁棒性评估结果
    robust_output_path = os.path.join(args.output_dir, f"{args.video_id}_robustness_results.json")
    with open(robust_output_path, 'w') as f:
        serializable_results = {}
        for transform_name, results in robustness_results.items():
            serializable_results[transform_name] = {
                "recall_at_k": {str(k): float(v) for k, v in results["recall_at_k"].items()},
                "precision_at_k": {str(k): float(v) for k, v in results["precision_at_k"].items()},
                "mrr": float(results["mrr"]),
                "ndcg": float(results["ndcg"])
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"鲁棒性评估结果已保存至: {robust_output_path}")
    print("所有评估完成！")


if __name__ == "__main__":
    main()