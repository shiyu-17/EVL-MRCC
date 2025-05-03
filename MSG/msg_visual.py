# 在4个数据集上进行测试并检索
# {data_split_dir}/{video_id}/{video_id}_frames/video_id_{frame_id}.png
#python msg_evaluator.py --video_id 001 --model_path /home/dell3/lsy/MSG/exp-results/aomsg/LOG_DATE --split test --device 0 --output_dir ./evaluation_results/001
import os
import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
import random
import json
from collections import defaultdict

# Import original code classes and functions
from msg_localizer import (
    build_msg_localizer, 
    SimpleImageDataset, 
    simple_collate_fn
)

# Import evaluation class from msg_evaluator
from msg_evaluator import (
    ImageRetrievalEvaluator,
    create_transform_variations
)

class MultiDatasetEvaluator:
    """
    Evaluate MSGLocalizer's image retrieval capability across multiple datasets 
    and generate comparative analysis
    """
    def __init__(self, dataset_configs, device="cuda:0", output_dir="./evaluation_results"):
        """
        Initialize multi-dataset evaluator
        
        Args:
            dataset_configs: List of dataset configurations
                Each config should have video_id, name, and optionally model_path
            device: Computation device
            output_dir: Directory for output files
        """
        self.dataset_configs = dataset_configs
        self.device = device
        self.output_dir = output_dir
        self.evaluators = {}
        self.results = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set matplotlib style for publication-quality figures
        self.setup_matplotlib_style()
    
    def setup_matplotlib_style(self):
        """Configure matplotlib for publication-quality plots"""
        plt.style.use('seaborn-v0_8-whitegrid')
        matplotlib.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.figsize': (9, 6),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
            'text.usetex': False# True
        })
    
    def initialize_evaluators(self, split="test"):
        """
        Initialize evaluators for all datasets
        
        Args:
            split: Data split to use
            
        Returns:
            dict: Mapping of dataset names to evaluators
        """
        print(f"Initializing evaluators for {len(self.dataset_configs)} datasets...")
        
        for config in self.dataset_configs:
            video_id = config['video_id']
            dataset_name = config.get('name', video_id)
            model_path = config.get('model_path', None)
            
            print(f"Initializing MSGLocalizer for dataset: {dataset_name} (video_id: {video_id})")
            
            # Build localizer
            localizer = build_msg_localizer(
                video_id=video_id,
                model_path=model_path,
                experiment_mode="localize",
                device=self.device,
                split=split
            )
            
            # Create evaluator
            self.evaluators[dataset_name] = ImageRetrievalEvaluator(
                localizer=localizer,
                dataset=localizer.data,
                device=self.device
            )
            
            print(f"Initialized evaluator for {dataset_name} with {len(localizer.data)} frames")
        
        return self.evaluators
    
    def evaluate_all_datasets(self, batch_size=100, topk=20, temporal_threshold=5):
        """
        Evaluate all datasets and collect results
        
        Args:
            batch_size: Number of query images to evaluate
            topk: Top-k results to retrieve
            temporal_threshold: Temporal threshold for relevance
            
        Returns:
            dict: Evaluation results for all datasets
        """
        if not self.evaluators:
            print("No evaluators initialized. Please run initialize_evaluators() first.")
            return
        
        print(f"Evaluating performance across {len(self.evaluators)} datasets...")
        
        for dataset_name, evaluator in self.evaluators.items():
            print(f"Evaluating {dataset_name}...")
            
            # Run evaluation
            evaluator.evaluate_batch(
                batch_size=batch_size, 
                topk=topk, 
                temporal_threshold=temporal_threshold
            )
            
            # Store results
            self.results[dataset_name] = {
                "recall_at_k": evaluator.results["recall_at_k"].copy(),
                "precision_at_k": evaluator.results["precision_at_k"].copy(),
                "mrr": evaluator.results["mrr"],
                "ndcg": evaluator.results["ndcg"],
                "avg_query_time": evaluator.results["avg_query_time"]
            }
            
            # Save individual results
            result_path = os.path.join(self.output_dir, f"{dataset_name}_results.json")
            evaluator.save_results(result_path)
            
            # Create individual plots
            plot_path = os.path.join(self.output_dir, f"{dataset_name}_metrics.png")
            evaluator.plot_metrics(save_path=plot_path)
        
        # Save aggregated results
        self.save_comparative_results()
        
        return self.results
    
    def evaluate_robustness_all_datasets(self):
        """
        Evaluate model robustness to image transformations across all datasets
        
        Returns:
            dict: Robustness evaluation results
        """
        if not self.evaluators:
            print("No evaluators initialized. Please run initialize_evaluators() first.")
            return
        
        robustness_results = {}
        
        for dataset_name, evaluator in self.evaluators.items():
            print(f"Evaluating robustness for {dataset_name}...")
            
            # Create transform variations
            original_transform = evaluator.dataset.transforms
            transform_variations = create_transform_variations(original_transform)
            
            # Run robustness evaluation
            dataset_robustness = evaluator.evaluate_robustness(transform_variations)
            robustness_results[dataset_name] = dataset_robustness
            
            # Save individual robustness results
            result_path = os.path.join(self.output_dir, f"{dataset_name}_robustness.json")
            with open(result_path, 'w') as f:
                serializable_results = {}
                for transform_name, results in dataset_robustness.items():
                    serializable_results[transform_name] = {
                        "recall_at_k": {str(k): float(v) for k, v in results["recall_at_k"].items()},
                        "precision_at_k": {str(k): float(v) for k, v in results["precision_at_k"].items()},
                        "mrr": float(results["mrr"]),
                        "ndcg": float(results["ndcg"])
                    }
                json.dump(serializable_results, f, indent=2)
        
        # Generate comparative robustness plots
        self.plot_comparative_robustness(robustness_results)
        
        return robustness_results
    
    def evaluate_cross_time_all_datasets(self, thresholds=[1, 5, 10, 30, 60]):
        """
        Evaluate cross-time performance across all datasets
        
        Args:
            thresholds: Time interval thresholds to evaluate
            
        Returns:
            dict: Cross-time evaluation results
        """
        if not self.evaluators:
            print("No evaluators initialized. Please run initialize_evaluators() first.")
            return
        
        time_results = {}
        
        for dataset_name, evaluator in self.evaluators.items():
            print(f"Evaluating cross-time performance for {dataset_name}...")
            
            # Run cross-time evaluation
            dataset_time_results = evaluator.evaluate_cross_time(
                frame_interval_thresholds=thresholds
            )
            time_results[dataset_name] = dataset_time_results
            
            # Save individual cross-time results
            result_path = os.path.join(self.output_dir, f"{dataset_name}_cross_time.json")
            with open(result_path, 'w') as f:
                serializable_results = {}
                for threshold, results in dataset_time_results.items():
                    serializable_results[str(threshold)] = {
                        "recall_at_k": {str(k): float(v) for k, v in results["recall_at_k"].items()},
                        "precision_at_k": {str(k): float(v) for k, v in results["precision_at_k"].items()},
                        "mrr": float(results["mrr"]),
                        "ndcg": float(results["ndcg"])
                    }
                json.dump(serializable_results, f, indent=2)
        
        # Generate comparative cross-time plots
        self.plot_comparative_cross_time(time_results, thresholds)
        
        return time_results
    
    def save_comparative_results(self):
        """Save comparative results to JSON file"""
        comparative_path = os.path.join(self.output_dir, "comparative_results.json")
        
        serializable_results = {}
        for dataset_name, results in self.results.items():
            serializable_results[dataset_name] = {
                "recall_at_k": {str(k): float(v) for k, v in results["recall_at_k"].items()},
                "precision_at_k": {str(k): float(v) for k, v in results["precision_at_k"].items()},
                "mrr": float(results["mrr"]),
                "ndcg": float(results["ndcg"]),
                "avg_query_time": float(results["avg_query_time"])
            }
        
        with open(comparative_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Comparative results saved to: {comparative_path}")
    
    def plot_comparative_metrics(self):
        """
        Generate comparative plots of metrics across datasets
        """
        if not self.results:
            print("No results available. Please run evaluate_all_datasets() first.")
            return
        
        # Create a figure with a 2x2 grid for different metrics
        fig = plt.figure(figsize=(12, 10))
        
        # 1. Plot Recall@k
        ax1 = fig.add_subplot(2, 2, 1)
        self._plot_metric_at_k(ax1, "recall_at_k", "Recall@k")
        
        # 2. Plot Precision@k
        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_metric_at_k(ax2, "precision_at_k", "Precision@k")
        
        # 3. Plot MRR and NDCG
        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_ranking_metrics(ax3)
        
        # 4. Plot Query Times
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_query_times(ax4)
        
        # Add super title and adjust layout
        plt.suptitle("Comparative Analysis of Image Retrieval Performance Across Datasets", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure
        output_path = os.path.join(self.output_dir, "comparative_metrics.pdf")
        plt.savefig(output_path)
        print(f"Comparative metrics plot saved to: {output_path}")
        
        # Also save as PNG for easier viewing
        plt.savefig(os.path.join(self.output_dir, "comparative_metrics.png"))
        
        plt.close()
    
    def _plot_metric_at_k(self, ax, metric_name, title):
        """Helper function to plot metrics@k"""
        markers = ['o', 's', '^', 'D']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (dataset_name, results) in enumerate(self.results.items()):
            metric_values = results[metric_name]
            k_values = sorted(metric_values.keys())
            values = [metric_values[k] for k in k_values]
            
            ax.plot(k_values, values, marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)], label=dataset_name, linewidth=2)
        
        ax.set_xlabel('k')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
    
    def _plot_ranking_metrics(self, ax):
        """Helper function to plot MRR and NDCG"""
        datasets = list(self.results.keys())
        mrr_values = [self.results[dataset]["mrr"] for dataset in datasets]
        ndcg_values = [self.results[dataset]["ndcg"] for dataset in datasets]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, mrr_values, width, label='MRR', color='#1f77b4')
        ax.bar(x + width/2, ndcg_values, width, label='NDCG', color='#ff7f0e')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Score')
        ax.set_title('MRR and NDCG Scores')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    def _plot_query_times(self, ax):
        """Helper function to plot query times"""
        datasets = list(self.results.keys())
        times = [self.results[dataset]["avg_query_time"] * 1000 for dataset in datasets]  # Convert to ms
        
        ax.bar(datasets, times, color='#2ca02c')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Query Time (ms)')
        ax.set_title('Average Query Time')
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add actual values on top of bars
        for i, v in enumerate(times):
            ax.text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=9)
    
    def plot_comparative_robustness(self, robustness_results):
        """Generate comparative robustness plots"""
        if not robustness_results:
            return
        
        # Get all transformation names
        all_transforms = set()
        for dataset_results in robustness_results.values():
            all_transforms.update(dataset_results.keys())
        all_transforms = sorted(all_transforms)
        
        # Create a figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # Get dataset names and assign colors
        datasets = list(robustness_results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 1. Recall@1 across transforms
        ax = axes[0]
        self._plot_metric_across_transforms(ax, robustness_results, datasets, colors, 
                                           "recall_at_k", 1, "Recall@1", all_transforms)
        
        # 2. Recall@5 across transforms
        ax = axes[1]
        self._plot_metric_across_transforms(ax, robustness_results, datasets, colors, 
                                           "recall_at_k", 5, "Recall@5", all_transforms)
        
        # 3. MRR across transforms
        ax = axes[2]
        self._plot_metric_across_transforms(ax, robustness_results, datasets, colors, 
                                           "mrr", None, "MRR", all_transforms)
        
        # 4. NDCG across transforms
        ax = axes[3]
        self._plot_metric_across_transforms(ax, robustness_results, datasets, colors, 
                                           "ndcg", None, "NDCG", all_transforms)
        
        # Add super title and adjust layout
        plt.suptitle("Robustness Analysis Across Image Transformations", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure
        output_path = os.path.join(self.output_dir, "comparative_robustness.pdf")
        plt.savefig(output_path)
        print(f"Comparative robustness plot saved to: {output_path}")
        
        # Also save as PNG for easier viewing
        plt.savefig(os.path.join(self.output_dir, "comparative_robustness.png"))
        
        plt.close()
    
    def _plot_metric_across_transforms(self, ax, robustness_results, datasets, colors, 
                                      metric_name, k_value, title, transforms):
        """Helper function to plot metrics across transforms"""
        x = np.arange(len(transforms))
        bar_width = 0.8 / len(datasets)
        
        for i, dataset in enumerate(datasets):
            values = []
            for transform in transforms:
                if transform in robustness_results[dataset]:
                    if k_value is not None:
                        # For metrics@k
                        values.append(
                            robustness_results[dataset][transform][metric_name].get(k_value, 0)
                        )
                    else:
                        # For simple metrics like MRR/NDCG
                        values.append(robustness_results[dataset][transform][metric_name])
                else:
                    values.append(0)
            
            ax.bar(x + i*bar_width - 0.4 + bar_width/2, values, bar_width, 
                   label=dataset, color=colors[i % len(colors)])
        
        ax.set_xlabel('Transformation')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(transforms, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        ax.set_ylim(0, 1.0)
    
    def plot_comparative_cross_time(self, time_results, thresholds):
        """Generate comparative cross-time threshold plots"""
        if not time_results:
            return
        
        # Create a figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # Get dataset names and assign colors
        datasets = list(time_results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']
        
        # 1. Recall@1 vs. time threshold
        ax = axes[0]
        self._plot_metric_vs_threshold(ax, time_results, datasets, colors, markers,
                                      "recall_at_k", 1, "Recall@1", thresholds)
        
        # 2. Recall@5 vs. time threshold
        ax = axes[1]
        self._plot_metric_vs_threshold(ax, time_results, datasets, colors, markers,
                                      "recall_at_k", 5, "Recall@5", thresholds)
        
        # 3. MRR vs. time threshold
        ax = axes[2]
        self._plot_metric_vs_threshold(ax, time_results, datasets, colors, markers,
                                      "mrr", None, "MRR", thresholds)
        
        # 4. NDCG vs. time threshold
        ax = axes[3]
        self._plot_metric_vs_threshold(ax, time_results, datasets, colors, markers,
                                      "ndcg", None, "NDCG", thresholds)
        
        # Add super title and adjust layout
        plt.suptitle("Performance Analysis Across Temporal Thresholds", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure
        output_path = os.path.join(self.output_dir, "comparative_cross_time.pdf")
        plt.savefig(output_path)
        print(f"Comparative cross-time plot saved to: {output_path}")
        
        # Also save as PNG for easier viewing
        plt.savefig(os.path.join(self.output_dir, "comparative_cross_time.png"))
        
        plt.close()
    
    def _plot_metric_vs_threshold(self, ax, time_results, datasets, colors, markers,
                                 metric_name, k_value, title, thresholds):
        """Helper function to plot metrics vs time thresholds"""
        for i, dataset in enumerate(datasets):
            values = []
            for threshold in thresholds:
                threshold_str = str(threshold)
                if threshold_str in time_results[dataset]:
                    if k_value is not None:
                        # For metrics@k
                        values.append(
                            time_results[dataset][threshold_str][metric_name].get(k_value, 0)
                        )
                    else:
                        # For simple metrics like MRR/NDCG
                        values.append(time_results[dataset][threshold_str][metric_name])
                else:
                    values.append(0)
            
            ax.plot(thresholds, values, marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)], label=dataset, linewidth=2)
        
        ax.set_xlabel('Temporal Threshold (frames)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        ax.set_ylim(0, 1.0)

    def generate_table_for_paper(self):
        """
        Generate LaTeX table with key metrics for paper inclusion
        
        Returns:
            str: LaTeX table code
        """
        if not self.results:
            print("No results available. Please run evaluate_all_datasets() first.")
            return
        
        # Define metrics to include in the table
        metrics = [
            ('Recall@1', lambda r: r['recall_at_k'][1]),
            ('Recall@5', lambda r: r['recall_at_k'][5]),
            ('Recall@10', lambda r: r['recall_at_k'][10]),
            ('MRR', lambda r: r['mrr']),
            ('NDCG', lambda r: r['ndcg']),
            ('Query Time (ms)', lambda r: r['avg_query_time'] * 1000)
        ]
        
        # Generate table header
        table = "\\begin{table}[ht]\n"
        table += "\\centering\n"
        table += "\\caption{Image Retrieval Performance Across Multiple Datasets}\n"
        table += "\\label{table:retrieval_performance}\n"
        table += "\\begin{tabular}{l" + "c" * len(self.results) + "}\n"
        table += "\\toprule\n"
        
        # Add dataset names as column headers
        table += "Metric & " + " & ".join(self.results.keys()) + " \\\\\n"
        table += "\\midrule\n"
        
        # Add each metric row
        for metric_name, metric_fn in metrics:
            values = []
            for dataset, results in self.results.items():
                try:
                    value = metric_fn(results)
                    if metric_name == 'Query Time (ms)':
                        values.append(f"{value:.2f}")
                    else:
                        values.append(f"{value:.3f}")
                except (KeyError, TypeError):
                    values.append("--")
            
            table += f"{metric_name} & " + " & ".join(values) + " \\\\\n"
        
        # Finalize table
        table += "\\bottomrule\n"
        table += "\\end{tabular}\n"
        table += "\\end{table}"
        
        # Save table to file
        table_path = os.path.join(self.output_dir, "metrics_table.tex")
        with open(table_path, 'w') as f:
            f.write(table)
        
        print(f"LaTeX table saved to: {table_path}")
        
        return table


def main():
    parser = argparse.ArgumentParser(description="Evaluate MSG image retrieval across multiple datasets")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=200, help="Evaluation query batch size")
    parser.add_argument("--topk", type=int, default=20, help="Top-k results to retrieve")
    parser.add_argument("--split", type=str, default="test", help="Data split (test, train, val)")
    
    args = parser.parse_args()
    
    # Define dataset configurations
    # Each dataset has a video_id and a name for identification
    dataset_configs = [
        {
            "video_id": "41069021",
            "name": "ARKitScenes",
            "model_path": None  # Use default model path
        },
        {
            "video_id": "001",
            "name": "7Scenes",
            "model_path": None
        },
        {
            "video_id": "002",
            "name": "TUM",
            "model_path": None
        },
        {
            "video_id": "003",
            "name": "Cambridge",
            "model_path": None
        }
    ]
    
    # Initialize multi-dataset evaluator
    evaluator = MultiDatasetEvaluator(
        dataset_configs=dataset_configs,
        device=f"{args.device}" if torch.cuda.is_available() else "cpu",
        output_dir=args.output_dir
    )
    
    # Initialize evaluators for each dataset
    evaluator.initialize_evaluators(split=args.split)
    
    # Run basic evaluation on all datasets
    evaluator.evaluate_all_datasets(
        batch_size=args.batch_size,
        topk=args.topk
    )
    
    # Generate comparative plots
    evaluator.plot_comparative_metrics()
    
    # Generate LaTeX table for paper
    evaluator.generate_table_for_paper()
    
    # Run cross-time evaluation
    evaluator.evaluate_cross_time_all_datasets(thresholds=[1, 5, 10, 30, 60])
    
    # Run robustness evaluation
    evaluator.evaluate_robustness_all_datasets()
    
    print(f"All evaluations completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()