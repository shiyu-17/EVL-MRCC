#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import re
import numpy as np
import json
from datetime import datetime
import argparse

# 定义数据集路径和输出目录
BASE_DIR = "/home/hri3090/lsy/data/7scences"
OUTPUT_BASE_DIR = "/home/hri3090/lsy/pose_estimate/results/7scenes_evaluation"

# 7个数据集名称
DATASETS = [
    "chess",
    "fire",
    "heads",
    "office",
    "pumpkin",
    "redkitchen",
    "stairs"
]

def extract_value(text, pattern):
    """从文本中提取匹配模式的值"""
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None

def extract_int_value(text, pattern):
    """从文本中提取整数值"""
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return None

def parse_result_file(file_path):
    """解析结果文件，提取关键评估数据"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # 找到Best方法部分的文本
            best_section_match = re.search(r"Best 方法:(.*?)(?=\n\n|\Z)", content, re.DOTALL)
            if not best_section_match:
                print(f"警告: 无法在 {file_path} 中找到Best方法部分")
                return None
                
            best_section = best_section_match.group(1)
            
            # 匹配模式
            pairs_pattern = r"满足误差阈值的图像对数量: (\d+)"
            standard_pattern = r"其中标准SIFT方法被选择: (\d+) 次"
            hybrid_pattern = r"其中混合方法被选择: (\d+) 次"
            rot_median_pattern = r"旋转误差中位数: (\d+\.\d+) 度"
            rot_mean_pattern = r"旋转误差均值: (\d+\.\d+) 度"
            trans_median_pattern = r"平移误差中位数: (\d+\.\d+) 米"
            trans_mean_pattern = r"平移误差均值: (\d+\.\d+) 米"
            
            # 提取数据
            data = {
                "valid_pairs": extract_int_value(best_section, pairs_pattern),
                "standard_count": extract_int_value(best_section, standard_pattern),
                "hybrid_count": extract_int_value(best_section, hybrid_pattern),
                "rotation_median": extract_value(best_section, rot_median_pattern),
                "rotation_mean": extract_value(best_section, rot_mean_pattern),
                "translation_median": extract_value(best_section, trans_median_pattern),
                "translation_mean": extract_value(best_section, trans_mean_pattern)
            }
            
            # 确保所有值都提取成功
            if None in data.values():
                missing = [k for k, v in data.items() if v is None]
                print(f"警告: 在 {file_path} 中找不到某些数据: {missing}")
            
            return data
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")
        return None

def run_evaluation_on_dataset(dataset_name, output_dir, frame_interval=1, max_pairs=100, max_rot_err=4.0, max_trans_err=2.0, text_prompts=None):
    """对指定数据集运行位姿估计评估"""
    print(f"\n{'='*50}")
    print(f"开始处理数据集: {dataset_name}")
    print(f"{'='*50}")
    
    # 构建数据集路径
    dataset_dir = os.path.join(BASE_DIR, dataset_name)
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 确保数据集路径存在
    if not os.path.exists(dataset_dir):
        print(f"错误: 数据集路径不存在 - {dataset_dir}")
        return None
    
    # 构建命令
    cmd = [
        "python", "semantic_pose_7scences.py",
        "--dataset-mode",
        "--dataset-type", "custom",
        "--dataset-dir", dataset_dir,
        "--output-dir", dataset_output_dir,
        "--frame-interval", str(frame_interval),
        "--max-pairs", str(max_pairs),
        "--max-rot-err", str(max_rot_err),
        "--max-trans-err", str(max_trans_err)
    ]
    
    # 添加文本提示
    if text_prompts:
        for prompt in text_prompts:
            cmd.extend(["--text-prompts", prompt])
    
    # 执行命令
    cmd_str = " ".join(cmd)
    print(f"执行命令:\n{cmd_str}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # 保存输出
        with open(os.path.join(dataset_output_dir, "stdout.txt"), "w") as f:
            f.write(stdout)
        
        if stderr:
            with open(os.path.join(dataset_output_dir, "stderr.txt"), "w") as f:
                f.write(stderr)
        
        # 检查是否成功
        if process.returncode != 0:
            print(f"警告: 命令返回非零退出码 {process.returncode}")
            print(f"错误输出: {stderr[:500]}{'...' if len(stderr) > 500 else ''}")
            return None
        
        # 解析结果文件
        result_file = os.path.join(dataset_output_dir, "custom_dataset_results.txt")
        if os.path.exists(result_file):
            result_data = parse_result_file(result_file)
            if result_data:
                result_data["dataset"] = dataset_name
            return result_data
        else:
            print(f"错误: 找不到结果文件 {result_file}")
            return None
            
    except Exception as e:
        print(f"执行命令时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="7-Scenes数据集语义引导位姿估计评估")
    
    parser.add_argument("--datasets", nargs="+", choices=DATASETS + ["all"], default=["all"],
                      help="要处理的数据集列表，或'all'表示全部")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_BASE_DIR,
                      help="输出目录")
    parser.add_argument("--frame-interval", type=int, default=1,
                      help="帧间隔")
    parser.add_argument("--max-pairs", type=int, default=100,
                      help="每个数据集最大处理的图像对数量")
    parser.add_argument("--max-rot-err", type=float, default=4.0,
                      help="最大旋转误差(度)")
    parser.add_argument("--max-trans-err", type=float, default=2.0,
                      help="最大平移误差(米)")
    parser.add_argument("--text-prompts", nargs="+", 
                      default=["furniture . objects . scene elements", 
                               "table . chair . cabinet", 
                               "monitor . computer . indoor items"],
                      help="语义文本提示")
                     
    args = parser.parse_args()
    
    # 确定要处理的数据集
    datasets_to_process = DATASETS if "all" in args.datasets else args.datasets
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果汇总文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f"7scenes_summary_{timestamp}.txt")
    json_file = os.path.join(output_dir, f"7scenes_results_{timestamp}.json")
    
    # 存储所有数据集的结果
    all_results = []
    total_values = {
        "valid_pairs": 0,
        "standard_count": 0,
        "hybrid_count": 0,
        "rotation_median": [],
        "rotation_mean": [],
        "translation_median": [],
        "translation_mean": []
    }
    
    # 处理每个数据集
    for dataset in datasets_to_process:
        dataset_result = run_evaluation_on_dataset(
            dataset,
            output_dir, 
            args.frame_interval,
            args.max_pairs,
            args.max_rot_err,
            args.max_trans_err,
            args.text_prompts
        )
        
        if dataset_result:
            all_results.append(dataset_result)
            
            # 累加统计
            total_values["valid_pairs"] += dataset_result["valid_pairs"]
            total_values["standard_count"] += dataset_result["standard_count"]
            total_values["hybrid_count"] += dataset_result["hybrid_count"]
            total_values["rotation_median"].append(dataset_result["rotation_median"])
            total_values["rotation_mean"].append(dataset_result["rotation_mean"])
            total_values["translation_median"].append(dataset_result["translation_median"])
            total_values["translation_mean"].append(dataset_result["translation_mean"])
    
    # 计算平均值
    avg_values = {}
    if all_results:
        avg_values["rotation_median"] = np.mean(total_values["rotation_median"])
        avg_values["rotation_mean"] = np.mean(total_values["rotation_mean"])
        avg_values["translation_median"] = np.mean(total_values["translation_median"])
        avg_values["translation_mean"] = np.mean(total_values["translation_mean"])
        
        # 计算标准SIFT和混合方法的使用比例
        total_pairs = total_values["valid_pairs"]
        if total_pairs > 0:
            avg_values["standard_proportion"] = total_values["standard_count"] / total_pairs * 100
            avg_values["hybrid_proportion"] = total_values["hybrid_count"] / total_pairs * 100
    
    # 写入汇总文件
    with open(summary_file, "w") as f:
        f.write("7-Scenes数据集位姿估计评估汇总\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理的数据集: {', '.join(datasets_to_process)}\n")
        f.write(f"帧间隔: {args.frame_interval}\n")
        f.write(f"每个数据集最大图像对数: {args.max_pairs}\n")
        f.write(f"最大旋转误差: {args.max_rot_err} 度\n")
        f.write(f"最大平移误差: {args.max_trans_err} 米\n")
        f.write(f"文本提示: {', '.join(args.text_prompts)}\n\n")
        
        f.write("各数据集结果:\n")
        f.write("-" * 50 + "\n\n")
        
        for result in all_results:
            dataset = result["dataset"]
            f.write(f"数据集: {dataset}\n")
            f.write(f"  有效图像对数量: {result['valid_pairs']}\n")
            f.write(f"  标准SIFT方法使用次数: {result['standard_count']}\n")
            f.write(f"  混合方法使用次数: {result['hybrid_count']}\n")
            f.write(f"  旋转误差中位数: {result['rotation_median']:.4f} 度\n")
            f.write(f"  旋转误差均值: {result['rotation_mean']:.4f} 度\n")
            f.write(f"  平移误差中位数: {result['translation_median']:.4f} 米\n")
            f.write(f"  平移误差均值: {result['translation_mean']:.4f} 米\n\n")
        
        f.write("所有数据集汇总结果:\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总有效图像对数量: {total_values['valid_pairs']}\n")
        f.write(f"总标准SIFT方法使用次数: {total_values['standard_count']}\n")
        f.write(f"总混合方法使用次数: {total_values['hybrid_count']}\n\n")
        
        if all_results:
            f.write(f"平均旋转误差中位数: {avg_values['rotation_median']:.4f} 度\n")
            f.write(f"平均旋转误差均值: {avg_values['rotation_mean']:.4f} 度\n")
            f.write(f"平均平移误差中位数: {avg_values['translation_median']:.4f} 米\n")
            f.write(f"平均平移误差均值: {avg_values['translation_mean']:.4f} 米\n\n")
            
            f.write(f"标准SIFT方法使用比例: {avg_values['standard_proportion']:.2f}%\n")
            f.write(f"混合方法使用比例: {avg_values['hybrid_proportion']:.2f}%\n")
    
    # 保存JSON格式的结果
    with open(json_file, "w") as f:
        json_data = {
            "datasets": {result["dataset"]: result for result in all_results},
            "total": total_values,
            "average": avg_values,
            "parameters": {
                "frame_interval": args.frame_interval,
                "max_pairs": args.max_pairs,
                "max_rot_err": args.max_rot_err,
                "max_trans_err": args.max_trans_err,
                "text_prompts": args.text_prompts
            }
        }
        json.dump(json_data, f, indent=2)
    
    # 打印汇总结果
    print("\n" + "=" * 50)
    print("7-Scenes数据集评估完成")
    print("=" * 50)
    print(f"结果保存在: {summary_file}")
    print(f"JSON数据保存在: {json_file}")
    
    print("\n所有数据集汇总结果:")
    print(f"总处理数据集数量: {len(all_results)}")
    print(f"总有效图像对数量: {total_values['valid_pairs']}")
    
    if all_results:
        print(f"标准SIFT方法使用比例: {avg_values['standard_proportion']:.2f}%")
        print(f"混合方法使用比例: {avg_values['hybrid_proportion']:.2f}%")
        print(f"平均旋转误差中位数: {avg_values['rotation_median']:.4f} 度")
        print(f"平均旋转误差均值: {avg_values['rotation_mean']:.4f} 度")
        print(f"平均平移误差中位数: {avg_values['translation_median']:.4f} 米")
        print(f"平均平移误差均值: {avg_values['translation_mean']:.4f} 米")

if __name__ == "__main__":
    main() 