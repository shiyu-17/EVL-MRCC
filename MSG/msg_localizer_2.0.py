# python msg_localizer.py  --video_id 001 --query_image_path /home/dell3/lsy/MSG/data/msg/test/001/001_frames/001_000071.png --split test
import os
import json
import torch
import torch.nn.functional as F
import argparse
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import time
import math

# 假设以下导入已在环境中可用
from models.msg import MSGer
from util.transforms import get_transform
from util.checkpointing import load_checkpoint
from util.config_utils import get_configs


class SimpleImageDataset(Dataset):
    """
    A simplified dataset that only loads images without relying on external detection files
    """
    def __init__(self, data_split_dir, video_id, image_transforms=None):
        """
        Args:
            data_split_dir: Directory containing the data split
            video_id: ID of the video
            image_transforms: Transforms to apply to the images
        """
        self.data_split_dir = data_split_dir
        self.video_id = video_id
        self.transforms = image_transforms
        
        # Get frame paths
        self.video_path = os.path.join(data_split_dir, video_id)
        self.frame_path = os.path.join(self.video_path, f"{video_id}_frames")
        
        # Get all frame files
        self.frame_files = [f for f in os.listdir(self.frame_path) 
                           if f.endswith((".png", ".jpg", ".jpeg"))]
        self.frame_files.sort()
        
        # Create frame IDs
        self.frame_ids = [os.path.splitext(f)[0].split("_")[1] for f in self.frame_files]
        
        print(f"Loaded {len(self.frame_files)} frames for video {video_id}")
    
    def __len__(self):
        return len(self.frame_files)
    
    def __getitem__(self, idx):
        frame_file = self.frame_files[idx]
        frame_path = os.path.join(self.frame_path, frame_file)
        
        # Load image
        image = read_image(frame_path)
        
        # Apply transforms if provided
        if self.transforms is not None:
            image = self.transforms(image)
        
        # Return image and its ID
        return {
            'image': image,
            'image_idx': self.frame_ids[idx],
            'frame_path': frame_path
        }


def simple_collate_fn(batch):
    """
    Custom collate function for SimpleImageDataset
    """
    images = torch.stack([item['image'] for item in batch])
    image_idx = [item['image_idx'] for item in batch]
    frame_paths = [item['frame_path'] for item in batch]
    
    return {
        'image': images,
        'image_idx': image_idx,
        'frame_path': frame_paths
    }


class MSGLocalizer:
    """
    Leverage MSG for localization
    """
    def __init__(self, video_id, model, data, data_path, device, image_transforms=None, 
                 robot_id="robot_0"):
        self.model = model
        self.data_path = data_path
        self.device = device
        self.video_id = video_id
        self.robot_id = robot_id  # 添加机器人ID
        
        # Get frame IDs directly from the dataset
        self.frame_ids = data.frame_ids
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        
        self.data = data
        self.image_transforms = image_transforms
        
        # obtain image embeddings for all images in the scene
        self.place_feature_bank = {}
        dataloader = DataLoader(self.data, batch_size=64, shuffle=False, num_workers=8, collate_fn=simple_collate_fn)
        
        print("Extracting embeddings for all frames...")
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                additional_info = {}  # No additional info needed
                
                results = self.model(images, additional_info)
                results['place_embeddings'] = results['place_embeddings'].detach().cpu()
                
                for i, image_id in enumerate(batch['image_idx']):
                    self.place_feature_bank[image_id] = results['place_embeddings'][i]
        
        place_embeddings = [self.place_feature_bank[image_id] for image_id in sorted(self.place_feature_bank.keys())]
        self.place_embeddings = torch.stack(place_embeddings, dim=0)
        print(f"Created embeddings bank with {len(self.place_feature_bank)} frames")
        
        # 添加用于时序一致性的历史记录
        self.prev_match = None  # 上一次匹配的帧索引
        
        # 添加协作定位相关参数
        self.collaborative_data = {}  # 存储其他机器人的定位结果
        self.localization_history = []  # 存储历史定位结果
            
    def localize(self, image_path: str, top_k=10, use_temporal=False, temporal_weight=0.3):
        """
        定位查询图像，支持时序一致性重排序
        
        Args:
            image_path: 查询图像路径
            top_k: 返回的最相似帧数量
            use_temporal: 是否使用时序一致性重排序
            temporal_weight: 时序得分的权重
            
        Returns:
            best_frame: 最佳匹配帧ID
            similarity: 相似度张量
            top_k_frames: Top-K帧ID和对应分数
        """
        query_image = read_image(image_path)
        if self.image_transforms is not None:
            query_image = self.image_transforms(query_image)
        query_image = query_image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            ret = self.model(query_image, {})
        
        query_embedding = ret["place_embeddings"].detach().cpu()
        cos_sim = F.cosine_similarity(query_embedding, self.place_embeddings, dim=1)

        filename = os.path.splitext(os.path.basename(image_path))[0]
        query_frame_id = filename.split('_')[1]

        if query_frame_id in self.frame2idx:
            self_idx = self.frame2idx[query_frame_id]
            cos_sim[self_idx] = -1.0
        
        # 获取Top-K结果
        top_k_values, top_k_indices = torch.topk(cos_sim, min(top_k, len(cos_sim)))
        top_k_frames = [(self.frame_ids[idx], cos_sim[idx].item()) for idx in top_k_indices]
        
        # 如果启用时序一致性且存在先前匹配
        if use_temporal and self.prev_match is not None:
            # 对Top-K结果应用时序一致性重排序
            top_k_frames = self.temporal_reranking(top_k_indices, cos_sim, temporal_weight)
        
        # 获取最佳匹配
        best_frame_idx = top_k_indices[0].item()
        best_frame = self.frame_ids[best_frame_idx]
        
        # 更新前一个匹配
        self.prev_match = best_frame_idx
        
        # 更新本机器人的历史定位结果
        query_timestamp = int(time.time() * 1000)  # 使用毫秒时间戳作为查询ID
        localization_result = {
            'query_id': query_timestamp,
            'query_frame_id': query_frame_id,
            'best_match': best_frame,
            'top_k_matches': top_k_frames,
            'timestamp': query_timestamp
        }
        self.localization_history.append(localization_result)
        
        return best_frame, cos_sim, top_k_frames
    
    def temporal_reranking(self, top_k_indices, similarity_scores, temporal_weight=0.3):
        """
        基于时序一致性对Top-K结果进行重排序
        
        Args:
            top_k_indices: Top-K索引
            similarity_scores: 原始相似度分数
            temporal_weight: 时序得分权重
            
        Returns:
            重排序后的(帧ID, 分数)列表
        """
        # 计算每个候选帧与上一个匹配帧的时序距离
        reranked_scores = []
        
        for idx in top_k_indices:
            idx = idx.item()
            # 原始相似度分数
            sim_score = similarity_scores[idx].item()
            
            # 计算时序得分 (基于与上一帧的距离)
            if self.prev_match is not None:
                # 计算与上一个匹配帧的帧间距离
                temporal_distance = abs(idx - self.prev_match)
                # 距离越近，时序得分越高 (使用高斯衰减)
                temporal_score = torch.exp(-temporal_distance / 10.0).item()
            else:
                temporal_score = 0.0
            
            # 综合得分 = 原始相似度得分 * (1-权重) + 时序得分 * 权重
            combined_score = sim_score * (1 - temporal_weight) + temporal_score * temporal_weight
            reranked_scores.append((self.frame_ids[idx], combined_score))
        
        # 按综合得分排序
        reranked_scores.sort(key=lambda x: x[1], reverse=True)
        return reranked_scores
    
    def reset_temporal_history(self):
        """重置时序历史记录"""
        self.prev_match = None
    
    def register_robot_result(self, robot_id, localization_result):
        """
        注册其他机器人的定位结果
        
        Args:
            robot_id: 机器人ID
            localization_result: 包含定位结果的字典
        """
        if robot_id not in self.collaborative_data:
            self.collaborative_data[robot_id] = []
        
        self.collaborative_data[robot_id].append(localization_result)
    
    def collaborative_localize(self, image_path, top_k=10, use_temporal=False, 
                               temporal_weight=0.3, voting_weight=0.5, 
                               max_time_diff=5000):
        """
        执行协作定位，整合多个机器人的定位结果
        
        Args:
            image_path: 查询图像路径
            top_k: 返回的候选帧数量
            use_temporal: 是否使用时序一致性
            temporal_weight: 时序得分权重
            voting_weight: 投票机制的权重
            max_time_diff: 考虑其他机器人结果的最大时间差(毫秒)
            
        Returns:
            best_frame: 最佳匹配帧ID
            similarity: 原始相似度张量
            top_k_frames: 融合后的Top-K结果
            ambiguity_level: 歧义程度 (0-1)
        """
        # 首先使用现有方法获取本机器人的定位结果
        best_frame, similarity, top_k_frames = self.localize(
            image_path, top_k, use_temporal, temporal_weight
        )
        
        # 检查是否有其他机器人的数据
        current_time = int(time.time() * 1000)
        recent_results = []
        
        # 收集其他机器人的最近定位结果
        for robot_id, results in self.collaborative_data.items():
            for result in results:
                # 检查结果是否足够新
                if current_time - result['timestamp'] <= max_time_diff:
                    recent_results.append(result)
        
        # 如果没有其他机器人的数据，直接返回本机器人的结果
        if not recent_results:
            print(f"机器人 {self.robot_id} 未收到其他机器人的数据，使用自身定位结果")
            # 返回歧义度为无法确定 (NaN)
            return best_frame, similarity, top_k_frames, float('nan')
        
        print(f"机器人 {self.robot_id} 收到 {len(recent_results)} 个其他机器人的定位结果")
        
        # 创建帧ID到得分的映射
        frame_scores = {}
        for frame_id, score in top_k_frames:
            frame_scores[frame_id] = score
        
        # 整合其他机器人的结果
        for result in recent_results:
            for frame_id, score in result['top_k_matches']:
                if frame_id in frame_scores:
                    # 如果帧ID已存在，增加其得分
                    frame_scores[frame_id] += score * voting_weight
                else:
                    # 如果是新的帧ID，以较低的初始得分添加
                    frame_scores[frame_id] = score * voting_weight
        
        # 转换回列表并重新排序
        collaborative_frames = [(frame_id, score) for frame_id, score in frame_scores.items()]
        collaborative_frames.sort(key=lambda x: x[1], reverse=True)
        collaborative_frames = collaborative_frames[:top_k]
        
        # 计算歧义度 - 检查最高分和次高分的差距
        if len(collaborative_frames) >= 2:
            top_score = collaborative_frames[0][1]
            second_score = collaborative_frames[1][1]
            ambiguity_level = 1.0 - (top_score - second_score) / top_score
        else:
            ambiguity_level = 0.0  # 只有一个候选，没有歧义
        
        # 获取协作定位的最佳结果
        collaborative_best_frame = collaborative_frames[0][0]
        
        print(f"独立定位结果: {best_frame}, 协作定位结果: {collaborative_best_frame}")
        print(f"歧义度: {ambiguity_level:.4f} " + 
              ("(高歧义)" if ambiguity_level > 0.2 else "(低歧义)"))
        
        return collaborative_best_frame, similarity, collaborative_frames, ambiguity_level
    
    def export_localization_result(self):
        """
        导出最近的定位结果，用于与其他机器人共享
        
        Returns:
            最近的定位结果
        """
        if not self.localization_history:
            return None
        return self.localization_history[-1]
    
    def resolve_ambiguity(self, ambiguity_level, top_frames, threshold=0.2):
        """
        根据歧义程度提供建议行动
        
        Args:
            ambiguity_level: 歧义程度
            top_frames: 候选帧列表
            threshold: 歧义阈值
            
        Returns:
            建议的下一步行动
        """
        if ambiguity_level < threshold:
            return {
                'action': 'accept',
                'message': '定位结果可信度高，可以直接采用'
            }
        else:
            # 候选区域 - 提取候选帧所代表的场景区域
            candidate_regions = {}
            for frame_id, _ in top_frames[:3]:  # 考虑前三个候选
                # 简单地用数字作为区域标识
                region_id = int(frame_id) // 10  # 简单区域划分，实际应根据真实场景拓扑
                if region_id not in candidate_regions:
                    candidate_regions[region_id] = 0
                candidate_regions[region_id] += 1
            
            # 找出歧义最大的区域
            ambiguous_regions = sorted(
                candidate_regions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return {
                'action': 'investigate',
                'message': '存在定位歧义，建议进一步调查',
                'candidate_regions': ambiguous_regions,
                'suggestion': '移动到新视角或请求其他机器人协助观察'
            }


def build_msg_localizer(video_id, model_path=None, experiment_mode="localize", device=0, split="mini-val"):
    base_config_dir = './configs/defaults'
    parser = argparse.ArgumentParser(description="Experiment configurations")
    parser.add_argument("--foo", default="construct args for compatibility")
    args = parser.parse_args([])
    args.__dict__.update({
        "experiment": experiment_mode,
        "device": device,
        "eval_split": split,
    })
    config = get_configs(base_config_dir, args, creat_subdir=False)
    
    device_no = config['device']
    device = torch.device(f"cuda:{device_no}" if torch.cuda.is_available() else "cpu")
    model = MSGer(config, device)
    
    if model_path is None:
        if 'eval_output_dir' not in config or config['eval_output_dir'] is None:
            raise AttributeError("eval_output_dir not specified in config")
        else:
            model_path = config['eval_output_dir']
    assert config["eval_chkpt"] is not None, "eval_chkpt not specified in config"
    chkpt_path = os.path.join(model_path, "checkpoints", config["eval_chkpt"])
    logger = logging.getLogger(__name__)
    load_checkpoint(
        model=model, 
        checkpoint_path=chkpt_path,
        logger=logger,
    )
    logger.info(f"Loading model from checkpoint: {chkpt_path}")
    model = model.to(device)
    model.eval()

    data_split_dir = os.path.join(config["dataset_path"], config["eval_split"])
    if "model_image_size" in config:
        transform = get_transform(config['model_image_size'])
    else:
        transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224), antialias=True)
        ])
    dataset = SimpleImageDataset(data_split_dir, video_id, transform)
    localizer = MSGLocalizer(
        video_id=video_id,
        model=model,
        data=dataset,
        data_path=os.path.join(data_split_dir, video_id),
        device=device,
        image_transforms=transform,
    )
    return localizer


# 添加全局变量，在时间上
import time

# 在if __name__ == '__main__'部分之前添加协作定位演示函数
def demonstrate_collaborative_localization(localizers, query_images):
    """
    演示多机器人协作定位功能
    
    Args:
        localizers: 多个MSGLocalizer实例的列表
        query_images: 每个机器人对应的查询图像路径列表
    """
    print("\n===== 开始协作定位演示 =====")
    print(f"机器人数量: {len(localizers)}")
    
    # 确保每个机器人都有对应的查询图像
    assert len(localizers) == len(query_images), "机器人数量必须与查询图像数量相同"
    
    # 依次让每个机器人独立定位
    independent_results = []
    for i, (localizer, image_path) in enumerate(zip(localizers, query_images)):
        print(f"\n机器人 {localizer.robot_id} 执行独立定位...")
        best_frame, similarity, top_k_frames = localizer.localize(
            image_path, top_k=10, use_temporal=False
        )
        independent_results.append({
            'robot_id': localizer.robot_id,
            'best_match': best_frame,
            'top_k_matches': top_k_frames
        })
        print(f"机器人 {localizer.robot_id} 独立定位结果: {best_frame}")
    
    # 机器人间共享结果
    print("\n机器人间共享定位结果...")
    for i, result in enumerate(independent_results):
        for j, localizer in enumerate(localizers):
            if i != j:  # 不需要自己给自己共享
                localizer.register_robot_result(
                    result['robot_id'], 
                    localizer.export_localization_result()
                )
    
    # 执行协作定位
    collaborative_results = []
    for i, (localizer, image_path) in enumerate(zip(localizers, query_images)):
        print(f"\n机器人 {localizer.robot_id} 执行协作定位...")
        collab_best, similarity, collab_top_k, ambiguity = localizer.collaborative_localize(
            image_path, top_k=10, use_temporal=False
        )
        collaborative_results.append({
            'robot_id': localizer.robot_id,
            'best_match': collab_best,
            'top_k_matches': collab_top_k,
            'ambiguity': ambiguity
        })
        
        # 根据歧义程度获取建议
        suggestion = localizer.resolve_ambiguity(ambiguity, collab_top_k)
        print(f"机器人 {localizer.robot_id} 的行动建议: {suggestion['action']}")
        print(f"建议详情: {suggestion['message']}")
    
    # 结果比较和分析
    print("\n===== 协作定位结果分析 =====")
    agreement_count = 0
    for i in range(len(localizers)):
        indep_match = independent_results[i]['best_match']
        collab_match = collaborative_results[i]['best_match']
        robot_id = localizers[i].robot_id
        
        print(f"机器人 {robot_id}:")
        print(f"  独立定位结果: {indep_match}")
        print(f"  协作定位结果: {collab_match}")
        print(f"  歧义度: {collaborative_results[i]['ambiguity']:.4f}")
        
        if indep_match == collab_match:
            agreement_count += 1
            print("  结果: 独立定位与协作定位结果一致")
        else:
            print("  结果: 协作定位修正了独立定位结果")
    
    agreement_rate = agreement_count / len(localizers) * 100
    print(f"\n总结: 独立定位与协作定位一致率: {agreement_rate:.1f}%")
    
    return independent_results, collaborative_results


# 单机器人协作定位模拟演示
def simulate_collaborative_localization(localizer, query_images, simulate_robots=3):
    """
    在单个机器人环境中模拟协作定位
    
    Args:
        localizer: MSGLocalizer实例
        query_images: 查询图像路径列表
        simulate_robots: 模拟的机器人数量
        
    Returns:
        独立定位和协作定位结果
    """
    print("\n===== 开始单机器人协作定位模拟 =====")
    print(f"实际机器人: 1, 模拟机器人数: {simulate_robots}")
    
    # 实际只有一个机器人，处理第一张图像
    real_image = query_images[0]
    print(f"\n真实机器人 {localizer.robot_id} 执行独立定位...")
    best_frame, similarity, top_k_frames = localizer.localize(
        real_image, top_k=10, use_temporal=False
    )
    real_result = {
        'robot_id': localizer.robot_id,
        'best_match': best_frame,
        'top_k_matches': top_k_frames
    }
    print(f"真实机器人独立定位结果: {best_frame}")
    
    # 为了模拟多机器人场景，我们对后续图像进行定位，作为"虚拟"机器人的结果
    virtual_results = []
    
    # 确保有足够的图像
    if len(query_images) >= simulate_robots + 1:
        sim_images = query_images[1:simulate_robots+1]
    else:
        # 不够图像，复制已有的
        sim_images = query_images[1:] + [query_images[0]] * (simulate_robots - len(query_images) + 1)
    
    # 模拟其他机器人的定位结果
    for i, image_path in enumerate(sim_images):
        virtual_robot_id = f"virtual_robot_{i+1}"
        print(f"\n模拟机器人 {virtual_robot_id} 执行独立定位...")
        # 重置时序历史以确保独立定位
        localizer.reset_temporal_history()
        sim_best, sim_similarity, sim_top_k = localizer.localize(
            image_path, top_k=10, use_temporal=False
        )
        
        # 构造模拟结果
        sim_result = {
            'query_id': int(time.time() * 1000) + i,
            'robot_id': virtual_robot_id,
            'best_match': sim_best,
            'top_k_matches': sim_top_k,
            'timestamp': int(time.time() * 1000)
        }
        virtual_results.append(sim_result)
        print(f"模拟机器人 {virtual_robot_id} 独立定位结果: {sim_best}")
        
        # 注册模拟结果
        localizer.register_robot_result(virtual_robot_id, sim_result)
    
    # 执行协作定位
    print("\n真实机器人执行协作定位...")
    collab_best, similarity, collab_top_k, ambiguity = localizer.collaborative_localize(
        real_image, top_k=10, use_temporal=False
    )
    
    # 获取歧义解决建议
    suggestion = localizer.resolve_ambiguity(ambiguity, collab_top_k)
    
    # 分析结果
    print("\n===== 协作定位结果分析 =====")
    print(f"真实机器人独立定位结果: {best_frame}")
    print(f"协作定位结果: {collab_best}")
    print(f"歧义度: {ambiguity:.4f}")
    print(f"行动建议: {suggestion['action']}")
    print(f"建议详情: {suggestion['message']}")
    
    if best_frame == collab_best:
        print("结果: 独立定位与协作定位结果一致")
    else:
        print("结果: 协作定位修正了独立定位结果")
    
    # 返回结果
    return {
        'independent': {
            'real': real_result,
            'virtual': virtual_results
        },
        'collaborative': {
            'best_match': collab_best,
            'top_k_matches': collab_top_k,
            'ambiguity': ambiguity,
            'suggestion': suggestion
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MSG Localizer CLI")
    parser.add_argument("--video_id", type=str, required=True, help="视频ID")
    parser.add_argument("--query_image_path", type=str, required=True, help="查询图像路径")
    parser.add_argument("--experiment_mode", type=str, default="localize", help="实验模式，默认为localize")
    parser.add_argument("--device", type=int, default=0, help="设备编号，默认为0")
    parser.add_argument("--split", type=str, default="test", help="数据集拆分，默认为test")
    parser.add_argument("--model_path", type=str, default=None, help="模型检查点路径，可选")
    parser.add_argument("--top_k", type=int, default=10, help="返回的候选帧数量，默认为10")
    parser.add_argument("--use_temporal", action="store_true", help="是否使用时序一致性重排序")
    parser.add_argument("--temporal_weight", type=float, default=0.3, help="时序一致性权重，默认为0.3")
    parser.add_argument("--query_dir", type=str, default=None, help="查询图像目录，用于连续帧时序重排序")
    # 添加协作定位相关参数
    parser.add_argument("--robot_id", type=str, default="robot_0", help="机器人ID")
    parser.add_argument("--collaborative", action="store_true", help="是否启用协作定位")
    parser.add_argument("--simulate_robots", type=int, default=0, 
                       help="模拟机器人数量，用于单机器人环境下模拟协作定位")
    args = parser.parse_args()

    # 构建本地化器
    localizer = build_msg_localizer(
        video_id=args.video_id,
        model_path=args.model_path,
        experiment_mode=args.experiment_mode,
        device=args.device,
        split=args.split,
    )
    # 设置机器人ID
    localizer.robot_id = args.robot_id

    # 协作定位模式
    if args.collaborative:
        if args.simulate_robots > 0:
            # 单机器人环境模拟协作定位
            if args.query_dir is not None:
                # 从目录获取多张图像
                image_files = [f for f in os.listdir(args.query_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                image_files.sort()
                if len(image_files) == 0:
                    print(f"错误: 查询目录 {args.query_dir} 中没有找到图像文件")
                    exit(1)
                
                query_images = [os.path.join(args.query_dir, f) for f in image_files]
                # 限制图像数量
                query_images = query_images[:args.simulate_robots+1]
                
                # 执行模拟
                simulate_collaborative_localization(
                    localizer, 
                    query_images, 
                    simulate_robots=args.simulate_robots
                )
            else:
                print("错误: 模拟协作定位需要提供--query_dir参数指定多张查询图像")
                exit(1)
        else:
            # 使用主查询图像执行协作定位(但没有其他机器人数据)
            collab_best, similarity, collab_top_k, ambiguity = localizer.collaborative_localize(
                args.query_image_path, 
                top_k=args.top_k,
                use_temporal=args.use_temporal,
                temporal_weight=args.temporal_weight
            )
            
            print(f"协作定位结果: {collab_best}")
            print(f"歧义度: {ambiguity if not math.isnan(ambiguity) else '未知 (无其他机器人数据)'}")
            
            print("\nTop-K协作定位结果:")
            for i, (frame_id, score) in enumerate(collab_top_k):
                print(f"  {i+1}. 帧ID: {frame_id}, 分数: {score:.4f}")
    
    # 处理单张查询图像
    elif args.query_dir is None:
        # 执行本地化
        loc, sim, top_k_frames = localizer.localize(
            args.query_image_path, 
            top_k=args.top_k,
            use_temporal=args.use_temporal,
            temporal_weight=args.temporal_weight
        )
        print(f"最佳匹配帧ID: {loc}")
        print(f"相似度张量大小: {sim.size()}, 数据库中的帧数: {len(localizer.frame2idx)}, 总帧数: {len(localizer.frame_ids)}")
        
        # 打印Top-K结果
        print("\nTop-K匹配结果:")
        for i, (frame_id, score) in enumerate(top_k_frames):
            print(f"  {i+1}. 帧ID: {frame_id}, 分数: {score:.4f}")

        # 可选：打印自相似度
        query_frame_id = os.path.splitext(os.path.basename(args.query_image_path))[0].split('_')[1]
        if query_frame_id in localizer.frame2idx:
            true_id = localizer.frame2idx[query_frame_id]
            print(f"\n查询帧在数据库中的索引: {true_id}, 自相似度: {sim[true_id]}")
    
    # 处理查询图像序列
    else:
        process_query_sequence(localizer, args)


def process_query_sequence(localizer, args):
    """
    处理查询图像序列，演示时序一致性重排序效果
    
    Args:
        localizer: MSGLocalizer实例
        args: 命令行参数
    """
    # 检查查询目录是否存在
    if not os.path.exists(args.query_dir):
        print(f"错误: 查询目录 {args.query_dir} 不存在")
        return
    
    # 获取目录中的所有图像
    image_files = [f for f in os.listdir(args.query_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # 确保按顺序处理
    
    if len(image_files) == 0:
        print(f"错误: 查询目录 {args.query_dir} 中没有找到图像文件")
        return
    
    print(f"处理查询序列，共 {len(image_files)} 张图像")
    
    # 创建结果列表，用于比较有无时序一致性的效果
    results_without_temporal = []
    results_with_temporal = []
    
    # 重置时序历史记录
    localizer.reset_temporal_history()
    
    # 依次处理每张图像
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(args.query_dir, image_file)
        print(f"\n处理图像 {i+1}/{len(image_files)}: {image_file}")
        
        # 不使用时序一致性
        localizer.reset_temporal_history()  # 确保没有历史影响
        loc_without, _, top_k_without = localizer.localize(
            image_path, 
            top_k=args.top_k,
            use_temporal=False
        )
        results_without_temporal.append(loc_without)
        
        # 重置时序历史记录
        if i == 0:
            localizer.reset_temporal_history()
            
        # 使用时序一致性
        loc_with, _, top_k_with = localizer.localize(
            image_path, 
            top_k=args.top_k,
            use_temporal=True,
            temporal_weight=args.temporal_weight
        )
        results_with_temporal.append(loc_with)
        
        # 打印比较结果
        print(f"不使用时序一致性的匹配结果: {loc_without}")
        print(f"使用时序一致性的匹配结果: {loc_with}")
        
        # 打印Top-3结果对比
        print("\n结果对比:")
        print("  不使用时序一致性的Top-3:")
        for j in range(min(3, len(top_k_without))):
            print(f"    {j+1}. 帧ID: {top_k_without[j][0]}, 分数: {top_k_without[j][1]:.4f}")
            
        print("  使用时序一致性的Top-3:")
        for j in range(min(3, len(top_k_with))):
            print(f"    {j+1}. 帧ID: {top_k_with[j][0]}, 分数: {top_k_with[j][1]:.4f}")
    
    # 输出总体统计信息
    print("\n序列匹配统计:")
    different_count = sum(1 for a, b in zip(results_without_temporal, results_with_temporal) if a != b)
    print(f"  总图像数: {len(image_files)}")
    print(f"  不同结果数: {different_count} ({different_count/len(image_files)*100:.1f}%)")
    print(f"  相同结果数: {len(image_files) - different_count} ({(len(image_files) - different_count)/len(image_files)*100:.1f}%)")
    
    # 计算匹配结果的帧号差异
    frame_jumps_without = []
    frame_jumps_with = []
    
    for i in range(1, len(results_without_temporal)):
        prev_idx_without = localizer.frame2idx[results_without_temporal[i-1]]
        curr_idx_without = localizer.frame2idx[results_without_temporal[i]]
        frame_jumps_without.append(abs(curr_idx_without - prev_idx_without))
        
        prev_idx_with = localizer.frame2idx[results_with_temporal[i-1]]
        curr_idx_with = localizer.frame2idx[results_with_temporal[i]]
        frame_jumps_with.append(abs(curr_idx_with - prev_idx_with))
    
    # 计算平均帧跳跃大小
    avg_jump_without = sum(frame_jumps_without) / len(frame_jumps_without) if frame_jumps_without else 0
    avg_jump_with = sum(frame_jumps_with) / len(frame_jumps_with) if frame_jumps_with else 0
    
    print("\n时序平滑度分析:")
    print(f"  不使用时序一致性的平均帧跳跃: {avg_jump_without:.2f}")
    print(f"  使用时序一致性的平均帧跳跃: {avg_jump_with:.2f}")
    print(f"  时序平滑改进: {(avg_jump_without - avg_jump_with) / avg_jump_without * 100:.1f}%")
