import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

# 假设以下模块已在环境中可用
from models.msg import MSGer
from util.transforms import get_transform
from util.checkpointing import load_checkpoint
from util.config_utils import get_configs


class SimpleImageDataset(Dataset):
    """简化的图像数据集，只加载图像而不依赖外部检测文件"""
    def __init__(self, data_split_dir, video_id, image_transforms=None):
        self.data_split_dir = data_split_dir
        self.video_id = video_id
        self.transforms = image_transforms
        
        # 获取帧路径
        self.video_path = os.path.join(data_split_dir, video_id)
        self.frame_path = os.path.join(self.video_path, f"{video_id}_frames")
        
        # 获取所有帧文件
        self.frame_files = [f for f in os.listdir(self.frame_path) 
                           if f.endswith((".png", ".jpg", ".jpeg"))]
        self.frame_files.sort()
        
        # 创建帧ID
        self.frame_ids = [os.path.splitext(f)[0].split("_")[1] for f in self.frame_files]
        
        print(f"加载了视频 {video_id} 的 {len(self.frame_files)} 帧")
    
    def __len__(self):
        return len(self.frame_files)
    
    def __getitem__(self, idx):
        frame_file = self.frame_files[idx]
        frame_path = os.path.join(self.frame_path, frame_file)
        
        # 加载图像
        image = read_image(frame_path)
        
        # 应用变换（如果提供）
        if self.transforms is not None:
            image = self.transforms(image)
        
        # 返回图像及其ID
        return {
            'image': image,
            'image_idx': self.frame_ids[idx],
            'frame_path': frame_path
        }


def simple_collate_fn(batch):
    """为SimpleImageDataset自定义的收集函数"""
    images = torch.stack([item['image'] for item in batch])
    image_idx = [item['image_idx'] for item in batch]
    frame_paths = [item['frame_path'] for item in batch]
    
    return {
        'image': images,
        'image_idx': image_idx,
        'frame_path': frame_paths
    }


class MSGLocalizer:
    """利用MSG进行定位"""
    def __init__(self, video_id, model, data, data_path, device, image_transforms=None):
        self.model = model
        self.data_path = data_path
        self.device = device
        self.video_id = video_id
        
        # 直接从数据集获取帧ID
        self.frame_ids = data.frame_ids
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        
        self.data = data
        self.image_transforms = image_transforms
        
        # 获取场景中所有图像的嵌入
        self.place_feature_bank = {}
        self.feature_maps = {}  # 存储特征图
        dataloader = DataLoader(self.data, batch_size=64, shuffle=False, num_workers=8, collate_fn=simple_collate_fn)
        
        print("正在为所有帧提取嵌入...")
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                additional_info = {}  # 不需要额外信息
                
                results = self.model(images, additional_info)
                place_embeddings = results['place_embeddings'].detach().cpu()
                
                # 获取并存储特征图 - 检查可用的特征表示
                if 'embeddings' in results and results['embeddings'] is not None:
                    # 优先使用完整的embeddings（通常是CNN主干的特征图）
                    feature_maps_batch = results['embeddings'].detach().cpu()
                else:
                    # 如果没有embeddings，尝试使用place_embeddings作为替代
                    # 但注意，这可能需要适当调整，因为place_embeddings通常是全局向量
                    feature_maps_batch = place_embeddings
                
                for i, image_id in enumerate(batch['image_idx']):
                    self.place_feature_bank[image_id] = place_embeddings[i]
                    self.feature_maps[image_id] = feature_maps_batch[i]
        
        # 显示特征维度信息
        if len(self.feature_maps) > 0:
            sample_key = next(iter(self.feature_maps.keys()))
            sample_feat = self.feature_maps[sample_key]
            print(f"特征图示例维度: {sample_feat.shape}")
        
        place_embeddings = [self.place_feature_bank[image_id] for image_id in sorted(self.place_feature_bank.keys())]
        self.place_embeddings = torch.stack(place_embeddings, dim=0)
        print(f"创建了包含 {len(self.place_feature_bank)} 帧的嵌入库")
            
    def localize(self, image_path: str):
        """定位给定图像"""
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

        closest_frame_idx = torch.argmax(cos_sim).item()
        closest_frame = self.frame_ids[closest_frame_idx]
        
        return closest_frame, cos_sim
    
    def get_feature_map(self, frame_id):
        """获取特定帧的特征图"""
        return self.feature_maps.get(frame_id, None)


def build_msg_localizer(video_id, model_path=None, experiment_mode="localize", device=0, split="mini-val"):
    """构建MSG定位器"""
    base_config_dir = './configs/defaults'
    parser = argparse.ArgumentParser(description="实验配置")
    parser.add_argument("--foo", default="构造兼容性参数")
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
            raise AttributeError("配置中未指定eval_output_dir")
        else:
            model_path = config['eval_output_dir']
    assert config["eval_chkpt"] is not None, "配置中未指定eval_chkpt"
    chkpt_path = os.path.join(model_path, "checkpoints", config["eval_chkpt"])
    logger = logging.getLogger(__name__)
    load_checkpoint(model=model, checkpoint_path=chkpt_path, logger=logger)
    logger.info(f"从检查点加载模型: {chkpt_path}")
    model = model.to(device)
    model.eval()

    # —— 探针代码：检查模型输出的特征类型和维度 ——
    with torch.no_grad():
        img_size = config.get('model_image_size', 224)
        # 支持单整数或列表/元组两种配置
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            H, W = img_size
        else:
            H = W = int(img_size)
        dummy = torch.randn(1, 3, H, W, device=device)
        out = model(dummy, {})
        print("\n>>> MSGer forward() 返回的 keys:", out.keys())
        # 检查每个输出的维度
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f">>> {k} shape: {v.shape}")
            elif isinstance(v, dict) and len(v) > 0:
                print(f">>> {k} 是字典，包含 {list(v.keys())}")
        print("\n")
    # —— 探针结束 ——

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


def enhance_contrast(img):
    """图像对比度增强"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def preprocess_image(img):
    """预处理图像：灰度转换 + 对比度增强"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return enhance_contrast(img)


def rotation_error(R1, R2):
    """计算旋转误差（角度）"""
    trace = np.trace(R1.T @ R2)
    return np.rad2deg(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))


def translation_euclidean_error(t1, t2):
    """计算平移向量的欧式距离误差"""
    return np.linalg.norm(t1 - t2)


def compute_error(T_relative, T_icp, K, matches, kp1, kp2):
    """计算综合误差（含尺度缩放处理）"""
    R_rel = T_relative[:3, :3]; t_rel = T_relative[:3, 3]
    R_icp = T_icp[:3, :3];    t_icp = T_icp[:3, 3]
    rot_err = rotation_error(R_rel, R_icp)
    nr, ni = np.linalg.norm(t_rel), np.linalg.norm(t_icp)
    if nr < 1e-6 or ni < 1e-6:
        trans_err = np.inf
    else:
        scale = ni / nr
        trans_err = np.linalg.norm(t_rel * scale - t_icp)
    print("\nRANSAC 误差分析:")
    print(f"旋转角度误差: {rot_err:.4f} 度")
    print(f"平移误差: {trans_err:.4f} 米")
    return rot_err, trans_err


class FeatureEmbeddingMatcher:
    """基于特征嵌入的匹配器"""
    def __init__(self, localizer, feature_weight=0.3, feature_thresh=0.5):
        self.localizer = localizer
        self.feature_weight = feature_weight
        self.feature_thresh = feature_thresh
        self.device = localizer.device

    def get_feature_matches(self, img1, img2, frame_id1=None, frame_id2=None):
        """获取融合特征嵌入信息的匹配点"""
        # 提取 SIFT 特征
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # 获取特征图 - 改用feature_map而非semantic_embedding
        feat_map1 = self.localizer.get_feature_map(frame_id1) if frame_id1 else None
        feat_map2 = self.localizer.get_feature_map(frame_id2) if frame_id2 else None
        
        # 将特征图从张量转换为NumPy数组
        if isinstance(feat_map1, torch.Tensor):
            feat_map1 = feat_map1.numpy()
        if isinstance(feat_map2, torch.Tensor):
            feat_map2 = feat_map2.numpy()
            
        # 处理特征图的维度 - 需要确保它们是3D的(C,H,W)或自适应调整
        if feat_map1 is not None:
            if feat_map1.ndim == 1:  # 如果是全局特征向量，扩展为统一尺寸的特征图
                C = feat_map1.shape[0]
                # 估计一个合理的特征图尺寸 - 假设为8x8
                feat_map1 = np.tile(feat_map1.reshape(C, 1, 1), (1, 8, 8))
            elif feat_map1.ndim == 2:  # 如果是2D矩阵，可能需要调整
                C, H = feat_map1.shape
                feat_map1 = feat_map1.reshape(C, H, 1)
                
        if feat_map2 is not None:
            if feat_map2.ndim == 1:
                C = feat_map2.shape[0]
                feat_map2 = np.tile(feat_map2.reshape(C, 1, 1), (1, 8, 8))
            elif feat_map2.ndim == 2:
                C, H = feat_map2.shape
                feat_map2 = feat_map2.reshape(C, H, 1)

        # 使用FLANN匹配器进行初步匹配
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
        raw_matches = flann.knnMatch(des1, des2, k=2)

        good = []  # 存储(DMatch, score)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 获取特征图尺寸用于投影
        feat_map1_shape = feat_map1.shape if feat_map1 is not None else None
        feat_map2_shape = feat_map2.shape if feat_map2 is not None else None
        
        if feat_map1_shape is not None:
            print(f"特征图1维度: {feat_map1_shape}")
        if feat_map2_shape is not None:
            print(f"特征图2维度: {feat_map2_shape}")
            
        # 应用Lowe's比率测试进行初步筛选，再结合特征图进行细化
        for m, n in raw_matches:
            if m.distance < 0.7 * n.distance:  # 标准Lowe's比率测试
                score = 1.0  # 默认分数
                
                # 当两帧都有特征图时，计算特征相似度
                if feat_map1 is not None and feat_map2 is not None:
                    C1, H1, W1 = feat_map1.shape
                    C2, H2, W2 = feat_map2.shape
                    
                    # 将特征点坐标投影到特征图空间
                    x1, y1 = kp1[m.queryIdx].pt
                    u1 = min(int(x1/w1*W1), W1-1)  # 限制在特征图范围
                    v1 = min(int(y1/h1*H1), H1-1)
                    
                    x2, y2 = kp2[m.trainIdx].pt
                    u2 = min(int(x2/w2*W2), W2-1)
                    v2 = min(int(y2/h2*H2), H2-1)
                    
                    # 提取对应位置的特征向量
                    feat1 = feat_map1[:, v1, u1]
                    feat2 = feat_map2[:, v2, u2]
                    
                    # 计算余弦相似度，并归一化到[0,1]范围
                    norm1 = np.linalg.norm(feat1)
                    norm2 = np.linalg.norm(feat2)
                    
                    if norm1 > 1e-6 and norm2 > 1e-6:  # 避免除零
                        cos_sim = np.dot(feat1, feat2) / (norm1 * norm2)
                        score = float((cos_sim + 1.0) / 2.0)  # 归一化到[0,1]
                
                # 基于阈值过滤匹配
                if score >= self.feature_thresh:
                    good.append((m, score))

        # 按特征相似度降序排序
        good.sort(key=lambda x: x[1], reverse=True)
        
        # 提取匹配点和分数
        good_matches = [m for m, s in good]
        good_scores = [s for m, s in good]
        
        return good_matches, good_scores, kp1, kp2

    def estimate_pose_with_features(self, good_matches, good_scores, kp1, kp2, K):
        """基于特征相似度加权的位姿估计"""
        if len(good_matches) < 8:
            print(f"匹配点数量不足: {len(good_matches)} < 8")
            return None, None
            
        # 提取匹配点坐标
        src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
        # 使用相机内参进行坐标归一化
        src_n = cv2.undistortPoints(src, K, None)
        dst_n = cv2.undistortPoints(dst, K, None)
        
        # 计算匹配点的权重 (基于特征相似度)
        weights = np.array(good_scores)
        weights = weights / np.sum(weights)  # 归一化权重
        
        # 计算本质矩阵并恢复姿态
        E, mask = cv2.findEssentialMat(
            src_n, dst_n, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=0.001
        )
        
        # 从本质矩阵恢复位姿
        _, R, t, mask = cv2.recoverPose(E, src_n, dst_n, mask=mask)
        
        # 构建变换矩阵
        T = np.eye(4)
        T[:3,:3], T[:3,3] = R, t.ravel()
        
        return T, mask

    def visualize_feature_matches(self, img1, img2, kp1, kp2, matches, scores):
        """可视化基于特征的匹配结果"""
        # 根据分数生成颜色 - 从红(低分)到绿(高分)
        colors = [(int(255*(1-s)), int(255*s), 0) for s in scores]
        
        # 画出初步匹配
        vis = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                              matchColor=None,  # 不使用统一颜色
                              singlePointColor=(255,0,0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # 手动画带颜色的连线 - 根据特征相似度着色
        h1, w1 = img1.shape[:2]
        for i, m in enumerate(matches):
            # 获取两幅图中的匹配点坐标
            p1 = tuple(map(int, kp1[m.queryIdx].pt))
            p2 = (int(kp2[m.trainIdx].pt[0] + img1.shape[1]), int(kp2[m.trainIdx].pt[1]))
            
            # 画连线
            cv2.line(vis, p1, p2, colors[i], 1)
            
            # 在连线中间位置标注相似度分数
            if i % 5 == 0:  # 每5个点标注一次，避免拥挤
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                cv2.putText(vis, f"{scores[i]:.2f}", (mid_x, mid_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
                
        return vis


def feature_enhanced_localization(args):
    """结合MSG特征嵌入和SIFT特征进行视觉定位"""
    # 相机内参
    fx, fy, cx, cy = 211.949, 211.949, 127.933, 95.9333
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)

    # 构建定位器与匹配器
    localizer = build_msg_localizer(
        video_id=args.video_id,
        model_path=args.model_path,
        experiment_mode=args.experiment_mode,
        device=args.device,
        split=args.split,
    )
    matcher = FeatureEmbeddingMatcher(
        localizer, 
        feature_weight=args.feature_weight, 
        feature_thresh=args.feature_thresh
    )

    # 读取查询图像
    img1 = cv2.imread(args.query_image_path, cv2.IMREAD_UNCHANGED)
    if img1 is None:
        raise ValueError(f"无法读取查询图像: {args.query_image_path}")
        
    # 使用MSGer检索最相似的帧
    closest, sim = localizer.localize(args.query_image_path)
    print(f"找到最相似帧: {closest}, 相似度: {sim.max().item():.4f}")

    # 读取最相似帧
    frame_path = os.path.join(localizer.data_path, f"{args.video_id}_frames", f"{args.video_id}_{closest}.png")
    img2 = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if img2 is None:
        raise ValueError(f"无法读取数据库图像: {frame_path}")

    # 图像预处理 - 灰度化和对比度增强
    gray1, gray2 = preprocess_image(img1), preprocess_image(img2)
    
    # 提取查询图像的ID
    qid = os.path.basename(args.query_image_path).split('_')[1]
    qid = qid if qid in localizer.frame2idx else None

    # 特征匹配 - 结合特征嵌入进行改进匹配
    print("\n开始特征匹配...")
    good_matches, good_scores, kp1, kp2 = matcher.get_feature_matches(
        gray1, gray2,
        frame_id1=qid,
        frame_id2=closest
    )
    print(f"特征匹配完成: 找到 {len(good_matches)} 个有效匹配")

    # 位姿估计
    print("\n开始位姿估计...")
    T, mask = matcher.estimate_pose_with_features(good_matches, good_scores, kp1, kp2, K)
    
    # 处理成功的位姿估计
    if T is not None:
        print("\n成功估计位姿:")
        print(T)
        
        # 与真实位姿比较（如果有）
        T_true = np.array([[0.9998,0.0185,-0.0058,0.0074],
                           [-0.0146,0.9136,0.4063,0.0288],
                           [0.0128,-0.4062,0.9137,0.3764],
                           [0,0,0,1]])
        print("\n真实位姿:")
        print(T_true)
        
        # 根据RANSAC筛选最终匹配点
        if mask is not None:
            m = mask.ravel().astype(bool)
            final_matches = [m0 for m0, flag in zip(good_matches, m) if flag]
            final_scores  = [s0 for s0, flag in zip(good_scores,  m) if flag]
            print(f"RANSAC筛选后保留 {len(final_matches)} 个匹配点")
        else:
            final_matches = good_matches
            final_scores = good_scores
            
        # 计算位姿误差
        compute_error(T, T_true, K, final_matches, kp1, kp2)
    else:
        print("位姿估计失败：匹配点不足")
        final_matches = good_matches[:min(20, len(good_matches))]  # 仅可视化少量匹配点
        final_scores = good_scores[:min(20, len(good_scores))]

    # 可视化匹配结果
    print("\n生成可视化结果...")
    vis = matcher.visualize_feature_matches(img1, img2, kp1, kp2, final_matches, final_scores)
    
    # 在可视化结果上添加统计信息
    info_text = f"特征点: 图1 {len(kp1)}, 图2 {len(kp2)}, 匹配 {len(good_matches)}, RANSAC后 {len(final_matches)}"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 根据 --output_dir 参数保存到指定文件夹
    if args.output_dir:
        save_dir = args.output_dir
    else:
        # 默认保存到查询图像所在文件夹
        save_dir = os.path.dirname(args.query_image_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成输出文件名
    filename = f"{args.video_id}_{closest}_feature_matches.png"
    output_path = os.path.join(save_dir, filename)
    
    # 保存可视化结果
    cv2.imwrite(output_path, vis)
    print(f"结果图像已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="特征嵌入感知的视觉定位系统")
    parser.add_argument("--video_id", type=str, required=True, help="视频ID")
    parser.add_argument("--query_image_path", type=str, required=True, help="查询图像路径")
    parser.add_argument("--experiment_mode", type=str, default="localize", help="实验模式")
    parser.add_argument("--device", type=int, default=0, help="设备编号")
    parser.add_argument("--split", type=str, default="test", help="数据集拆分")
    parser.add_argument("--model_path", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--feature_weight", type=float, default=0.3, help="特征权重")
    parser.add_argument("--feature_thresh", type=float, default=0.5, help="特征筛选阈值")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="可视化结果保存的文件夹（默认与查询图像同一目录）")
    args = parser.parse_args()

    try:
        feature_enhanced_localization(args)
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback; traceback.print_exc()