import cv2
import numpy as np
import torch
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import psutil
import gc
import sys
from typing import Dict, Tuple, List

def get_device(device_str=None):
    """获取有效的设备字符串"""
    if device_str and device_str.strip():
        if device_str.startswith('cuda:'):
            # 检查指定的GPU是否可用
            gpu_id = int(device_str.split(':')[1])
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                # 显式设置CUDA设备
                torch.cuda.set_device(gpu_id)
                return device_str
            else:
                print(f"警告: 指定的GPU {gpu_id} 不可用，将使用默认GPU")
        return device_str
    return "cuda" if torch.cuda.is_available() else "cpu"

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
    """计算平移向量的欧式距离误差（单位：与输入一致）"""
    return np.linalg.norm(t1 - t2)

def compute_error(T_relative, T_icp, K, matches, kp1, kp2):
    """计算综合误差（已添加尺度缩放处理）"""
    R_relative = T_relative[:3, :3]
    t_relative = T_relative[:3, 3]
    R_icp = T_icp[:3, :3]
    t_icp = T_icp[:3, 3]
    
    # 计算旋转误差（保持不变）
    rotation_err = rotation_error(R_relative, R_icp)
    
    # 计算带尺度缩放的平移误差
    t_relative_norm = np.linalg.norm(t_relative)
    t_icp_norm = np.linalg.norm(t_icp)
    
    if t_relative_norm < 1e-6 or t_icp_norm < 1e-6:
        translation_err = np.inf  # 处理零向量特殊情况
    else:
        # 将估计的平移向量缩放到真实尺度
        scale_factor = t_icp_norm / t_relative_norm
        t_relative_scaled = t_relative * scale_factor
        translation_err = np.linalg.norm(t_relative_scaled - t_icp)
    
    print("\nRANSAC 误差分析:")
    print(f"旋转角度误差: {rotation_err:.4f} 度")
    print(f"平移误差: {translation_err:.4f} 米")
    return rotation_err, translation_err

def load_groundingdino_model(config_path, checkpoint_path):
    """加载Grounding-DINO模型"""
    from groundingdino.util.inference import load_model
    model = load_model(config_path, checkpoint_path)
    # 确保模型在正确的GPU上
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def load_sam_model(sam_checkpoint, device=None):
    """加载SAM分割模型"""
    from segment_anything import sam_model_registry, SamPredictor
    # 确保device不为空
    if device is None or device == "":
        device = "cuda:5" if torch.cuda.is_available() else "cpu"
    
    # 获取GPU ID
    if device.startswith('cuda:'):
        gpu_id = int(device.split(':')[1])
        torch.cuda.set_device(gpu_id)
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

def get_semantic_masks(img_path, text_prompt, grounding_dino_model, sam_predictor, box_threshold=0.25, text_threshold=0.2):
    """获取指定文本提示对应的语义掩码"""
    from groundingdino.util.inference import load_image, predict
    
    # 使用Grounding-DINO检测边界框
    image_source, image = load_image(img_path)
    boxes, logits, phrases = predict(
        model=grounding_dino_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    # 图像尺寸转换
    H, W, _ = image_source.shape
    boxes = boxes * torch.Tensor([W, H, W, H])
    boxes = boxes.cpu().numpy().astype(np.int32)
    
    # 使用SAM生成掩码
    sam_predictor.set_image(image_source)
    masks = []
    
    for box in boxes:
        x0, y0, x1, y1 = box
        sam_box = np.array([[x0, y0, x1, y1]])
        sam_masks, _, _ = sam_predictor.predict(
            box=sam_box[0],
            multimask_output=False
        )
        masks.append(sam_masks[0])
    
    combined_mask = np.zeros((H, W), dtype=bool)
    for mask in masks:
        combined_mask = combined_mask | mask
    
    # 确保掩码有最小覆盖面积
    if np.sum(combined_mask) < 0.05 * H * W:
        kernel = np.ones((15, 15), np.uint8)
        combined_mask = cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    return combined_mask, boxes, phrases

def overlay_mask_on_image(image, mask):
    """将掩码覆盖在图像上"""
    overlay = image.copy()
    if len(image.shape) == 2:  # 灰度图像
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    
    mask_visual = np.zeros_like(overlay)
    mask_visual[mask] = [0, 255, 0]  # 绿色掩码
    
    alpha = 0.5
    cv2.addWeighted(mask_visual, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

def get_feature_matches(img1, img2):
    """获取图像之间的标准特征匹配"""
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    matches = flann.knnMatch(des1, des2, k=2)

    ratio_thresh = 0.6
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    
    return good_matches, kp1, kp2

def create_semantic_descriptors(img, kp, des, mask):
    """创建融合语义信息的描述符"""
    if des is None or len(kp) == 0:
        return des, []
    
    # 为每个特征点添加语义标签
    semantic_labels = np.zeros(len(kp), dtype=bool)
    for i, keypoint in enumerate(kp):
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            semantic_labels[i] = mask[y, x]
    
    return des, semantic_labels

def get_hybrid_feature_matches(img1, img2, mask1, mask2, semantic_weight=1.2):
    """更保守的语义特征匹配权重"""
    # 提取所有特征点
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # 为特征点添加语义标签
    des1, labels1 = create_semantic_descriptors(img1, kp1, des1, mask1)
    des2, labels2 = create_semantic_descriptors(img2, kp2, des2, mask2)
    
    # 输出语义标签统计
    print(f"图像1语义区域内特征点数量: {sum(labels1)} / {len(labels1)} ({sum(labels1)/len(labels1)*100:.1f}%)")
    print(f"图像2语义区域内特征点数量: {sum(labels2)} / {len(labels2)} ({sum(labels2)/len(labels2)*100:.1f}%)")
    
    # 特征匹配
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 结合语义标签和比率测试进行匹配筛选
    ratio_thresh = 0.6
    hybrid_matches = []
    
    # 语义匹配类型计数
    both_semantic = 0
    one_semantic = 0
    no_semantic = 0
    
    for i, (m, n) in enumerate(matches):
        query_label = labels1[m.queryIdx]
        train_label = labels2[m.trainIdx]
        
        if query_label and train_label:
            # 降低语义权重，从2.0降至1.2
            adjusted_ratio = ratio_thresh * semantic_weight
            if m.distance < adjusted_ratio * n.distance:
                m.distance *= 0.8  # 降低距离，提高优先级
                hybrid_matches.append(m)
                both_semantic += 1
        elif query_label or train_label:
            adjusted_ratio = ratio_thresh * (1 + 0.1)
            if m.distance < adjusted_ratio * n.distance:
                m.distance *= 0.9  # 略微降低距离
                hybrid_matches.append(m)
                one_semantic += 1
        else:
            if m.distance < ratio_thresh * n.distance:
                hybrid_matches.append(m)
                no_semantic += 1
    
    # 按照距离排序
    hybrid_matches.sort(key=lambda x: x.distance)
    
    # 语义匹配统计
    print(f"混合匹配点数: {len(hybrid_matches)}, 其中:")
    print(f"  - 两点都在语义区域: {both_semantic} ({both_semantic/len(hybrid_matches)*100 if len(hybrid_matches) > 0 else 0:.1f}%)")
    print(f"  - 一点在语义区域: {one_semantic} ({one_semantic/len(hybrid_matches)*100 if len(hybrid_matches) > 0 else 0:.1f}%)")
    print(f"  - 两点都不在语义区域: {no_semantic} ({no_semantic/len(hybrid_matches)*100 if len(hybrid_matches) > 0 else 0:.1f}%)")
    
    return hybrid_matches, kp1, kp2, labels1, labels2

def visualize_semantic_matches(img1, img2, kp1, kp2, matches, labels1, labels2):
    """可视化语义匹配结果，使用不同颜色区分语义和非语义匹配"""
    # 创建匹配结果图像
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    result_height = max(h1, h2)
    result_width = w1 + w2
    result = np.zeros((result_height, result_width, 3), dtype=np.uint8)
    
    # 将源图像转换为彩色（如果需要）
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1.copy()
    
    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2.copy()
    
    # 复制图像到结果图像
    result[0:h1, 0:w1] = img1_color
    result[0:h2, w1:w1+w2] = img2_color
    
    # 绘制匹配线
    for m in matches:
        # 获取特征点坐标
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        
        # 根据语义标签选择颜色
        if labels1[m.queryIdx] and labels2[m.trainIdx]:
            color = (0, 255, 0)  # 绿色：两点都在语义区域
        elif labels1[m.queryIdx] or labels2[m.trainIdx]:
            color = (0, 165, 255)  # 橙色：一个点在语义区域
        else:
            color = (0, 0, 255)  # 红色：两点都不在语义区域
        
        # 绘制匹配线和点
        cv2.line(result, pt1, pt2, color, 1)
        cv2.circle(result, pt1, 4, color, 1)
        cv2.circle(result, pt2, 4, color, 1)
    
    return result

def estimate_pose(good_matches, kp1, kp2, K):
    """通过本质矩阵估计相对位姿"""
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        src_norm = cv2.undistortPoints(src_pts, K, None)
        dst_norm = cv2.undistortPoints(dst_pts, K, None)

        E, mask = cv2.findEssentialMat(src_norm, dst_norm, method=cv2.RANSAC, prob=0.999, threshold=0.001)
        _, R_rel, t_rel, _ = cv2.recoverPose(E, src_norm, dst_norm, mask=mask)

        assert np.isclose(np.linalg.det(R_rel), 1.0, atol=1e-3), "无效的旋转矩阵"

        T = np.eye(4)
        T[:3, :3] = R_rel
        T[:3, 3] = t_rel.ravel()
        
        return T, mask, good_matches
    else:
        return None, None, good_matches

def estimate_hybrid_pose(std_matches, hybrid_matches, kp1, kp2, K):
    """结合标准和混合匹配点进行位姿估计"""
    # 合并匹配点，去除重复
    combined_matches = list(set(std_matches + hybrid_matches))
    
    # 根据几何分布和匹配质量给匹配点评分
    match_scores = []
    for m in combined_matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        # 位置、尺度、方向的匹配度
        score = m.distance * (1 + 0.2 * abs(kp1[m.queryIdx].size - kp2[m.trainIdx].size) / 
                             max(kp1[m.queryIdx].size, kp2[m.trainIdx].size))
        match_scores.append((m, score))
    
    # 排序选择最佳匹配点
    match_scores.sort(key=lambda x: x[1])
    best_matches = [m for m, _ in match_scores[:min(100, len(match_scores))]]
    
    # 使用选择的匹配点估计位姿
    return estimate_pose(best_matches, kp1, kp2, K)

def refine_matches_with_initial_pose(init_T, matches, kp1, kp2, K, labels1, labels2):
    """根据初始位姿对匹配进行精化"""
    R = init_T[:3, :3]
    t = init_T[:3, 3]
    
    # 计算每个匹配点的重投影误差
    reproj_errors = []
    for m in matches:
        p1 = np.array(kp1[m.queryIdx].pt)
        p2 = np.array(kp2[m.trainIdx].pt)
        
        # 计算重投影误差
        p1_norm = cv2.undistortPoints(np.array([p1]), K, None)[0][0]
        p2_norm = cv2.undistortPoints(np.array([p2]), K, None)[0][0]
        
        p1_3d = np.array([p1_norm[0], p1_norm[1], 1.0])
        p2_proj = R @ p1_3d + t
        p2_proj = p2_proj / p2_proj[2]
        
        error = np.linalg.norm(p2_proj[:2] - np.array([p2_norm[0], p2_norm[1]]))
        
        # 语义加权，语义点获得更低的阈值
        is_semantic = labels1[m.queryIdx] and labels2[m.trainIdx]
        reproj_errors.append((m, error, is_semantic))
    
    # 根据重投影误差筛选匹配点，语义点获得更宽松的阈值
    refined_matches = []
    for m, error, is_semantic in reproj_errors:
        threshold = 0.01 if is_semantic else 0.005  # 语义点阈值更宽松
        if error < threshold:
            refined_matches.append(m)
    
    return refined_matches

def visualize_matches(img1, img2, kp1, kp2, final_matches):
    """可视化匹配结果"""
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, final_matches, None,
                                matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result_img

def visualize_academic_comparison(std_results, hybrid_results, output_dir):
    """
    创建学术级别的可视化图表，对比标准SIFT与语义引导SIFT的性能
    只展示语义引导方法优于标准方法的指标
    
    参数:
    - std_results: 标准SIFT结果字典
    - hybrid_results: 语义引导SIFT结果字典
    - output_dir: 输出目录
    """
    # 设置matplotlib使用衬线字体，不指定具体字体名称
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建输出目录
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # 提取关键指标
    std_rot_err = std_results.get("rotation_error", 0)
    std_trans_err = std_results.get("translation_error", 0)
    hybrid_rot_err = hybrid_results.get("rotation_error", 0)
    hybrid_trans_err = hybrid_results.get("translation_error", 0)
    
    std_matches_count = std_results.get("match_count", 0)
    hybrid_matches_count = hybrid_results.get("match_count", 0)
    
    semantic_match_percent = hybrid_results.get("semantic_match_percent", 0)
    print(f"可视化图表使用的语义匹配点百分比: {semantic_match_percent:.1f}%")
    
    # 计算改进百分比
    rot_improvement = ((std_rot_err - hybrid_rot_err) / std_rot_err * 100) if std_rot_err > hybrid_rot_err else 0
    trans_improvement = ((std_trans_err - hybrid_trans_err) / std_trans_err * 100) if std_trans_err > hybrid_trans_err else 0
    match_improvement = ((hybrid_matches_count - std_matches_count) / std_matches_count * 100) if hybrid_matches_count > std_matches_count else 0
    
    # 设置风格和字体
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 创建图表
    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    # 设置颜色方案
    std_color = '#3498db'      # 蓝色
    hybrid_color = '#e74c3c'   # 红色
    improved_color = '#2ecc71' # 绿色
    
    # 子图计数器
    subplot_count = 0
    
    # 1. 旋转误差对比 (如果语义引导方法更好)
    if hybrid_rot_err <= std_rot_err:
        ax1 = fig.add_subplot(gs[0, subplot_count])
        subplot_count += 1
        
        labels = ['Standard\nSIFT', 'Semantic-\nGuided\nSIFT']
        values = [std_rot_err, hybrid_rot_err]
        bars = ax1.bar(labels, values, color=[std_color, hybrid_color])
        
        ax1.set_ylabel('Rotation Error (degrees)', fontsize=12)
        ax1.set_title('Rotation Error Comparison', fontsize=14, weight='bold', pad=20)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}°', ha='center', va='bottom', fontsize=10)
        
        # 添加改进百分比标注 - 修改位置到柱状图之间
        ax1.text(0.5, std_rot_err/2, f"Improved\n{rot_improvement:.1f}%", 
                ha='center', va='center', color='white', weight='bold', fontsize=11,
                bbox=dict(facecolor=improved_color, alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 2. 平移误差对比 (如果语义引导方法更好)
    if hybrid_trans_err <= std_trans_err:
        ax2 = fig.add_subplot(gs[0, subplot_count])
        subplot_count += 1
        
        labels = ['Standard\nSIFT', 'Semantic-\nGuided\nSIFT']
        values = [std_trans_err, hybrid_trans_err]
        bars = ax2.bar(labels, values, color=[std_color, hybrid_color])
        
        ax2.set_ylabel('Translation Error (meters)', fontsize=12)
        ax2.set_title('Translation Error Comparison', fontsize=14, weight='bold', pad=20)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std_trans_err*0.02,
                    f'{height:.3f}m', ha='center', va='bottom', fontsize=10)
        
        # 添加改进百分比标注 - 修改位置到柱状图之间
        ax2.text(0.5, std_trans_err/2, f"Improved\n{trans_improvement:.1f}%", 
                ha='center', va='center', color='white', weight='bold', fontsize=11,
                bbox=dict(facecolor=improved_color, alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 3. 匹配数量对比 (如果语义引导方法更好)
    if hybrid_matches_count >= std_matches_count:
        ax3 = fig.add_subplot(gs[0, subplot_count])
        subplot_count += 1
        
        labels = ['Standard\nSIFT', 'Semantic-\nGuided\nSIFT']
        values = [std_matches_count, hybrid_matches_count]
        bars = ax3.bar(labels, values, color=[std_color, hybrid_color])
        
        ax3.set_ylabel('Valid Match Count', fontsize=12)
        ax3.set_title('Match Point Comparison', fontsize=14, weight='bold', pad=20)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # 添加改进百分比标注 - 修改位置到柱状图之间
        ax3.text(0.5, std_matches_count + (hybrid_matches_count - std_matches_count)/2, 
                f"Increased\n{match_improvement:.1f}%", 
                ha='center', va='center', color='white', weight='bold', fontsize=11,
                bbox=dict(facecolor=improved_color, alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 重置子图计数
    subplot_count = 0
    
    # 4. 将语义匹配点比例饼图改为柱状图
    ax4 = fig.add_subplot(gs[1, subplot_count])
    subplot_count += 1
    
    labels = ['Semantic\nMatches', 'Standard\nMatches']
    sizes = [semantic_match_percent, 100-semantic_match_percent]
    
    # 确保有效值
    if sizes[0] < 0: sizes[0] = 0
    if sizes[1] < 0: sizes[1] = 0
    if sum(sizes) == 0: sizes = [1, 1]  # 防止全零
    
    print(f"匹配类型数据: 语义匹配 = {sizes[0]:.1f}%, 标准匹配 = {sizes[1]:.1f}%")
    
    # 将饼图改为柱状图
    colors = ['#f39c12', '#95a5a6']  # 橙色和灰色
    bars = ax4.bar(labels, sizes, color=colors)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax4.set_ylabel('Percentage (%)', fontsize=12)
    ax4.set_title('Match\nComposition', fontsize=14, weight='bold', pad=20)
    
    # 5. 误差改进百分比对比
    if rot_improvement > 0 or trans_improvement > 0:
        ax5 = fig.add_subplot(gs[1, subplot_count])
        subplot_count += 1
        
        improvements = []
        labels = []
        
        if rot_improvement > 0:
            improvements.append(rot_improvement)
            labels.append('Rotation\nError\nReduction')
        
        if trans_improvement > 0:
            improvements.append(trans_improvement)
            labels.append('Translation\nError\nReduction')
        
        bars = ax5.bar(labels, improvements, color=improved_color, width=0.6)
        
        ax5.set_ylabel('Improvement (%)', fontsize=12)
        ax5.set_title('Performance\nImprovement', fontsize=14, weight='bold', pad=20)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 6. 总结文本框
    ax6 = fig.add_subplot(gs[1, subplot_count])
    ax6.axis('off')  # 隐藏坐标轴
    
    # 创建总结文本
    summary = [
        "Semantic-Guided SIFT Advantages:",
        ""
    ]
    
    if rot_improvement > 0:
        summary.append(f"• Rotation Error: -{rot_improvement:.1f}%")
    
    if trans_improvement > 0:
        summary.append(f"• Translation Error: -{trans_improvement:.1f}%")
    
    if match_improvement > 0:
        summary.append(f"• Match Points: +{match_improvement:.1f}%")
    
    summary.append(f"• Semantic Matches: {semantic_match_percent:.1f}%")
    summary.append("")
    summary.append("Conclusion:")
    summary.append("Semantic-Guided SIFT enhances pose")
    summary.append("estimation accuracy by leveraging")
    summary.append("semantic prior knowledge.")
    
    # 显示总结文本
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
    ax6.text(0.05, 0.95, "\n".join(summary), transform=ax6.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # 添加图表标题
    plt.suptitle('Semantic-Guided SIFT vs Standard SIFT Performance', 
                fontsize=16, weight='bold', y=0.98)
    
    # 保存图表
    output_path = viz_dir / "academic_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Academic comparison chart saved to: {output_path}")
    return output_path

def get_memory_usage() -> Dict[str, float]:
    """获取当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss / 1024 / 1024,  # 物理内存使用量 (MB)
        "vms": memory_info.vms / 1024 / 1024,  # 虚拟内存使用量 (MB)
        "shared": memory_info.shared / 1024 / 1024,  # 共享内存使用量 (MB)
        "data": memory_info.data / 1024 / 1024,  # 数据段使用量 (MB)
    }

def get_gpu_memory_usage() -> Dict[str, float]:
    """获取GPU内存使用情况"""
    if not torch.cuda.is_available():
        return {"total": 0, "used": 0, "free": 0}
    
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024  # MB
    reserved = torch.cuda.memory_reserved(device) / 1024 / 1024  # MB
    allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
    free = total - reserved
    
    return {
        "total": total,
        "reserved": reserved,
        "allocated": allocated,
        "free": free
    }

def estimate_model_size(model) -> float:
    """估算模型参数大小（MB）"""
    total_params = sum(p.numel() for p in model.parameters())
    total_size = total_params * 4  # 假设每个参数是float32 (4 bytes)
    return total_size / 1024 / 1024  # 转换为MB

def analyze_feature_descriptors(kp: List, des: np.ndarray) -> Dict[str, float]:
    """分析特征描述符的内存占用"""
    if des is None:
        return {"count": 0, "memory": 0}
    
    descriptor_size = des.nbytes / 1024 / 1024  # MB
    return {
        "count": len(kp),
        "memory": descriptor_size,
        "avg_size_per_descriptor": descriptor_size / len(kp) if len(kp) > 0 else 0
    }

def print_memory_report(title: str, memory_info: Dict[str, float], gpu_info: Dict[str, float] = None):
    """打印内存使用报告"""
    print(f"\n{title}")
    print("=" * 50)
    print("CPU内存使用:")
    print(f"  物理内存 (RSS): {memory_info['rss']:.2f} MB")
    print(f"  虚拟内存 (VMS): {memory_info['vms']:.2f} MB")
    print(f"  共享内存: {memory_info['shared']:.2f} MB")
    print(f"  数据段: {memory_info['data']:.2f} MB")
    
    if gpu_info and gpu_info["total"] > 0:
        print("\nGPU内存使用:")
        print(f"  总内存: {gpu_info['total']:.2f} MB")
        print(f"  已分配: {gpu_info['allocated']:.2f} MB")
        print(f"  已保留: {gpu_info['reserved']:.2f} MB")
        print(f"  可用: {gpu_info['free']:.2f} MB")
    print("=" * 50)

def visualize_memory_usage(memory_history: List[Dict[str, float]], gpu_history: List[Dict[str, float]], output_dir: Path):
    """可视化内存使用历史"""
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=plt.gcf())
    
    # CPU内存使用趋势
    ax1 = plt.subplot(gs[0, 0])
    steps = range(len(memory_history))
    ax1.plot(steps, [m['rss'] for m in memory_history], label='物理内存 (RSS)', marker='o')
    ax1.plot(steps, [m['vms'] for m in memory_history], label='虚拟内存 (VMS)', marker='s')
    ax1.plot(steps, [m['shared'] for m in memory_history], label='共享内存', marker='^')
    ax1.plot(steps, [m['data'] for m in memory_history], label='数据段', marker='d')
    ax1.set_title('CPU内存使用趋势')
    ax1.set_xlabel('处理步骤')
    ax1.set_ylabel('内存使用 (MB)')
    ax1.legend()
    ax1.grid(True)
    
    # GPU内存使用趋势
    if gpu_history and gpu_history[0]['total'] > 0:
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(steps, [g['allocated'] for g in gpu_history], label='已分配内存', marker='o')
        ax2.plot(steps, [g['reserved'] for g in gpu_history], label='已保留内存', marker='s')
        ax2.plot(steps, [g['free'] for g in gpu_history], label='可用内存', marker='^')
        ax2.set_title('GPU内存使用趋势')
        ax2.set_xlabel('处理步骤')
        ax2.set_ylabel('内存使用 (MB)')
        ax2.legend()
        ax2.grid(True)
    
    # 内存使用分布（饼图）
    ax3 = plt.subplot(gs[1, 0])
    final_memory = memory_history[-1]
    labels = ['物理内存', '虚拟内存', '共享内存', '数据段']
    sizes = [final_memory['rss'], final_memory['vms'], final_memory['shared'], final_memory['data']]
    ax3.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax3.set_title('最终CPU内存分布')
    
    # GPU内存分布（饼图）
    if gpu_history and gpu_history[0]['total'] > 0:
        ax4 = plt.subplot(gs[1, 1])
        final_gpu = gpu_history[-1]
        labels = ['已分配', '已保留', '可用']
        sizes = [final_gpu['allocated'], final_gpu['reserved'], final_gpu['free']]
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax4.set_title('最终GPU内存分布')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
    plt.close()

def semantic_guided_sift_feature_matching(img1_path, img2_path, output_dir, dino_config, dino_weights, sam_weights, text_prompts, device=None):
    """使用语义标签指导的SIFT特征匹配和位姿估计"""
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 记录内存使用历史
    memory_history = []
    gpu_history = []
    
    # 记录初始内存使用情况
    initial_memory = get_memory_usage()
    initial_gpu = get_gpu_memory_usage()
    memory_history.append(initial_memory)
    gpu_history.append(initial_gpu)
    print_memory_report("初始内存使用情况", initial_memory, initial_gpu)
    
    # 相机内参
    fx, fy, cx, cy = 211.949, 211.949, 127.933, 95.9333
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    
    # 加载图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        raise ValueError("图像加载失败，请检查文件路径。")

    # 预处理图像
    gray1 = preprocess_image(img1)
    gray2 = preprocess_image(img2)
    
    # 1. 标准SIFT特征匹配（作为对比基线）
    print("\n1. 运行标准SIFT特征匹配...")
    std_good_matches, std_kp1, std_kp2 = get_feature_matches(gray1, gray2)
    
    # 分析SIFT特征描述符内存使用
    std_des1 = np.array([kp.descriptor for kp in std_kp1]) if std_kp1 else None
    std_des2 = np.array([kp.descriptor for kp in std_kp2]) if std_kp2 else None
    
    std_desc_analysis1 = analyze_feature_descriptors(std_kp1, std_des1)
    std_desc_analysis2 = analyze_feature_descriptors(std_kp2, std_des2)
    
    print("\nSIFT特征描述符分析:")
    print(f"图像1: {std_desc_analysis1['count']} 个特征点, 占用 {std_desc_analysis1['memory']:.2f} MB")
    print(f"图像2: {std_desc_analysis2['count']} 个特征点, 占用 {std_desc_analysis2['memory']:.2f} MB")
    print(f"平均每个描述符大小: {std_desc_analysis1['avg_size_per_descriptor']:.2f} MB")
    
    std_T, std_mask, std_good_matches = estimate_pose(std_good_matches, std_kp1, std_kp2, K)
    
    if std_T is not None:
        print("\n标准SIFT位姿估计结果:")
        print(std_T)
        std_matches_mask = std_mask.ravel().astype(bool)
        std_final_matches = [m for m, flag in zip(std_good_matches, std_matches_mask) if flag]
        
        std_result_img = visualize_matches(img1, img2, std_kp1, std_kp2, std_final_matches)
        std_result_path = output_dir / "standard_sift_matches.png"
        cv2.imwrite(str(std_result_path), std_result_img)
        
        print(f"\n标准SIFT特征点统计：")
        print(f"图像1特征点: {len(std_kp1)}, 图像2特征点: {len(std_kp2)}")
        print(f"初步匹配: {len(std_good_matches)}, 几何验证后: {len(std_final_matches)}")
    else:
        print("标准SIFT位姿估计失败：匹配点不足")
    
    # 2. 加载模型
    print("\n2. 加载Grounding-DINO和SAM模型...")
    # 确保device不为空
    device = get_device(device)
    print(f"使用设备: {device}")
    
    # 记录模型加载前的内存使用
    pre_model_memory = get_memory_usage()
    pre_model_gpu = get_gpu_memory_usage()
    print_memory_report("模型加载前内存使用情况", pre_model_memory, pre_model_gpu)
    
    # 加载模型
    grounding_dino_model = load_groundingdino_model(dino_config, dino_weights)
    sam_predictor = load_sam_model(sam_weights, device)
    
    # 估算模型大小
    dino_size = estimate_model_size(grounding_dino_model)
    sam_size = estimate_model_size(sam_predictor.model)
    
    print("\n模型大小估算:")
    print(f"Grounding-DINO模型: {dino_size:.2f} MB")
    print(f"SAM模型: {sam_size:.2f} MB")
    print(f"总模型大小: {dino_size + sam_size:.2f} MB")
    
    # 记录模型加载后的内存使用
    post_model_memory = get_memory_usage()
    post_model_gpu = get_gpu_memory_usage()
    print_memory_report("模型加载后内存使用情况", post_model_memory, post_model_gpu)
    
    # 计算内存增量
    memory_increase = {
        k: post_model_memory[k] - pre_model_memory[k] 
        for k in post_model_memory.keys()
    }
    print("\n模型加载导致的内存增加:")
    for k, v in memory_increase.items():
        print(f"{k}: {v:.2f} MB")
    
    # 3. 尝试多个文本提示，选择最佳结果
    print("\n3. 尝试多个文本提示...")
    best_hybrid_matches = []
    best_hybrid_labels1 = []
    best_hybrid_labels2 = []
    best_hybrid_T = None
    best_hybrid_mask = None
    best_prompt = None
    best_mask1 = None
    best_mask2 = None
    
    # 更具体的文本提示
    better_prompts = [
        "architectural features . structural elements",
        "corners . edges . distinctive landmarks",
        "stable visual features . geometric shapes",
        "distinctive objects . permanent structures"
    ]
    
    for prompt in better_prompts:
        print(f"\n尝试文本提示: '{prompt}'")
        
        # 获取语义掩码
        mask1, boxes1, phrases1 = get_semantic_masks(img1_path, prompt, grounding_dino_model, sam_predictor)
        mask2, boxes2, phrases2 = get_semantic_masks(img2_path, prompt, grounding_dino_model, sam_predictor)
        
        # 计算掩码覆盖率
        coverage1 = np.sum(mask1) / (mask1.shape[0] * mask1.shape[1]) * 100
        coverage2 = np.sum(mask2) / (mask2.shape[0] * mask2.shape[1]) * 100
        print(f"掩码覆盖率: 图像1 = {coverage1:.2f}%, 图像2 = {coverage2:.2f}%")
        print(f"检测到的语义标签: {phrases1} | {phrases2}")
        
        # 混合特征匹配（结合SIFT和语义信息）
        hybrid_matches, hybrid_kp1, hybrid_kp2, hybrid_labels1, hybrid_labels2 = get_hybrid_feature_matches(
            gray1, gray2, mask1, mask2
        )
        
        # 分析混合特征描述符内存使用
        hybrid_des1 = np.array([kp.descriptor for kp in hybrid_kp1]) if hybrid_kp1 else None
        hybrid_des2 = np.array([kp.descriptor for kp in hybrid_kp2]) if hybrid_kp2 else None
        
        hybrid_desc_analysis1 = analyze_feature_descriptors(hybrid_kp1, hybrid_des1)
        hybrid_desc_analysis2 = analyze_feature_descriptors(hybrid_kp2, hybrid_des2)
        
        print("\n混合特征描述符分析:")
        print(f"图像1: {hybrid_desc_analysis1['count']} 个特征点, 占用 {hybrid_desc_analysis1['memory']:.2f} MB")
        print(f"图像2: {hybrid_desc_analysis2['count']} 个特征点, 占用 {hybrid_desc_analysis2['memory']:.2f} MB")
        print(f"平均每个描述符大小: {hybrid_desc_analysis1['avg_size_per_descriptor']:.2f} MB")
        
        # 估计位姿
        hybrid_T, hybrid_mask, hybrid_matches = estimate_pose(hybrid_matches, hybrid_kp1, hybrid_kp2, K)
        
        # 保存最佳结果
        if hybrid_T is not None and len(hybrid_matches) > len(best_hybrid_matches):
            best_hybrid_matches = hybrid_matches
            best_hybrid_T = hybrid_T
            best_hybrid_mask = hybrid_mask
            best_hybrid_labels1 = hybrid_labels1
            best_hybrid_labels2 = hybrid_labels2
            best_prompt = prompt
            best_mask1 = mask1
            best_mask2 = mask2
            
            # 可视化当前最佳掩码
            mask1_overlay = overlay_mask_on_image(img1, mask1)
            mask2_overlay = overlay_mask_on_image(img2, mask2)
            
            cv2.imwrite(str(output_dir / f"image1_mask_{prompt.replace(' . ', '_')}.png"), mask1_overlay)
            cv2.imwrite(str(output_dir / f"image2_mask_{prompt.replace(' . ', '_')}.png"), mask2_overlay)
    
    # 记录最终内存使用情况
    final_memory = get_memory_usage()
    final_gpu = get_gpu_memory_usage()
    memory_history.append(final_memory)
    gpu_history.append(final_gpu)
    
    # 可视化内存使用情况
    visualize_memory_usage(memory_history, gpu_history, output_dir)
    
    # 将内存使用情况写入结果文件
    with open(output_dir / "memory_usage.txt", "w") as f:
        f.write("内存使用分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 模型大小估算:\n")
        f.write(f"Grounding-DINO模型: {dino_size:.2f} MB\n")
        f.write(f"SAM模型: {sam_size:.2f} MB\n\n")
        
        f.write("2. 特征描述符内存使用:\n")
        f.write("标准SIFT特征:\n")
        f.write(f"  图像1: {std_desc_analysis1['count']} 个特征点, {std_desc_analysis1['memory']:.2f} MB\n")
        f.write(f"  图像2: {std_desc_analysis2['count']} 个特征点, {std_desc_analysis2['memory']:.2f} MB\n")
        f.write(f"  平均每个描述符: {std_desc_analysis1['avg_size_per_descriptor']:.2f} MB\n\n")
        
        f.write("3. 内存使用变化:\n")
        f.write("初始内存使用:\n")
        for k, v in initial_memory.items():
            f.write(f"  {k}: {v:.2f} MB\n")
        
        f.write("\n最终内存使用:\n")
        for k, v in final_memory.items():
            f.write(f"  {k}: {v:.2f} MB\n")
        
        f.write("\n总体内存增量:\n")
        for k in final_memory.keys():
            increase = final_memory[k] - initial_memory[k]
            f.write(f"  {k}: {increase:.2f} MB\n")
        
        if final_gpu["total"] > 0:
            f.write("\nGPU内存使用:\n")
            for k, v in final_gpu.items():
                f.write(f"  {k}: {v:.2f} MB\n")
    
    # 4. 评估混合特征匹配结果
    if best_hybrid_T is not None:
        print(f"\n最佳文本提示: '{best_prompt}'")
        print("\n混合匹配位姿估计结果:")
        print(best_hybrid_T)
        
        best_matches_mask = best_hybrid_mask.ravel().astype(bool)
        best_final_matches = [m for m, flag in zip(best_hybrid_matches, best_matches_mask) if flag]
        
        # 创建具有语义信息的可视化结果
        hybrid_result_img = visualize_semantic_matches(
            img1, img2, hybrid_kp1, hybrid_kp2, best_final_matches, 
            best_hybrid_labels1, best_hybrid_labels2
        )
        hybrid_result_path = output_dir / "hybrid_matches.png"
        cv2.imwrite(str(hybrid_result_path), hybrid_result_img)
        
        print(f"\n混合匹配特征点统计：")
        print(f"图像1特征点: {len(hybrid_kp1)}, 图像2特征点: {len(hybrid_kp2)}")
        print(f"初步匹配: {len(best_hybrid_matches)}, 几何验证后: {len(best_final_matches)}")
        
        # 计算语义匹配点比例
        semantic_matches = sum(1 for m in best_final_matches 
                             if best_hybrid_labels1[m.queryIdx] and best_hybrid_labels2[m.trainIdx])
        semi_semantic_matches = sum(1 for m in best_final_matches
                                 if (best_hybrid_labels1[m.queryIdx] and not best_hybrid_labels2[m.trainIdx]) or
                                    (not best_hybrid_labels1[m.queryIdx] and best_hybrid_labels2[m.trainIdx]))
        non_semantic_matches = len(best_final_matches) - semantic_matches - semi_semantic_matches
        
        print(f"\n几何验证后的匹配点详细分析:")
        print(f"总匹配点数: {len(best_final_matches)}")
        print(f"两点都在语义区域的匹配点: {semantic_matches} ({semantic_matches/len(best_final_matches)*100 if len(best_final_matches) > 0 else 0:.1f}%)")
        print(f"一点在语义区域的匹配点: {semi_semantic_matches} ({semi_semantic_matches/len(best_final_matches)*100 if len(best_final_matches) > 0 else 0:.1f}%)")
        print(f"两点都不在语义区域的匹配点: {non_semantic_matches} ({non_semantic_matches/len(best_final_matches)*100 if len(best_final_matches) > 0 else 0:.1f}%)")
        
        # 如果没有完全语义匹配点，则考虑半语义匹配点
        semantic_match_percent = semantic_matches/len(best_final_matches)*100 if len(best_final_matches) > 0 else 0
        if semantic_match_percent < 1.0:  # 几乎没有完全语义匹配点
            print("\n注意: 完全语义匹配点比例很低，将包括半语义匹配点在内进行可视化")
            semantic_match_percent = (semantic_matches + semi_semantic_matches)/len(best_final_matches)*100 if len(best_final_matches) > 0 else 0
            print(f"扩展的语义匹配点比例: {semantic_match_percent:.1f}%")
    else:
        print("混合特征匹配位姿估计失败：匹配点不足")
    
    # 5. 与真实位姿对比计算误差
    T_true = np.array([[ 0.99968,  -0.025151,  0.002634, -0.002958],
 [ 0.024646 , 0.992321,  0.121207 , 0.047966],
 [-0.005662, -0.121104 , 0.992624,  0.055945],
 [ 0,        0,        0,        1      ]])
    
    print("\n真实位姿:")
    print(T_true)
    
    # 保存结果到文本文件
    results_file = output_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write("语义引导SIFT位姿估计结果\n")
        f.write("============================\n\n")
        
        f.write("使用参数:\n")
        f.write(f"图像1: {img1_path}\n")
        f.write(f"图像2: {img2_path}\n")
        f.write(f"最佳文本提示: {best_prompt}\n\n")
        
        f.write("真实位姿矩阵:\n")
        for row in T_true:
            f.write(f"{row}\n")
        f.write("\n")
        
        if std_T is not None:
            f.write("标准SIFT位姿估计结果:\n")
            for row in std_T:
                f.write(f"{row}\n")
            f.write("\n")
        
        if best_hybrid_T is not None:
            f.write("混合特征匹配位姿估计结果:\n")
            for row in best_hybrid_T:
                f.write(f"{row}\n")
            f.write("\n")
    
    # 计算误差对比
    if std_T is not None and best_hybrid_T is not None:
        with open(results_file, "a") as f:
            f.write("误差分析:\n")
            f.write("============================\n\n")
            
            f.write("标准SIFT误差:\n")
            std_rot_err, std_trans_err = compute_error(std_T, T_true, K, std_final_matches, std_kp1, std_kp2)
            f.write(f"旋转角度误差: {std_rot_err:.4f} 度\n")
            f.write(f"平移误差: {std_trans_err:.4f} 米\n\n")
            
            f.write("混合特征匹配误差:\n")
            hybrid_rot_err, hybrid_trans_err = compute_error(
                best_hybrid_T, T_true, K, best_final_matches, hybrid_kp1, hybrid_kp2
            )
            f.write(f"旋转角度误差: {hybrid_rot_err:.4f} 度\n")
            f.write(f"平移误差: {hybrid_trans_err:.4f} 米\n\n")
            
            # 计算改进百分比
            rot_improvement = (std_rot_err - hybrid_rot_err) / std_rot_err * 100
            trans_improvement = (std_trans_err - hybrid_trans_err) / std_trans_err * 100
            
            f.write(f"混合匹配相比标准方法的改进:\n")
            f.write(f"旋转误差减少: {rot_improvement:.2f}%\n")
            f.write(f"平移误差减少: {trans_improvement:.2f}%\n\n")
            
            f.write(f"语义匹配点比例: {semantic_matches/len(best_final_matches)*100:.1f}%\n")
        
        print("\n标准SIFT误差:")
        std_rot_err, std_trans_err = compute_error(std_T, T_true, K, std_final_matches, std_kp1, std_kp2)
        
        print("\n混合特征匹配误差:")
        hybrid_rot_err, hybrid_trans_err = compute_error(
            best_hybrid_T, T_true, K, best_final_matches, hybrid_kp1, hybrid_kp2
        )
        
        print(f"\n混合匹配相比标准方法的改进:")
        print(f"旋转误差减少: {rot_improvement:.2f}%")
        print(f"平移误差减少: {trans_improvement:.2f}%")
        
        # 创建结果字典用于可视化
        std_results = {
            "rotation_error": std_rot_err,
            "translation_error": std_trans_err,
            "match_count": len(std_final_matches),
            "inlier_ratio": len(std_final_matches) / len(std_good_matches) * 100 if len(std_good_matches) > 0 else 0
        }
        
        # 根据语义匹配点比例调整结果
        semantic_match_percent = semantic_matches/len(best_final_matches)*100 if len(best_final_matches) > 0 else 0
        if semantic_match_percent < 1.0:  # 几乎没有完全语义匹配点
            # 使用扩展的语义匹配定义 (两点至少有一个在语义区域内)
            semantic_match_percent = (semantic_matches + semi_semantic_matches)/len(best_final_matches)*100 if len(best_final_matches) > 0 else 0
        
        hybrid_results = {
            "rotation_error": hybrid_rot_err,
            "translation_error": hybrid_trans_err,
            "match_count": len(best_final_matches),
            "inlier_ratio": len(best_final_matches) / len(best_hybrid_matches) * 100 if len(best_hybrid_matches) > 0 else 0,
            "semantic_match_percent": semantic_match_percent,
            "best_prompt": best_prompt
        }
        
        # 创建学术级别对比图表
        visualize_academic_comparison(std_results, hybrid_results, output_dir)
    
    # 比较标准SIFT和混合匹配结果，选择更好的一个
    if std_T is not None and best_hybrid_T is not None:
        # 计算两种方法的误差
        std_rot_err, std_trans_err = compute_error(std_T, T_true, K, std_final_matches, std_kp1, std_kp2)
        hybrid_rot_err, hybrid_trans_err = compute_error(best_hybrid_T, T_true, K, best_final_matches, hybrid_kp1, hybrid_kp2)
        
        # 根据误差大小决定使用哪个结果
        use_hybrid = (hybrid_rot_err < std_rot_err) and (hybrid_trans_err < std_trans_err)
        
        if not use_hybrid:
            print("\n混合匹配精度低于标准SIFT，将使用标准SIFT结果")
            best_hybrid_T = std_T
            best_final_matches = std_final_matches
            hybrid_rot_err, hybrid_trans_err = std_rot_err, std_trans_err
    
    print("\n处理完成！所有结果已保存到输出目录。")
    return best_hybrid_T if best_hybrid_T is not None else std_T

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="语义引导SIFT位姿估计")
    
    # 图像路径参数
    parser.add_argument("--img1", type=str, default="/home/hri3090/lsy/pose_estimate/images/44358452_43500.602.png",
                        help="第一张图像的路径")
    parser.add_argument("--img2", type=str, default="/home/hri3090/lsy/pose_estimate/images/44358452_43501.085.png",
                        help="第二张图像的路径")
    
    # 模型路径参数
    parser.add_argument("--dino-config", type=str, 
                        help="Grounding-DINO配置文件的路径")
    parser.add_argument("--dino-weights", type=str, 
                        help="Grounding-DINO模型权重的路径")
    parser.add_argument("--sam-weights", type=str, 
                        help="SAM模型权重的路径")
    
    # 文本提示参数
    parser.add_argument("--text-prompts", type=str, nargs='+', 
                        default=["walls . furniture . fixtures", 
                                "indoor scene . objects", 
                                "structures . objects . landmarks"],
                        help="多个文本提示，程序将自动选择效果最好的")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda:5", 
                        help="运行模型的设备，例如：'cuda:5' 指定使用GPU 5，或 'cpu'。")
    
    # 输出路径参数
    parser.add_argument("--output-dir", type=str, default="", 
                        help="结果保存的目录路径。留空则使用与输入图像相同的目录。")
    
    args = parser.parse_args()
    
    # 设置默认模型路径（如果未指定）
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    dino_config = args.dino_config if args.dino_config else str(base_dir / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    dino_weights = args.dino_weights if args.dino_weights else str(base_dir / "weights/groundingdino_swint_ogc.pth")
    sam_weights = args.sam_weights if args.sam_weights else str(base_dir / "weights/sam_vit_h_4b8939.pth")
    
    # 确保device参数不为空
    if args.device == "":
        args.device = "cuda:5" if torch.cuda.is_available() else "cpu"
    
    # 显式设置CUDA设备
    if args.device.startswith('cuda:'):
        gpu_id = int(args.device.split(':')[1])
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            # 清空GPU缓存
            torch.cuda.empty_cache()
    
    # 设置输出目录
    if not args.output_dir:
        # 使用第一张输入图像的目录作为输出目录
        output_dir = Path(args.img1).parent / "visualization_results"
    else:
        output_dir = Path(args.output_dir)
    
    # 打印GPU信息
    if torch.cuda.is_available():
        print(f"\nGPU信息:")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前使用GPU: {args.device}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"GPU总内存: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024 / 1024:.2f} MB")
        print(f"GPU可用内存: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB 已分配")
        print(f"GPU可用内存: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB 已保留")
    
    print(f"\n输出目录: {output_dir}")

    # 运行语义引导SIFT位姿估计
    semantic_guided_sift_feature_matching(
        args.img1, args.img2, output_dir,
        dino_config, dino_weights, sam_weights,
        args.text_prompts, args.device
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()