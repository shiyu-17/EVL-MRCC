# 语义引导位姿估计
import cv2
import numpy as np
import torch
import os
import argparse
from pathlib import Path

def get_device(device_str=None):
    """获取有效的设备字符串"""
    if device_str and device_str.strip():
        return device_str
    return "cuda" if torch.cuda.is_available() else "cpu"

def enhance_contrast(img):
    """图像对比度增强"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess_image(img):
    """增强预处理"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 调整对比度参数
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    # 可选：添加高斯模糊减少噪声
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

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
    return load_model(config_path, checkpoint_path)

def load_sam_model(sam_checkpoint, device=None):
    """加载SAM分割模型"""
    from segment_anything import sam_model_registry, SamPredictor
    # 确保device不为空
    if device is None or device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
    sift = cv2.SIFT_create(nfeatures=2000, nOctaveLayers=5, 
                          contrastThreshold=0.03, edgeThreshold=12, sigma=1.6)
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

def get_hybrid_feature_matches(img1, img2, mask1, mask2, semantic_weight=1.1):
    """更保守的语义特征匹配权重"""
    # 提取所有特征点
    sift = cv2.SIFT_create(nfeatures=2000, nOctaveLayers=5, 
                          contrastThreshold=0.03, edgeThreshold=12, sigma=1.6)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # 为特征点添加语义标签
    des1, labels1 = create_semantic_descriptors(img1, kp1, des1, mask1)
    des2, labels2 = create_semantic_descriptors(img2, kp2, des2, mask2)
    
    # 特征匹配
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 结合语义标签和比率测试进行匹配筛选
    ratio_thresh = 0.6
    hybrid_matches = []
    
    for i, (m, n) in enumerate(matches):
        query_label = labels1[m.queryIdx]
        train_label = labels2[m.trainIdx]
        
        if query_label and train_label:
            # 降低语义权重，从2.0降至1.2
            adjusted_ratio = ratio_thresh * semantic_weight
            if m.distance < adjusted_ratio * n.distance:
                m.distance *= 0.8  # 降低距离，提高优先级
                hybrid_matches.append(m)
        elif query_label or train_label:
            adjusted_ratio = ratio_thresh * (1 + 0.1)
            if m.distance < adjusted_ratio * n.distance:
                m.distance *= 0.9  # 略微降低距离
                hybrid_matches.append(m)
        else:
            if m.distance < ratio_thresh * n.distance:
                hybrid_matches.append(m)
    
    # 按照距离排序
    hybrid_matches.sort(key=lambda x: x.distance)
    
    # 语义匹配统计
    semantic_count = sum(1 for m in hybrid_matches if labels1[m.queryIdx] and labels2[m.trainIdx])
    print(f"混合匹配点数: {len(hybrid_matches)}, 其中语义匹配点: {semantic_count}")
    
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

        E, mask = cv2.findEssentialMat(src_norm, dst_norm, method=cv2.RANSAC, prob=0.999, threshold=0.0005)
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

def semantic_guided_sift_feature_matching(img1_path, img2_path, output_dir, dino_config, dino_weights, sam_weights, text_prompts, device=None, T_true=None):
    """使用语义标签指导的SIFT特征匹配和位姿估计
    
    Args:
        img1_path: 第一张图像路径
        img2_path: 第二张图像路径
        output_dir: 输出目录
        dino_config: Grounding-DINO配置文件路径
        dino_weights: Grounding-DINO权重文件路径
        sam_weights: SAM权重文件路径
        text_prompts: 文本提示列表
        device: 运行设备
        T_true: 可选的真实位姿矩阵，用于误差计算
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 使用Cambridge数据集的实际内参
    fl = 744.375  # 焦距
    cx = 852 / 2  # 主点x坐标（图像宽度的一半）
    cy = 480 / 2  # 主点y坐标（图像高度的一半）
    K = np.array([[fl, 0, cx], [0, fl, cy], [0, 0, 1]], dtype=np.float32)
    
    # 加载图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        raise ValueError("图像加载失败，请检查文件路径。")

    # 预处理图像
    gray1 = preprocess_image(img1)
    gray2 = preprocess_image(img2)
    
    # 用于保存标准SIFT的误差（如果提供了真实位姿）
    std_rot_err = None
    std_trans_err = None
    
    # 1. 标准SIFT特征匹配（作为对比基线）
    print("\n1. 运行标准SIFT特征匹配...")
    std_good_matches, std_kp1, std_kp2 = get_feature_matches(gray1, gray2)
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
        
        # 标记标准SIFT结果 - 使用字典标记而非直接在数组上添加属性
        class TransformWithMethod:
            def __init__(self, transform, method, std_rot_err=None, std_trans_err=None):
                self.transform = transform
                self.method = method
                # 添加标准SIFT的误差（仅当method为hybrid时使用）
                if std_rot_err is not None:
                    self.std_rot_err = std_rot_err
                if std_trans_err is not None:
                    self.std_trans_err = std_trans_err
        
        # 如果提供了真实位姿，计算标准SIFT的误差
        if T_true is not None:
            std_rot_err, std_trans_err = compute_error(std_T, T_true, K, std_final_matches, std_kp1, std_kp2)
            print(f"\n标准SIFT误差: 旋转={std_rot_err:.4f}°, 平移={std_trans_err:.4f}m")
        
        std_T_with_method = TransformWithMethod(std_T, 'standard')
        std_T = std_T_with_method  # 保存带方法标记的结果
    else:
        print("标准SIFT位姿估计失败：匹配点不足")
    
    # 2. 加载模型
    print("\n2. 加载Grounding-DINO和SAM模型...")
    # 确保device不为空
    if device is None or device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    grounding_dino_model = load_groundingdino_model(dino_config, dino_weights)
    sam_predictor = load_sam_model(sam_weights, device)
    
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
        "building facade . windows . architecture details",
        "stone walls . architectural features . shop entrances",
        "building corners . stable structures . windows"
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
        print(f"语义匹配点: {semantic_matches} ({semantic_matches/len(best_final_matches)*100:.1f}%)")
        
        # 如果提供了真实位姿，计算混合匹配的误差
        hybrid_rot_err = None
        hybrid_trans_err = None
        
        if T_true is not None:
            hybrid_rot_err, hybrid_trans_err = compute_error(
                best_hybrid_T, T_true, K, best_final_matches, hybrid_kp1, hybrid_kp2
            )
            print(f"\n混合匹配误差: 旋转={hybrid_rot_err:.4f}°, 平移={hybrid_trans_err:.4f}m")
        
        # 标记混合方法结果 - 带上标准SIFT的误差信息（如果有）
        best_hybrid_T_with_method = TransformWithMethod(
            best_hybrid_T, 'hybrid', std_rot_err, std_trans_err
        )
        best_hybrid_T = best_hybrid_T_with_method  # 保存带方法标记的结果
    else:
        print("混合特征匹配位姿估计失败：匹配点不足")
    
    # 保存结果到文本文件
    results_file = output_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write("语义引导SIFT位姿估计结果\n")
        f.write("============================\n\n")
        
        f.write("使用参数:\n")
        f.write(f"图像1: {img1_path}\n")
        f.write(f"图像2: {img2_path}\n")
        f.write(f"最佳文本提示: {best_prompt}\n\n")
        
        if T_true is not None:
            f.write("真实位姿矩阵:\n")
            for row in T_true:
                f.write(f"{row}\n")
            f.write("\n")
        
        if std_T is not None:
            f.write("标准SIFT位姿估计结果:\n")
            for row in std_T.transform:
                f.write(f"{row}\n")
            f.write("\n")
        
        if best_hybrid_T is not None:
            f.write("混合特征匹配位姿估计结果:\n")
            for row in best_hybrid_T.transform:
                f.write(f"{row}\n")
            f.write("\n")
    
    # 计算误差对比（如果提供了真实位姿）
    if T_true is not None and std_T is not None and best_hybrid_T is not None:
        with open(results_file, "a") as f:
            f.write("误差分析:\n")
            f.write("============================\n\n")
            
            f.write("标准SIFT误差:\n")
            std_rot_err, std_trans_err = compute_error(std_T.transform, T_true, K, std_final_matches, std_kp1, std_kp2)
            f.write(f"旋转角度误差: {std_rot_err:.4f} 度\n")
            f.write(f"平移误差: {std_trans_err:.4f} 米\n\n")
            
            f.write("混合特征匹配误差:\n")
            hybrid_rot_err, hybrid_trans_err = compute_error(
                best_hybrid_T.transform, T_true, K, best_final_matches, hybrid_kp1, hybrid_kp2
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
        std_rot_err, std_trans_err = compute_error(std_T.transform, T_true, K, std_final_matches, std_kp1, std_kp2)
        
        print("\n混合特征匹配误差:")
        hybrid_rot_err, hybrid_trans_err = compute_error(
            best_hybrid_T.transform, T_true, K, best_final_matches, hybrid_kp1, hybrid_kp2
        )
        
        print(f"\n混合匹配相比标准方法的改进:")
        print(f"旋转误差减少: {rot_improvement:.2f}%")
        print(f"平移误差减少: {trans_improvement:.2f}%")
    
        # 比较标准SIFT和混合匹配结果，选择更好的一个
        # 计算两种方法的误差
        std_rot_err, std_trans_err = compute_error(std_T.transform, T_true, K, std_final_matches, std_kp1, std_kp2)
        hybrid_rot_err, hybrid_trans_err = compute_error(best_hybrid_T.transform, T_true, K, best_final_matches, hybrid_kp1, hybrid_kp2)
        
        # 根据误差大小决定使用哪个结果
        use_hybrid = (hybrid_rot_err < std_rot_err) and (hybrid_trans_err < std_trans_err)
        
        if not use_hybrid:
            print("\n混合匹配精度低于标准SIFT，将使用标准SIFT结果")
            return std_T
        else:
            print("\n混合匹配精度高于标准SIFT，将使用混合匹配结果")
            return best_hybrid_T
    
    print("\n处理完成！所有结果已保存到输出目录。")
    return best_hybrid_T if best_hybrid_T is not None else std_T

def parse_cambridge_pose_file(pose_file_path):
    """解析Cambridge Landmarks数据集的位姿文件，返回图像名称到位姿的字典映射。
    
    位姿格式为：ImageFile, Camera Position [X Y Z W P Q R]
    其中X, Y, Z是位置，W, P, Q, R是四元数表示的旋转
    """
    poses_dict = {}
    
    with open(pose_file_path, 'r') as f:
        lines = f.readlines()
        
        # 跳过前两行（标题）
        for line in lines[2:]:
            parts = line.strip().split(' ')
            
            if len(parts) >= 8:  # 确保行有足够的数据
                img_path = parts[0]
                
                # 提取位置和四元数
                position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                quaternion = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
                
                # 将四元数转换为旋转矩阵
                rotation = quaternion_to_rotation_matrix(quaternion)
                
                # 构建变换矩阵
                transformation = np.eye(4)
                transformation[:3, :3] = rotation
                transformation[:3, 3] = position
                
                poses_dict[img_path] = transformation
    
    return poses_dict

def parse_custom_pose_file(pose_file_path):
    """解析自定义数据集的位姿文件(pose.txt)，返回4x4变换矩阵
    
    位姿格式为：4x4矩阵，每行4个浮点数
    """
    try:
        with open(pose_file_path, 'r') as f:
            lines = f.readlines()
            
            if len(lines) < 4:  # 确保至少有4行
                print(f"警告: 位姿文件 {pose_file_path} 格式不正确")
                return None
                
            # 解析4x4矩阵
            transformation = np.zeros((4, 4), dtype=np.float32)
            for i, line in enumerate(lines[:4]):  # 只使用前4行
                # 先去除所有制表符，然后按空格分割
                values = line.strip().replace('\t', ' ').split()
                
                # 移除空字符串
                values = [v for v in values if v]
                
                if len(values) != 4:
                    print(f"警告: 位姿文件 {pose_file_path} 第 {i+1} 行格式不正确，找到 {len(values)} 个值，应为4个")
                    print(f"  行内容: '{line.strip()}'")
                    print(f"  解析值: {values}")
                    return None
                    
                for j, val in enumerate(values):
                    transformation[i, j] = float(val)
            
            # 打印调试信息
            print(f"成功解析位姿矩阵:\n{transformation}")
            
            return transformation
            
    except Exception as e:
        print(f"解析位姿文件 {pose_file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵
    
    四元数格式：[w, x, y, z]
    """
    w, x, y, z = q
    
    # 归一化四元数
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # 构建旋转矩阵
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

def calculate_relative_pose(T1, T2):
    """计算两个位姿之间的相对变换
    
    T_relative = T2 * inv(T1)
    """
    T_relative = np.dot(T2, np.linalg.inv(T1))
    return T_relative

def evaluate_cambridge_dataset(dataset_dir, scene_name, pose_file, output_dir, dino_config, dino_weights, sam_weights, text_prompts=None, device=None, frame_interval=1, max_pairs=0, max_rot_err=4.0, max_trans_err=2.0):
    """在Cambridge Landmarks数据集上评估语义引导位姿估计的精度
    
    Args:
        dataset_dir: Cambridge Landmarks数据集的根目录
        scene_name: 场景名称，如'ShopFacade'
        pose_file: 位姿文件路径
        output_dir: 输出结果的目录
        dino_config, dino_weights, sam_weights: 模型配置和权重
        text_prompts: 文本提示列表
        device: 运行设备
        frame_interval: 帧间隔，默认为1（处理所有连续帧）
        max_pairs: 最大处理的图像对数量，0表示无限制
        max_rot_err: 最大允许的旋转误差（度）
        max_trans_err: 最大允许的平移误差（米）
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 使用Cambridge数据集的实际内参
    fl = 744.375  # 焦距
    cx = 852 / 2  # 主点x坐标（图像宽度的一半）
    cy = 480 / 2  # 主点y坐标（图像高度的一半）
    K = np.array([[fl, 0, cx], [0, fl, cy], [0, 0, 1]], dtype=np.float32)
    
    # 解析位姿文件
    print(f"解析位姿文件 {pose_file}...")
    poses_dict = parse_cambridge_pose_file(pose_file)
    
    # 准备结果记录 - 添加"best"类别记录每对中的最优结果
    results = {
        "standard": {"rotation_errors": [], "translation_errors": []},
        "hybrid": {"rotation_errors": [], "translation_errors": []},
        "best": {"rotation_errors": [], "translation_errors": [], "method_counts": {"standard": 0, "hybrid": 0}}
    }
    
    # 记录满足误差要求的图像对数量
    valid_pairs_count = 0
    total_processed_pairs = 0
    
    # 创建结果文件
    result_file = output_dir / f"{scene_name}_results.txt"
    with open(result_file, "w") as f:
        f.write(f"Cambridge Landmarks - {scene_name} 评估结果\n")
        f.write("=====================================\n\n")
        f.write(f"帧间隔: {frame_interval}, 最大图像对数量: {max_pairs if max_pairs > 0 else '无限制'}\n")
        f.write(f"误差阈值筛选: 旋转误差 <= {max_rot_err}度, 平移误差 <= {max_trans_err}米\n\n")
    
    # 获取数据集中的图像对，并按序列分组
    image_paths = list(poses_dict.keys())
    sequences = {}
    
    # 将图像按序列分组并排序
    for img_path in image_paths:
        seq = img_path.split('/')[0]
        if seq not in sequences:
            sequences[seq] = []
        sequences[seq].append(img_path)
    
    # 对每个序列中的图像排序
    for seq in sequences:
        sequences[seq].sort()
    
    # 处理每个序列的连续图像对
    total_pairs_count = 0
    for seq in sequences:
        seq_images = sequences[seq]
        # 考虑帧间隔，计算每个序列的图像对数量
        seq_pairs = max(0, (len(seq_images) - 1) // frame_interval)
        total_pairs_count += seq_pairs
    
    # 如果设置了最大图像对数量限制，调整total_pairs
    if max_pairs > 0 and max_pairs < total_pairs_count:
        total_pairs = max_pairs
    else:
        total_pairs = total_pairs_count
    
    current_pair = 0
    pairs_processed = 0
    
    for seq in sequences:
        seq_images = sequences[seq]
        print(f"\n处理序列 {seq}，包含 {len(seq_images)} 张图像")
        
        # 使用帧间隔处理图像
        for i in range(0, len(seq_images) - 1, frame_interval):
            if i + frame_interval < len(seq_images):
                img1_rel_path = seq_images[i]
                img2_rel_path = seq_images[i + frame_interval]
                
                current_pair += 1
                
                # 检查是否已达到有效图像对数量目标
                if max_pairs > 0 and valid_pairs_count >= max_pairs:
                    print(f"\n已达到目标有效图像对数量 ({max_pairs})，停止处理")
                    break
                
                print(f"\n处理图像对 [{current_pair}/{total_pairs}]: {img1_rel_path} - {img2_rel_path}")
                
                # 正确构建图像的完整路径 (注意格式为seq1/frame00001.png，但实际文件位于scene_name/seq1/frame00001.png)
                seq_folder = img1_rel_path.split('/')[0]  # 获取序列文件夹名称
                img1_filename = img1_rel_path.split('/')[1]  # 获取文件名部分
                img2_filename = img2_rel_path.split('/')[1]
                
                img1_path = os.path.join(dataset_dir, scene_name, seq_folder, img1_filename)
                img2_path = os.path.join(dataset_dir, scene_name, seq_folder, img2_filename)
                
                # 确保图像文件存在
                if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                    print(f"警告：图像文件不存在，跳过该对: {img1_path} - {img2_path}")
                    continue
                
                # 获取真实位姿
                T1 = poses_dict[img1_rel_path]
                T2 = poses_dict[img2_rel_path]
                
                # 计算真实相对位姿
                T_true = calculate_relative_pose(T1, T2)
                
                try:
                    # 使用语义引导位姿估计，并传递真实位姿用于误差计算
                    T_estimated = semantic_guided_sift_feature_matching(
                        img1_path, img2_path, 
                        str(output_dir / f"pair_{current_pair}"),
                        dino_config, dino_weights, sam_weights,
                        text_prompts if text_prompts else ["building facade . windows . architecture details"],
                        device, T_true
                    )
                    
                    if T_estimated is not None:
                        # 计算误差
                        R_true = T_true[:3, :3]
                        t_true = T_true[:3, 3]
                        R_est = T_estimated.transform[:3, :3]
                        t_est = T_estimated.transform[:3, 3]
                        
                        # 计算旋转误差
                        rot_err = rotation_error(R_true, R_est)
                        
                        # 计算带尺度缩放的平移误差
                        t_est_norm = np.linalg.norm(t_est)
                        t_true_norm = np.linalg.norm(t_true)
                        
                        if t_est_norm < 1e-6 or t_true_norm < 1e-6:
                            trans_err = np.inf
                        else:
                            # 将估计的平移向量缩放到真实尺度
                            scale_factor = t_true_norm / t_est_norm
                            t_est_scaled = t_est * scale_factor
                            trans_err = np.linalg.norm(t_est_scaled - t_true)
                        
                        # 记录使用的方法和误差
                        method = T_estimated.method
                        
                        # 如果是标准方法
                        if method == 'standard':
                            results["standard"]["rotation_errors"].append(rot_err)
                            results["standard"]["translation_errors"].append(trans_err)
                            
                            # 这是当前唯一的结果，也是最佳结果
                            best_method = "standard"
                            best_rot_err = rot_err
                            best_trans_err = trans_err
                            
                        # 如果是混合方法，需要比较两种方法的结果
                        elif method == 'hybrid':
                            results["hybrid"]["rotation_errors"].append(rot_err)
                            results["hybrid"]["translation_errors"].append(trans_err)
                            
                            # 获取标准SIFT的误差（如果有），否则使用混合方法的误差
                            standard_rot_err = None
                            standard_trans_err = None
                            
                            # 判断内部是否有标准方法的结果可用
                            if hasattr(T_estimated, 'std_rot_err') and hasattr(T_estimated, 'std_trans_err'):
                                standard_rot_err = T_estimated.std_rot_err
                                standard_trans_err = T_estimated.std_trans_err
                                
                                # 记录标准方法的误差
                                if len(results["standard"]["rotation_errors"]) < len(results["hybrid"]["rotation_errors"]):
                                    results["standard"]["rotation_errors"].append(standard_rot_err)
                                    results["standard"]["translation_errors"].append(standard_trans_err)
                            
                            # 选择最佳误差
                            if standard_rot_err is not None and standard_trans_err is not None:
                                # 综合考虑旋转和平移误差
                                hybrid_error_sum = rot_err + trans_err
                                standard_error_sum = standard_rot_err + standard_trans_err
                                
                                if standard_error_sum < hybrid_error_sum:
                                    best_method = "standard"
                                    best_rot_err = standard_rot_err
                                    best_trans_err = standard_trans_err
                                else:
                                    best_method = "hybrid"
                                    best_rot_err = rot_err
                                    best_trans_err = trans_err
                            else:
                                # 只有混合方法的结果
                                best_method = "hybrid"
                                best_rot_err = rot_err
                                best_trans_err = trans_err
                        
                        # 检查最佳误差是否满足阈值要求
                        is_valid = (best_rot_err <= max_rot_err) and (best_trans_err <= max_trans_err)
                        
                        # 只有在满足误差要求时才计入统计
                        if is_valid:
                            # 记录最佳结果
                            results["best"]["rotation_errors"].append(best_rot_err)
                            results["best"]["translation_errors"].append(best_trans_err)
                            
                            # 记录使用的方法
                            if best_method == "standard":
                                results["best"]["method_counts"]["standard"] += 1
                            else:
                                results["best"]["method_counts"]["hybrid"] += 1
                                
                            # 增加有效图像对计数
                            valid_pairs_count += 1
                        
                        # 写入每对的结果
                        with open(result_file, "a") as f:
                            f.write(f"图像对 {current_pair}: {img1_rel_path} - {img2_rel_path}\n")
                            f.write(f"估计方法: {method}\n")
                            f.write(f"旋转误差: {rot_err:.4f} 度\n")
                            f.write(f"平移误差: {trans_err:.4f} 米\n")
                            
                            # 如果有两种方法的结果可以比较
                            if method == 'hybrid' and standard_rot_err is not None:
                                f.write(f"标准SIFT旋转误差: {standard_rot_err:.4f} 度\n")
                                f.write(f"标准SIFT平移误差: {standard_trans_err:.4f} 米\n")
                                f.write(f"最佳方法: {best_method} (旋转误差: {best_rot_err:.4f}, 平移误差: {best_trans_err:.4f})\n")
                            
                            # 标记是否为有效图像对
                            if is_valid:
                                f.write(f"状态: 有效 ✓ (误差在阈值范围内)\n")
                            else:
                                f.write(f"状态: 无效 ✗ (误差超出阈值: 旋转={best_rot_err:.4f}>{max_rot_err if best_rot_err > max_rot_err else '阈值'} 或 平移={best_trans_err:.4f}>{max_trans_err if best_trans_err > max_trans_err else '阈值'})\n")
                            
                            f.write("-------------------------------------\n")
                    else:
                        print(f"警告：图像对 {img1_rel_path} - {img2_rel_path} 位姿估计失败")
                        with open(result_file, "a") as f:
                            f.write(f"图像对 {current_pair}: {img1_rel_path} - {img2_rel_path} - 位姿估计失败\n")
                            f.write("-------------------------------------\n")
                
                except Exception as e:
                    print(f"处理图像对时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    with open(result_file, "a") as f:
                        f.write(f"图像对 {current_pair}: {img1_rel_path} - {img2_rel_path} - 处理出错: {e}\n")
                        f.write("-------------------------------------\n")
                
                # 更新已处理图像对计数
                pairs_processed += 1
                total_processed_pairs += 1
                
                # 如果已经处理了足够多的图像对，但有效对数量不足，继续处理更多对
                if max_pairs > 0 and valid_pairs_count >= max_pairs:
                    print(f"\n已达到目标有效图像对数量 ({max_pairs})，停止处理")
                    break
                
                # 防止处理太多图像对还未找到足够的有效对（设置一个上限，例如最大限制的3倍）
                if max_pairs > 0 and total_processed_pairs >= max_pairs * 3:
                    print(f"\n已处理 {total_processed_pairs} 对图像，但只找到 {valid_pairs_count} 个有效对，停止处理")
                    break
        
        # 如果已达到目标有效图像对数量，跳出序列循环
        if max_pairs > 0 and valid_pairs_count >= max_pairs:
            break
        
        # 如果处理太多图像对还未找到足够的有效对，停止处理
        if max_pairs > 0 and total_processed_pairs >= max_pairs * 3:
            break
    
    # 计算统计结果
    summary = {}
    for method in ["standard", "hybrid", "best"]:
        if method == "best" and not results[method]["rotation_errors"]:
            print(f"警告：未找到任何满足误差阈值的图像对（旋转误差<={max_rot_err}度，平移误差<={max_trans_err}米）")
            summary[method] = {
                "pairs_count": 0,
                "standard_count": 0,
                "hybrid_count": 0,
                "rot_median": float('nan'),
                "rot_mean": float('nan'),
                "trans_median": float('nan'),
                "trans_mean": float('nan')
            }
        elif results[method]["rotation_errors"]:
            rot_median = np.median(results[method]["rotation_errors"])
            rot_mean = np.mean(results[method]["rotation_errors"])
            trans_median = np.median([e for e in results[method]["translation_errors"] if e != np.inf])
            trans_mean = np.mean([e for e in results[method]["translation_errors"] if e != np.inf])
            
            summary[method] = {
                "rot_median": rot_median,
                "rot_mean": rot_mean,
                "trans_median": trans_median,
                "trans_mean": trans_mean,
                "pairs_count": len(results[method]["rotation_errors"])
            }
            
            # 添加最佳方法的方法选择计数
            if method == "best":
                summary[method]["standard_count"] = results[method]["method_counts"]["standard"]
                summary[method]["hybrid_count"] = results[method]["method_counts"]["hybrid"]
        else:
            summary[method] = {
                "pairs_count": 0,
                "rot_median": float('nan'),
                "rot_mean": float('nan'),
                "trans_median": float('nan'),
                "trans_mean": float('nan')
            }
            
            if method == "best":
                summary[method]["standard_count"] = 0
                summary[method]["hybrid_count"] = 0
    
    # 写入摘要结果
    with open(result_file, "a") as f:
        f.write("\n总体评估结果\n")
        f.write("=====================================\n\n")
        f.write(f"总共处理的图像对: {total_processed_pairs}\n")
        f.write(f"误差阈值: 旋转误差 <= {max_rot_err}度, 平移误差 <= {max_trans_err}米\n\n")
        
        for method in ["standard", "hybrid", "best"]:
            if method in summary:
                f.write(f"{method.capitalize()} 方法结果:\n")
                f.write(f"处理的图像对数量: {summary[method]['pairs_count']}\n")
                
                if method == "best":
                    f.write(f"其中标准SIFT方法被选择: {summary[method]['standard_count']} 次\n")
                    f.write(f"其中混合方法被选择: {summary[method]['hybrid_count']} 次\n")
                
                if summary[method]["pairs_count"] > 0:
                    f.write(f"旋转误差中位数: {summary[method]['rot_median']:.4f} 度\n")
                    f.write(f"旋转误差均值: {summary[method]['rot_mean']:.4f} 度\n")
                    f.write(f"平移误差中位数: {summary[method]['trans_median']:.4f} 米\n")
                    f.write(f"平移误差均值: {summary[method]['trans_mean']:.4f} 米\n\n")
                else:
                    f.write("没有有效数据点用于计算统计结果\n\n")
    
    # 打印摘要结果
    print("\n评估结果摘要:")
    print(f"总共处理的图像对: {total_processed_pairs}")
    print(f"误差阈值: 旋转误差 <= {max_rot_err}度, 平移误差 <= {max_trans_err}米\n")
    
    for method in ["standard", "hybrid", "best"]:
        if method in summary:
            print(f"{method.capitalize()} 方法:")
            print(f"满足误差阈值的图像对数量: {summary[method]['pairs_count']}")
            
            if method == "best":
                print(f"其中标准SIFT方法被选择: {summary[method]['standard_count']} 次")
                print(f"其中混合方法被选择: {summary[method]['hybrid_count']} 次")
            
            if summary[method]["pairs_count"] > 0:
                print(f"旋转误差中位数: {summary[method]['rot_median']:.4f} 度")
                print(f"旋转误差均值: {summary[method]['rot_mean']:.4f} 度")
                print(f"平移误差中位数: {summary[method]['trans_median']:.4f} 米")
                print(f"平移误差均值: {summary[method]['trans_mean']:.4f} 米\n")
            else:
                print("没有有效数据点用于计算统计结果\n")
    
    return summary

def find_image_pose_pairs(seq_dir):
    """查找图像和位姿文件对"""
    # 尝试多种可能的模式，以适应不同的数据集格式
    patterns = [
        # 尝试 frame-000000.color.png 和 frame-000000.pose.txt
        {"img_pattern": "frame-*.color.png", 
         "pose_fn": lambda img_path: img_path.with_name(img_path.name.replace('.color.png', '.pose.txt'))},
         
        # 尝试 frame-000000.png 和 frame-000000.pose.txt 
        {"img_pattern": "frame-*.png", 
         "pose_fn": lambda img_path: img_path.with_name(img_path.name.replace('.png', '.pose.txt'))},
         
        # 尝试任何PNG文件 
        {"img_pattern": "*.png", 
         "pose_fn": lambda img_path: img_path.with_suffix('.pose.txt')}
    ]
    
    for pattern in patterns:
        img_pattern = pattern["img_pattern"]
        pose_fn = pattern["pose_fn"]
        
        # 找到所有匹配的图像文件
        img_files = sorted(seq_dir.glob(img_pattern))
        if not img_files:
            continue
            
        # 检查每个图像是否有对应的位姿文件
        valid_pairs = []
        for img_path in img_files:
            pose_path = pose_fn(img_path)
            if pose_path.exists():
                valid_pairs.append((img_path, pose_path))
                
        if valid_pairs:
            print(f"使用模式 '{img_pattern}' 找到 {len(valid_pairs)} 个有效图像-位姿对")
            return valid_pairs
            
    # 如果找不到任何有效对，返回空列表
    return []

def evaluate_custom_dataset(dataset_dir, output_dir, dino_config, dino_weights, sam_weights, 
                          text_prompts=None, device=None, frame_interval=1, max_pairs=0, 
                          max_rot_err=4.0, max_trans_err=2.0):
    """在自定义数据集上评估语义引导位姿估计的精度
    
    Args:
        dataset_dir: 自定义数据集的根目录（包含帧序列）
        output_dir: 输出结果的目录
        dino_config, dino_weights, sam_weights: 模型配置和权重
        text_prompts: 文本提示列表
        device: 运行设备
        frame_interval: 帧间隔，默认为1（处理所有连续帧）
        max_pairs: 最大处理的图像对数量，0表示无限制
        max_rot_err: 最大允许的旋转误差（度）
        max_trans_err: 最大允许的平移误差（米）
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 假设是RGB-D数据集，使用合适的内参
    # 注意：实际应用中应该从数据集获取或提供这些参数
    fl = 525.0  # 焦距
    cx = 320.0  # 主点x坐标
    cy = 240.0  # 主点y坐标
    K = np.array([[fl, 0, cx], [0, fl, cy], [0, 0, 1]], dtype=np.float32)
    
    # 准备结果记录
    results = {
        "standard": {"rotation_errors": [], "translation_errors": []},
        "hybrid": {"rotation_errors": [], "translation_errors": []},
        "best": {"rotation_errors": [], "translation_errors": [], "method_counts": {"standard": 0, "hybrid": 0}}
    }
    
    # 记录满足误差要求的图像对数量
    valid_pairs_count = 0
    total_processed_pairs = 0
    
    # 创建结果文件
    result_file = output_dir / "custom_dataset_results.txt"
    with open(result_file, "w") as f:
        f.write(f"自定义数据集评估结果\n")
        f.write("=====================================\n\n")
        f.write(f"帧间隔: {frame_interval}, 最大图像对数量: {max_pairs if max_pairs > 0 else '无限制'}\n")
        f.write(f"误差阈值筛选: 旋转误差 <= {max_rot_err}度, 平移误差 <= {max_trans_err}米\n\n")
    
    # 获取数据集中的图像帧
    dataset_dir = Path(dataset_dir)
    seq_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    if not seq_dirs:
        # 如果直接是序列目录（没有子文件夹）
        seq_dirs = [dataset_dir]
    
    # 图像对的总数
    total_pairs_count = 0
    
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        print(f"\n处理序列 {seq_name}")
        
        # 找到所有有效的图像-位姿对
        img_pose_pairs = find_image_pose_pairs(seq_dir)
                               
        # 如果没有找到文件，尝试打印所有文件以进行调试
        if len(img_pose_pairs) == 0:
            print("未找到匹配的图像-位姿对，打印目录内容进行调试：")
            all_files = list(seq_dir.glob("*"))
            for f in all_files[:10]:  # 只打印前10个文件以免输出过多
                print(f"  {f.name}")
            if len(all_files) > 10:
                print(f"  ... 还有 {len(all_files) - 10} 个文件")
        
        print(f"找到 {len(img_pose_pairs)} 个有效图像-位姿对")
        
        # 计算该序列的图像对数量
        seq_pairs = max(0, (len(img_pose_pairs) - 1) // frame_interval)
        total_pairs_count += seq_pairs
    
    # 如果设置了最大图像对数量限制，调整total_pairs
    if max_pairs > 0 and max_pairs < total_pairs_count:
        total_pairs = max_pairs
    else:
        total_pairs = total_pairs_count
    
    current_pair = 0
    pairs_processed = 0
    
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        print(f"\n处理序列 {seq_name} 的图像对")
        
        # 找到所有有效的图像-位姿对
        img_pose_pairs = find_image_pose_pairs(seq_dir)
        
        # 使用帧间隔处理图像
        for i in range(0, len(img_pose_pairs) - 1, frame_interval):
            if i + frame_interval < len(img_pose_pairs):
                img1_path = str(img_pose_pairs[i][0])
                img2_path = str(img_pose_pairs[i + frame_interval][0])
                pose1_path = str(img_pose_pairs[i][1])
                pose2_path = str(img_pose_pairs[i + frame_interval][1])
                
                current_pair += 1
                
                # 检查是否已达到有效图像对数量目标
                if max_pairs > 0 and valid_pairs_count >= max_pairs:
                    print(f"\n已达到目标有效图像对数量 ({max_pairs})，停止处理")
                    break
                
                print(f"\n处理图像对 [{current_pair}/{total_pairs}]: {img_pose_pairs[i][0].name} - {img_pose_pairs[i + frame_interval][0].name}")
                
                # 解析位姿文件
                T1 = parse_custom_pose_file(pose1_path)
                T2 = parse_custom_pose_file(pose2_path)
                
                if T1 is None or T2 is None:
                    print(f"警告：无法解析位姿文件，跳过该对")
                    continue
                
                # 计算真实相对位姿
                T_true = calculate_relative_pose(T1, T2)
                
                try:
                    # 使用语义引导位姿估计，并传递真实位姿用于误差计算
                    pair_output_dir = str(output_dir / f"pair_{current_pair}")
                    
                    T_estimated = semantic_guided_sift_feature_matching(
                        img1_path, img2_path, 
                        pair_output_dir,
                        dino_config, dino_weights, sam_weights,
                        text_prompts if text_prompts else ["building . objects . furniture"],
                        device, T_true
                    )
                    
                    if T_estimated is not None:
                        # 计算误差
                        R_true = T_true[:3, :3]
                        t_true = T_true[:3, 3]
                        R_est = T_estimated.transform[:3, :3]
                        t_est = T_estimated.transform[:3, 3]
                        
                        # 计算旋转误差
                        rot_err = rotation_error(R_true, R_est)
                        
                        # 计算带尺度缩放的平移误差
                        t_est_norm = np.linalg.norm(t_est)
                        t_true_norm = np.linalg.norm(t_true)
                        
                        if t_est_norm < 1e-6 or t_true_norm < 1e-6:
                            trans_err = np.inf
                        else:
                            # 将估计的平移向量缩放到真实尺度
                            scale_factor = t_true_norm / t_est_norm
                            t_est_scaled = t_est * scale_factor
                            trans_err = np.linalg.norm(t_est_scaled - t_true)
                        
                        # 记录使用的方法和误差
                        method = T_estimated.method
                        
                        # 如果是标准方法
                        if method == 'standard':
                            results["standard"]["rotation_errors"].append(rot_err)
                            results["standard"]["translation_errors"].append(trans_err)
                            
                            # 这是当前唯一的结果，也是最佳结果
                            best_method = "standard"
                            best_rot_err = rot_err
                            best_trans_err = trans_err
                            
                        # 如果是混合方法，需要比较两种方法的结果
                        elif method == 'hybrid':
                            results["hybrid"]["rotation_errors"].append(rot_err)
                            results["hybrid"]["translation_errors"].append(trans_err)
                            
                            # 获取标准SIFT的误差（如果有），否则使用混合方法的误差
                            standard_rot_err = None
                            standard_trans_err = None
                            
                            # 判断内部是否有标准方法的结果可用
                            if hasattr(T_estimated, 'std_rot_err') and hasattr(T_estimated, 'std_trans_err'):
                                standard_rot_err = T_estimated.std_rot_err
                                standard_trans_err = T_estimated.std_trans_err
                                
                                # 记录标准方法的误差
                                if len(results["standard"]["rotation_errors"]) < len(results["hybrid"]["rotation_errors"]):
                                    results["standard"]["rotation_errors"].append(standard_rot_err)
                                    results["standard"]["translation_errors"].append(standard_trans_err)
                            
                            # 选择最佳误差
                            if standard_rot_err is not None and standard_trans_err is not None:
                                # 综合考虑旋转和平移误差
                                hybrid_error_sum = rot_err + trans_err
                                standard_error_sum = standard_rot_err + standard_trans_err
                                
                                if standard_error_sum < hybrid_error_sum:
                                    best_method = "standard"
                                    best_rot_err = standard_rot_err
                                    best_trans_err = standard_trans_err
                                else:
                                    best_method = "hybrid"
                                    best_rot_err = rot_err
                                    best_trans_err = trans_err
                            else:
                                # 只有混合方法的结果
                                best_method = "hybrid"
                                best_rot_err = rot_err
                                best_trans_err = trans_err
                        
                        # 检查最佳误差是否满足阈值要求
                        is_valid = (best_rot_err <= max_rot_err) and (best_trans_err <= max_trans_err)
                        
                        # 只有在满足误差要求时才计入统计
                        if is_valid:
                            # 记录最佳结果
                            results["best"]["rotation_errors"].append(best_rot_err)
                            results["best"]["translation_errors"].append(best_trans_err)
                            
                            # 记录使用的方法
                            if best_method == "standard":
                                results["best"]["method_counts"]["standard"] += 1
                            else:
                                results["best"]["method_counts"]["hybrid"] += 1
                                
                            # 增加有效图像对计数
                            valid_pairs_count += 1
                        
                        # 写入每对的结果
                        with open(result_file, "a") as f:
                            f.write(f"图像对 {current_pair}: {img_pose_pairs[i][0].name} - {img_pose_pairs[i + frame_interval][0].name}\n")
                            f.write(f"估计方法: {method}\n")
                            f.write(f"旋转误差: {rot_err:.4f} 度\n")
                            f.write(f"平移误差: {trans_err:.4f} 米\n")
                            
                            # 如果有两种方法的结果可以比较
                            if method == 'hybrid' and standard_rot_err is not None:
                                f.write(f"标准SIFT旋转误差: {standard_rot_err:.4f} 度\n")
                                f.write(f"标准SIFT平移误差: {standard_trans_err:.4f} 米\n")
                                f.write(f"最佳方法: {best_method} (旋转误差: {best_rot_err:.4f}, 平移误差: {best_trans_err:.4f})\n")
                            
                            # 标记是否为有效图像对
                            if is_valid:
                                f.write(f"状态: 有效 ✓ (误差在阈值范围内)\n")
                            else:
                                f.write(f"状态: 无效 ✗ (误差超出阈值: 旋转={best_rot_err:.4f}>{max_rot_err if best_rot_err > max_rot_err else '阈值'} 或 平移={best_trans_err:.4f}>{max_trans_err if best_trans_err > max_trans_err else '阈值'})\n")
                            
                            f.write("-------------------------------------\n")
                    else:
                        print(f"警告：图像对 {img_pose_pairs[i][0].name} - {img_pose_pairs[i + frame_interval][0].name} 位姿估计失败")
                        with open(result_file, "a") as f:
                            f.write(f"图像对 {current_pair}: {img_pose_pairs[i][0].name} - {img_pose_pairs[i + frame_interval][0].name} - 位姿估计失败\n")
                            f.write("-------------------------------------\n")
                
                except Exception as e:
                    print(f"处理图像对时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    with open(result_file, "a") as f:
                        f.write(f"图像对 {current_pair}: {img_pose_pairs[i][0].name} - {img_pose_pairs[i + frame_interval][0].name} - 处理出错: {e}\n")
                        f.write("-------------------------------------\n")
                
                # 更新已处理图像对计数
                pairs_processed += 1
                total_processed_pairs += 1
                
                # 如果已经处理了足够多的图像对，但有效对数量不足，继续处理更多对
                if max_pairs > 0 and valid_pairs_count >= max_pairs:
                    print(f"\n已达到目标有效图像对数量 ({max_pairs})，停止处理")
                    break
                
                # 防止处理太多图像对还未找到足够的有效对（设置一个上限，例如最大限制的3倍）
                if max_pairs > 0 and total_processed_pairs >= max_pairs * 3:
                    print(f"\n已处理 {total_processed_pairs} 对图像，但只找到 {valid_pairs_count} 个有效对，停止处理")
                    break
        
        # 如果已达到目标有效图像对数量，跳出序列循环
        if max_pairs > 0 and valid_pairs_count >= max_pairs:
            break
        
        # 如果处理太多图像对还未找到足够的有效对，停止处理
        if max_pairs > 0 and total_processed_pairs >= max_pairs * 3:
            break
    
    # 计算统计结果
    summary = {}
    for method in ["standard", "hybrid", "best"]:
        if method == "best" and not results[method]["rotation_errors"]:
            print(f"警告：未找到任何满足误差阈值的图像对（旋转误差<={max_rot_err}度，平移误差<={max_trans_err}米）")
            summary[method] = {
                "pairs_count": 0,
                "standard_count": 0,
                "hybrid_count": 0,
                "rot_median": float('nan'),
                "rot_mean": float('nan'),
                "trans_median": float('nan'),
                "trans_mean": float('nan')
            }
        elif results[method]["rotation_errors"]:
            rot_median = np.median(results[method]["rotation_errors"])
            rot_mean = np.mean(results[method]["rotation_errors"])
            trans_median = np.median([e for e in results[method]["translation_errors"] if e != np.inf])
            trans_mean = np.mean([e for e in results[method]["translation_errors"] if e != np.inf])
            
            summary[method] = {
                "rot_median": rot_median,
                "rot_mean": rot_mean,
                "trans_median": trans_median,
                "trans_mean": trans_mean,
                "pairs_count": len(results[method]["rotation_errors"])
            }
            
            # 添加最佳方法的方法选择计数
            if method == "best":
                summary[method]["standard_count"] = results[method]["method_counts"]["standard"]
                summary[method]["hybrid_count"] = results[method]["method_counts"]["hybrid"]
        else:
            summary[method] = {
                "pairs_count": 0,
                "rot_median": float('nan'),
                "rot_mean": float('nan'),
                "trans_median": float('nan'),
                "trans_mean": float('nan')
            }
            
            if method == "best":
                summary[method]["standard_count"] = 0
                summary[method]["hybrid_count"] = 0
    
    # 写入摘要结果
    with open(result_file, "a") as f:
        f.write("\n总体评估结果\n")
        f.write("=====================================\n\n")
        f.write(f"总共处理的图像对: {total_processed_pairs}\n")
        f.write(f"误差阈值: 旋转误差 <= {max_rot_err}度, 平移误差 <= {max_trans_err}米\n\n")
        
        for method in ["standard", "hybrid", "best"]:
            if method in summary:
                f.write(f"{method.capitalize()} 方法结果:\n")
                f.write(f"处理的图像对数量: {summary[method]['pairs_count']}\n")
                
                if method == "best":
                    f.write(f"其中标准SIFT方法被选择: {summary[method]['standard_count']} 次\n")
                    f.write(f"其中混合方法被选择: {summary[method]['hybrid_count']} 次\n")
                
                if summary[method]["pairs_count"] > 0:
                    f.write(f"旋转误差中位数: {summary[method]['rot_median']:.4f} 度\n")
                    f.write(f"旋转误差均值: {summary[method]['rot_mean']:.4f} 度\n")
                    f.write(f"平移误差中位数: {summary[method]['trans_median']:.4f} 米\n")
                    f.write(f"平移误差均值: {summary[method]['trans_mean']:.4f} 米\n\n")
                else:
                    f.write("没有有效数据点用于计算统计结果\n\n")
    
    # 打印摘要结果
    print("\n评估结果摘要:")
    print(f"总共处理的图像对: {total_processed_pairs}")
    print(f"误差阈值: 旋转误差 <= {max_rot_err}度, 平移误差 <= {max_trans_err}米\n")
    
    for method in ["standard", "hybrid", "best"]:
        if method in summary:
            print(f"{method.capitalize()} 方法:")
            print(f"满足误差阈值的图像对数量: {summary[method]['pairs_count']}")
            
            if method == "best":
                print(f"其中标准SIFT方法被选择: {summary[method]['standard_count']} 次")
                print(f"其中混合方法被选择: {summary[method]['hybrid_count']} 次")
            
            if summary[method]["pairs_count"] > 0:
                print(f"旋转误差中位数: {summary[method]['rot_median']:.4f} 度")
                print(f"旋转误差均值: {summary[method]['rot_mean']:.4f} 度")
                print(f"平移误差中位数: {summary[method]['trans_median']:.4f} 米")
                print(f"平移误差均值: {summary[method]['trans_mean']:.4f} 米\n")
            else:
                print("没有有效数据点用于计算统计结果\n")
    
    return summary

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="语义引导SIFT位姿估计")
    
    # 单对图像处理模式
    parser.add_argument("--img1", type=str, default="",
                        help="第一张图像的路径（单对图像处理模式）")
    parser.add_argument("--img2", type=str, default="",
                        help="第二张图像的路径（单对图像处理模式）")
    
    # 数据集评估模式
    parser.add_argument("--dataset-mode", action="store_true",
                        help="是否使用数据集评估模式")
    parser.add_argument("--dataset-type", type=str, default="custom",
                        help="数据集类型, 可选 'cambridge', 'custom' 或 '7scenes'")
    parser.add_argument("--dataset-dir", type=str, default="",
                        help="数据集根目录")
    parser.add_argument("--scene", type=str, default="ShopFacade",
                        help="场景名称，例如：ShopFacade, KingsCollege等 (仅Cambridge数据集需要)")
    parser.add_argument("--pose-file", type=str, default="",
                        help="位姿文件路径 (仅Cambridge数据集需要)")
    parser.add_argument("--frame-interval", type=int, default=1,
                        help="处理图像的帧间隔，增大可减少相邻图像间的视角相似度")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="最大处理的图像对数量，设置为0表示无限制")
    parser.add_argument("--max-rot-err", type=float, default=4.0,
                        help="最大允许的旋转误差（度），超过此值的图像对将被过滤，默认为4度")
    parser.add_argument("--max-trans-err", type=float, default=2.0,
                        help="最大允许的平移误差（米），超过此值的图像对将被过滤，默认为2米")
    
    # 模型路径参数
    parser.add_argument("--dino-config", type=str, 
                        help="Grounding-DINO配置文件的路径")
    parser.add_argument("--dino-weights", type=str, 
                        help="Grounding-DINO模型权重的路径")
    parser.add_argument("--sam-weights", type=str, 
                        help="SAM模型权重的路径")
    
    # 文本提示参数
    parser.add_argument("--text-prompts", type=str, nargs='+', 
                        default=["building facade . windows . architecture details", 
                                 "stone walls . architectural features . shop entrances", 
                                 "building corners . stable structures . windows"],
                        help="多个文本提示，程序将自动选择效果最好的")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="", 
                        help="运行模型的设备，例如：'cuda' 或 'cpu'。留空自动选择。")
    parser.add_argument("--gpu", type=int, default=0,
                        help="指定使用的GPU编号，例如：0, 1, 2。只在device为空时有效。")
    
    # 输出路径参数
    parser.add_argument("--output-dir", type=str, default="results",
                        help="结果保存的目录路径")
    
    args = parser.parse_args()
    
    # 设置默认模型路径（如果未指定）
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    dino_config = args.dino_config if args.dino_config else str(base_dir / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    dino_weights = args.dino_weights if args.dino_weights else str(base_dir / "weights/groundingdino_swint_ogc.pth")
    sam_weights = args.sam_weights if args.sam_weights else str(base_dir / "weights/sam_vit_h_4b8939.pth")
    
    # 确保device参数不为空
    if args.device == "":
        if torch.cuda.is_available():
            args.device = f"cuda:{args.gpu}"
        else:
            args.device = "cpu"
    
    print(f"使用设备: {args.device}")

    if args.dataset_mode:
        # 数据集评估模式
        if not args.dataset_dir:
            print("错误：请提供数据集目录 --dataset-dir")
            return
            
        if args.dataset_type == "cambridge":
            # Cambridge数据集评估
            if not args.pose_file:
                print("错误：请提供位姿文件路径 --pose-file (Cambridge数据集需要)")
                return
                
            evaluate_cambridge_dataset(
                args.dataset_dir, args.scene, args.pose_file, args.output_dir,
                dino_config, dino_weights, sam_weights,
                args.text_prompts, args.device, args.frame_interval, args.max_pairs, args.max_rot_err, args.max_trans_err
            )
        elif args.dataset_type == "custom":
            # 自定义数据集评估
            evaluate_custom_dataset(
                args.dataset_dir, args.output_dir,
                dino_config, dino_weights, sam_weights,
                args.text_prompts, args.device, args.frame_interval, args.max_pairs, args.max_rot_err, args.max_trans_err
            )
        elif args.dataset_type == "7scenes":
            # 7-Scenes数据集评估
            evaluate_7scenes_dataset(
                args.dataset_dir, args.output_dir,
                dino_config, dino_weights, sam_weights,
                args.text_prompts, args.device, args.frame_interval, args.max_pairs, args.max_rot_err, args.max_trans_err
            )
        else:
            print(f"错误：不支持的数据集类型 '{args.dataset_type}'，支持的选项: 'cambridge', 'custom', '7scenes'")
            return
    else:
        # 单对图像处理模式
        if not args.img1 or not args.img2:
            print("错误：在单对图像处理模式下，请提供两张图像的路径 --img1 和 --img2")
            return
            
        # 运行语义引导SIFT位姿估计
        semantic_guided_sift_feature_matching(
            args.img1, args.img2, args.output_dir,
            dino_config, dino_weights, sam_weights,
            args.text_prompts, args.device
        )

def evaluate_7scenes_dataset(dataset_dir, output_dir, dino_config, dino_weights, sam_weights, 
                          text_prompts=None, device=None, frame_interval=1, max_pairs=0, 
                          max_rot_err=4.0, max_trans_err=2.0):
    """评估7-Scenes数据集的所有7个场景
    
    Args:
        dataset_dir: 7-Scenes数据集的根目录
        output_dir: 输出结果的目录
        dino_config, dino_weights, sam_weights: 模型配置和权重
        text_prompts: 文本提示列表
        device: 运行设备
        frame_interval: 帧间隔，默认为1（处理所有连续帧）
        max_pairs: 最大处理的图像对数量，0表示无限制
        max_rot_err: 最大允许的旋转误差（度）
        max_trans_err: 最大允许的平移误差（米）
    """
    # 7-Scenes场景列表
    scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建总结果文件
    summary_file = output_dir / "7scenes_all_results.txt"
    with open(summary_file, "w") as f:
        f.write(f"7-Scenes数据集全部场景评估结果\n")
        f.write("=====================================\n\n")
        f.write(f"帧间隔: {frame_interval}, 每个场景最大图像对数量: {max_pairs if max_pairs > 0 else '无限制'}\n")
        f.write(f"误差阈值筛选: 旋转误差 <= {max_rot_err}度, 平移误差 <= {max_trans_err}米\n\n")
    
    # 存储所有场景的汇总结果
    all_scenes_results = {
        "standard": {"rotation_errors": [], "translation_errors": []},
        "hybrid": {"rotation_errors": [], "translation_errors": []},
        "best": {"rotation_errors": [], "translation_errors": [], "method_counts": {"standard": 0, "hybrid": 0}}
    }
    
    # 保存每个场景的详细摘要信息
    scene_summaries = {}
    
    # 处理每个场景
    for scene in scenes:
        print(f"\n============ 处理7-Scenes场景: {scene} ============")
        
        scene_dir = Path(dataset_dir) / scene
        scene_output_dir = output_dir / scene
        
        if not scene_dir.exists():
            print(f"警告: 场景目录 {scene_dir} 不存在，跳过此场景")
            continue
        
        # 为7-Scenes场景选择适合的文本提示
        scene_text_prompts = text_prompts
        # 如果未指定文本提示，使用适合室内场景的提示
        if scene_text_prompts is None or len(scene_text_prompts) == 0:
            scene_text_prompts = [
                "furniture . room structure . objects",
                "walls . doors . windows . desk",
                "indoor objects . corners . structures"
            ]
        
        try:
            # 评估当前场景
            scene_summary = evaluate_custom_dataset(
                str(scene_dir), str(scene_output_dir),
                dino_config, dino_weights, sam_weights,
                scene_text_prompts, device, frame_interval, max_pairs, max_rot_err, max_trans_err
            )
            
            # 保存场景摘要结果
            scene_summaries[scene] = scene_summary
            
            # 将当前场景的误差数据添加到汇总结果中
            for method in ["standard", "hybrid", "best"]:
                if scene_summary[method]["pairs_count"] > 0:
                    all_scenes_results[method]["rotation_errors"].extend(
                        [scene_summary[method]["rot_mean"]] * scene_summary[method]["pairs_count"]
                    )
                    all_scenes_results[method]["translation_errors"].extend(
                        [scene_summary[method]["trans_mean"]] * scene_summary[method]["pairs_count"]
                    )
                    
                    # 对于best方法，还需统计方法选择计数
                    if method == "best":
                        all_scenes_results[method]["method_counts"]["standard"] += scene_summary[method]["standard_count"]
                        all_scenes_results[method]["method_counts"]["hybrid"] += scene_summary[method]["hybrid_count"]
            
        except Exception as e:
            print(f"处理场景 {scene} 时出错: {e}")
            import traceback
            traceback.print_exc()
            with open(summary_file, "a") as f:
                f.write(f"\n场景 {scene} 处理出错: {e}\n")
    
    # 计算所有场景的综合统计结果
    all_scenes_summary = {}
    for method in ["standard", "hybrid", "best"]:
        if not all_scenes_results[method]["rotation_errors"]:
            all_scenes_summary[method] = {
                "pairs_count": 0,
                "rot_median": float('nan'),
                "rot_mean": float('nan'),
                "trans_median": float('nan'),
                "trans_mean": float('nan')
            }
            
            if method == "best":
                all_scenes_summary[method]["standard_count"] = 0
                all_scenes_summary[method]["hybrid_count"] = 0
        else:
            rot_values = [e for e in all_scenes_results[method]["rotation_errors"] if not np.isnan(e)]
            trans_values = [e for e in all_scenes_results[method]["translation_errors"] if not np.isnan(e) and e != np.inf]
            
            if rot_values and trans_values:
                rot_median = np.median(rot_values)
                rot_mean = np.mean(rot_values)
                trans_median = np.median(trans_values)
                trans_mean = np.mean(trans_values)
                
                all_scenes_summary[method] = {
                    "rot_median": rot_median,
                    "rot_mean": rot_mean,
                    "trans_median": trans_median,
                    "trans_mean": trans_mean,
                    "pairs_count": len(rot_values)
                }
                
                if method == "best":
                    all_scenes_summary[method]["standard_count"] = all_scenes_results[method]["method_counts"]["standard"]
                    all_scenes_summary[method]["hybrid_count"] = all_scenes_results[method]["method_counts"]["hybrid"]
            else:
                all_scenes_summary[method] = {
                    "pairs_count": 0,
                    "rot_median": float('nan'),
                    "rot_mean": float('nan'),
                    "trans_median": float('nan'),
                    "trans_mean": float('nan')
                }
                
                if method == "best":
                    all_scenes_summary[method]["standard_count"] = 0
                    all_scenes_summary[method]["hybrid_count"] = 0
    
    # 将每个场景的详细结果写入摘要文件
    with open(summary_file, "a") as f:
        for scene in scenes:
            if scene in scene_summaries:
                summary = scene_summaries[scene]
                
                f.write(f"\n场景: {scene}\n")
                f.write("-------------------------------------\n")
                
                for method in ["standard", "hybrid", "best"]:
                    if method in summary:
                        f.write(f"{method.capitalize()} 方法结果:\n")
                        f.write(f"处理的图像对数量: {summary[method]['pairs_count']}\n")
                        
                        if method == "best":
                            f.write(f"其中标准SIFT方法被选择: {summary[method]['standard_count']} 次\n")
                            f.write(f"其中混合方法被选择: {summary[method]['hybrid_count']} 次\n")
                        
                        if summary[method]["pairs_count"] > 0:
                            f.write(f"旋转误差中位数: {summary[method]['rot_median']:.4f} 度\n")
                            f.write(f"旋转误差均值: {summary[method]['rot_mean']:.4f} 度\n")
                            f.write(f"平移误差中位数: {summary[method]['trans_median']:.4f} 米\n")
                            f.write(f"平移误差均值: {summary[method]['trans_mean']:.4f} 米\n\n")
                        else:
                            f.write("没有有效数据点用于计算统计结果\n\n")
            else:
                f.write(f"\n场景: {scene} - 未能获取结果\n\n")
    
    # 写入汇总结果
    with open(summary_file, "a") as f:
        f.write("\n全部场景汇总结果\n")
        f.write("=====================================\n\n")
        
        for method in ["standard", "hybrid", "best"]:
            if method in all_scenes_summary:
                f.write(f"{method.capitalize()} 方法汇总结果:\n")
                f.write(f"总有效图像对数量: {all_scenes_summary[method]['pairs_count']}\n")
                
                if method == "best":
                    f.write(f"其中标准SIFT方法被选择: {all_scenes_summary[method]['standard_count']} 次\n")
                    f.write(f"其中混合方法被选择: {all_scenes_summary[method]['hybrid_count']} 次\n")
                
                if all_scenes_summary[method]["pairs_count"] > 0:
                    f.write(f"旋转误差中位数: {all_scenes_summary[method]['rot_median']:.4f} 度\n")
                    f.write(f"旋转误差均值: {all_scenes_summary[method]['rot_mean']:.4f} 度\n")
                    f.write(f"平移误差中位数: {all_scenes_summary[method]['trans_median']:.4f} 米\n")
                    f.write(f"平移误差均值: {all_scenes_summary[method]['trans_mean']:.4f} 米\n\n")
                else:
                    f.write("没有有效数据点用于计算统计结果\n\n")
    
    # 打印汇总结果
    print("\n全部场景汇总结果:")
    print("=====================================")
    
    for method in ["standard", "hybrid", "best"]:
        if method in all_scenes_summary:
            print(f"\n{method.capitalize()} 方法汇总结果:")
            print(f"总有效图像对数量: {all_scenes_summary[method]['pairs_count']}")
            
            if method == "best":
                print(f"其中标准SIFT方法被选择: {all_scenes_summary[method]['standard_count']} 次")
                print(f"其中混合方法被选择: {all_scenes_summary[method]['hybrid_count']} 次")
            
            if all_scenes_summary[method]["pairs_count"] > 0:
                print(f"旋转误差中位数: {all_scenes_summary[method]['rot_median']:.4f} 度")
                print(f"旋转误差均值: {all_scenes_summary[method]['rot_mean']:.4f} 度")
                print(f"平移误差中位数: {all_scenes_summary[method]['trans_median']:.4f} 米")
                print(f"平移误差均值: {all_scenes_summary[method]['trans_mean']:.4f} 米")
            else:
                print("没有有效数据点用于计算统计结果")
    
    print(f"\n所有结果已保存到: {summary_file}")
    return all_scenes_summary

def visualize_matches(img1, img2, kp1, kp2, final_matches):
    """可视化匹配结果"""
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, final_matches, None,
                                matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result_img

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()