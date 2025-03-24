import cv2
import numpy as np

# ------------------------- 确保所有函数在使用前定义 -------------------------
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

def translation_direction_error(t1, t2):
    """计算平移方向的角度误差（单位：度）"""
    norm1 = np.linalg.norm(t1)
    norm2 = np.linalg.norm(t2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 180.0
    t1_normalized = t1 / norm1
    t2_normalized = t2 / norm2
    dot_product = np.dot(t1_normalized, t2_normalized)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.rad2deg(angle_rad)

def compute_reprojection_error(T, K, matches, kp1, kp2):
    """计算重投影误差（单位：像素）"""
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    src_norm = cv2.undistortPoints(src_pts.reshape(-1, 1, 2), K, None)
    dst_norm = cv2.undistortPoints(dst_pts.reshape(-1, 1, 2), K, None)
    
    # 提取相对位姿的R和t
    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    
    points_4d = cv2.triangulatePoints(
        np.eye(3, 4), 
        np.hstack((R, t)),
        src_norm, 
        dst_norm
    )
    points_3d = points_4d[:3] / points_4d[3]
    
    # 将旋转矩阵转换为旋转向量
    rvec, _ = cv2.Rodrigues(R)
    
    # 使用正确的R和t进行投影
    projected_pts, _ = cv2.projectPoints(points_3d.T,
                                       rvec,
                                       t,
                                       K,
                                       None)
    projected_pts = projected_pts.squeeze()
    
    errors = np.linalg.norm(projected_pts - dst_pts, axis=1)
    return np.mean(errors)

def compute_error(T_relative, T_icp, K, matches, kp1, kp2):
    """计算综合误差"""
    R_relative = T_relative[:3, :3]
    t_relative = T_relative[:3, 3]
    R_icp = T_icp[:3, :3]
    t_icp = T_icp[:3, 3]
    
    rotation_err = rotation_error(R_relative, R_icp)
    direction_err = translation_direction_error(t_relative, t_icp)
    reproj_err = compute_reprojection_error(T_relative, K, matches, kp1, kp2)
    
    print("\n综合误差分析:")
    print(f"旋转角度误差: {rotation_err:.4f} 度")
    print(f"平移方向误差: {direction_err:.4f} 度")
    print(f"平均重投影误差: {reproj_err:.4f} 像素")
    return rotation_err, direction_err, reproj_err

def get_feature_matches(img1, img2):
    """获取图像之间的特征匹配"""
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    matches = flann.knnMatch(des1, des2, k=2)

    ratio_thresh = 0.8
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    
    return good_matches, kp1, kp2

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

def visualize_matches(img1, img2, kp1, kp2, final_matches):
    """可视化匹配结果"""
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, final_matches, None,
                                 matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result_img

# ------------------------- 主函数 -------------------------
def sift_feature_matching():
    fx, fy, cx, cy = 211.949, 211.949, 127.933, 95.9333
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    
    img1 = cv2.imread("./41125700_3813.347.png", cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread("./41125700_3813.846.png", cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        raise ValueError("图像加载失败，请检查文件路径。")

    gray1 = preprocess_image(img1)  # 此处调用 preprocess_image
    gray2 = preprocess_image(img2)
    
    good_matches, kp1, kp2 = get_feature_matches(gray1, gray2)

    T, mask, good_matches = estimate_pose(good_matches, kp1, kp2, K)
    print("\n估计位姿：")
    print(T)
    
    if T is not None:
        T_true = np.array([[ 0.999226,  0.02439,   0.030866,  0.016651],
                   [-0.024926,  0.999543,  0.01709,   0.001163],
                   [-0.030435, -0.017846,  0.999377,  0.045385],
                   [ 0.,        0.,        0.,        1.]])
        print("\n真实位姿：")
        print(T_true)
        matches_mask = mask.ravel().astype(bool)
        final_matches = [m for m, flag in zip(good_matches, matches_mask) if flag]
        compute_error(T, T_true, K, final_matches, kp1, kp2)

    else:
        print("位姿估计失败：匹配点不足")
        return

    result_img = visualize_matches(img1, img2, kp1, kp2, final_matches)
    print("\n特征点统计：")
    print(f"图像1特征点: {len(kp1)}, 图像2特征点: {len(kp2)}")
    print(f"初步匹配: {len(good_matches)}, 几何验证后: {len(final_matches)}")

    cv2.imshow('Feature Matching', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        sift_feature_matching()
    except Exception as e:
        print(f"发生错误: {e}")