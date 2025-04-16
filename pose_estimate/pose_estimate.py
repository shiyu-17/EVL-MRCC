# sift + ransac 计算两张图片间相对位姿
import cv2
import numpy as np

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

def get_feature_matches(img1, img2):
    """获取图像之间的特征匹配"""
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    matches = flann.knnMatch(des1, des2, k=2)

    ratio_thresh = 0.6
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

def sift_feature_matching():
    fx, fy, cx, cy = 211.949, 211.949, 127.933, 95.9333
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    
    img1 = cv2.imread("/Users/shiyu/mycode/EVL-MRCC/images/44358452_43500.602.png", cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread("/Users/shiyu/mycode/EVL-MRCC/images/44358452_43501.085.png", cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        raise ValueError("图像加载失败，请检查文件路径。")

    gray1 = preprocess_image(img1)  # 此处调用 preprocess_image
    gray2 = preprocess_image(img2)
    
    good_matches, kp1, kp2 = get_feature_matches(gray1, gray2)

    T, mask, good_matches = estimate_pose(good_matches, kp1, kp2, K)
    print("\n估计位姿：")
    print(T)
    
    if T is not None:
        T_true = np.array([[ 0.99968,  -0.025151,  0.002634, -0.002958],
 [ 0.024646 , 0.992321,  0.121207 , 0.047966],
 [-0.005662, -0.121104 , 0.992624,  0.055945],
 [ 0,        0,        0,        1      ]])
        # T_true = np.array([[0.99981151, 0.01853355, -0.00578381, 0.00737292],
        #                   [-0.0145849, 0.91360526, 0.40634063, 0.02877406],
        #                   [0.01281505, -0.40617969, 0.91370336, 0.37641721],
        #                   [0, 0, 0, 1]])
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