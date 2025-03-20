import cv2
import numpy as np

def enhance_contrast(img):
    """图像对比度增强"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def sift_feature_matching():
    # 相机内参
    fx, fy, cx, cy = 211.949, 211.949, 127.933, 95.9333
    K = np.array([[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]], dtype=np.float32)
    
    # 图像加载
    img1 = cv2.imread("./41069021_305.377.png", cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread("./41069021_306.360.png", cv2.IMREAD_UNCHANGED)

    # 预处理流程
    def preprocess(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return enhance_contrast(img)
    
    gray1 = preprocess(img1)
    gray2 = preprocess(img2)

    # SIFT特征检测器
    sift = cv2.SIFT_create(
        nfeatures=0, nOctaveLayers=3,
        contrastThreshold=0.04, edgeThreshold=10, sigma=1.6
    )

    # 特征检测与匹配
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # FLANN匹配器
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=100)
    )
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's比率测试
    ratio_thresh = 0.6
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    # 位姿估计
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
        # 归一化坐标
        src_norm = cv2.undistortPoints(src_pts, K, None)
        dst_norm = cv2.undistortPoints(dst_pts, K, None)
        
        # 估计本质矩阵
        E, mask = cv2.findEssentialMat(
            src_norm, dst_norm, 
            method=cv2.RANSAC, 
            prob=0.999, threshold=0.001
        )
        
        # 恢复位姿
        _, R, t, _ = cv2.recoverPose(E, src_norm, dst_norm, mask=mask)

        # 验证分解后的旋转矩阵有效性
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-3), "无效的旋转矩阵"

        # 检查平移方向合理性（需结合场景先验）
        print("平移方向:", t.ravel())
        
        # 构建4x4变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        
        print("相对位姿矩阵：")
        print(np.round(T, 4))
        
        # 更新内点匹配
        matches_mask = mask.ravel().astype(bool)
        final_matches = [m for m, flag in zip(good_matches, matches_mask) if flag]
    else:
        final_matches = good_matches

    # 可视化结果
    result_img = cv2.drawMatches(
        img1, kp1, img2, kp2, final_matches, None,
        matchColor=(0,255,0), singlePointColor=(255,0,0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    print(f"\n特征点统计：")
    print(f"图像1特征点: {len(kp1)}, 图像2特征点: {len(kp2)}")
    print(f"初步匹配: {len(good_matches)}, 几何验证后: {len(final_matches)}")

    # cv2.imshow('Feature Matching', result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    sift_feature_matching()