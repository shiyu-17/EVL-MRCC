import cv2
import numpy as np

def enhance_contrast(img):
    """图像对比度增强"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def feature_matching():
    # 读取图像
    img1 = cv2.imread("./41069021_305.377.png", cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread("./41069021_306.360.png", cv2.IMREAD_UNCHANGED)

    # 预处理函数
    def preprocess(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return enhance_contrast(img)
    
    gray1 = preprocess(img1)
    gray2 = preprocess(img2)

    # 初始化ORB检测器
    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=10,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=10
    )

    # 检测特征点和描述子
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 初始化FLANN匹配器
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,
        key_size=12,
        multi_probe_level=1
    )
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 特征匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's比率测试筛选匹配点
    ratio_thresh = 0.75
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 初始化4x4变换矩阵
    T = np.eye(4, dtype=np.float32)
    
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 相机内参（示例参数）
        fx, fy, cx, cy = 211.949, 211.949, 127.933, 95.9333
        K = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)

        # 计算本质矩阵（RANSAC）
        norm_threshold = 1.0 / fx
        E, mask_e = cv2.findEssentialMat(src_pts, dst_pts, cameraMatrix=K,
                                        method=cv2.RANSAC, prob=0.999, threshold=norm_threshold)

        if E is not None and E.shape == (3, 3):
            # 恢复相对位姿
            _, R, t, mask_recover = cv2.recoverPose(E, src_pts, dst_pts, cameraMatrix=K, mask=mask_e)
            
            # 构建4x4变换矩阵
            T[:3, :3] = R          # 旋转部分
            T[:3, 3] = t.ravel()   # 平移部分（注意尺度不确定性）
            
            print("\n4x4相对位姿矩阵：")
            print(T)
            
            matches_mask = mask_recover.ravel().tolist()
        else:
            print("无法计算有效的本质矩阵")
            matches_mask = []
        
        final_matches = [m for m, flag in zip(good_matches, matches_mask) if flag]
    else:
        print("匹配点不足，无法进行位姿估计")
        final_matches = good_matches

    # 可视化结果
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, final_matches, None, **draw_params)

    print(f"\n总匹配数: {len(good_matches)}")
    print(f"几何验证后匹配数: {len(final_matches)}")

    cv2.imshow('4x4 Pose Estimation', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    feature_matching()