import cv2
import numpy as np

def enhance_contrast(img):
    """图像对比度增强"""
    # CLAHE直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def feature_matching():
    # 图像路径
    img1 = cv2.imread("./41069021_305.377.png", cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread("./41069021_306.360.png", cv2.IMREAD_UNCHANGED)

    # 预处理
    def preprocess(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return enhance_contrast(img)
    
    gray1 = preprocess(img1)
    gray2 = preprocess(img2)

    # ORB参数优化
    orb = cv2.ORB_create(
        nfeatures=2000,        # 增加最大特征点数
        scaleFactor=1.2,       # 金字塔缩放系数
        nlevels=10,            # 金字塔层数
        edgeThreshold=15,      # 减小边界阈值
        patchSize=31,          # 描述子区域大小
        fastThreshold=10       # 降低FAST角点阈值
    )

    # 检测特征点
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # FLANN匹配器（适合ORB）
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6,
                       key_size = 12,
                       multi_probe_level = 1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 双向匹配（增加匹配数量）
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 几何一致性验证（RANSAC）
    if len(good_matches) >= 4:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        
        # 单应性矩阵估计
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        
        # 应用几何约束
        final_matches = [m for m, flag in zip(good_matches, matches_mask) if flag]
    else:
        final_matches = good_matches

    # 可视化参数
    draw_params = dict(
        matchColor = (0,255,0),
        singlePointColor = (255,0,0),
        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 绘制结果
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, final_matches, None, **draw_params)

    # 显示统计信息
    print(f"总匹配数: {len(good_matches)}")
    print(f"几何验证后匹配数: {len(final_matches)}")

    cv2.imshow('Optimized Feature Matching', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    feature_matching()