import cv2
import numpy as np

def enhance_contrast(img):
    """图像对比度增强（保持与原始代码一致）"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def sift_feature_matching():
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

    # SIFT特征检测器配置
    sift = cv2.SIFT_create(
        nfeatures=0,          # 无限制特征点数量
        nOctaveLayers=3,      # 金字塔每组层数
        contrastThreshold=0.04,  # 对比度阈值
        edgeThreshold=10,     # 边缘阈值
        sigma=1.6             # 高斯模糊系数
    )

    # 特征检测与描述子计算
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN参数配置（适配SIFT的浮点描述子）
    FLANN_INDEX_KDTREE = 1
    index_params = dict(
        algorithm=FLANN_INDEX_KDTREE,
        trees=5
    )
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # KNN匹配（保持双向匹配）
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's比率测试
    ratio_thresh = 0.7
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 几何验证（与原始代码保持一致）
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        final_matches = [m for m, flag in zip(good_matches, matches_mask) if flag]
    else:
        final_matches = good_matches

    # 可视化设置（保持与原始代码一致）
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    result_img = cv2.drawMatches(img1, kp1, img2, kp2, final_matches, None, **draw_params)

    # 输出统计信息
    print(f"SIFT特征点数 图像1: {len(kp1)}, 图像2: {len(kp2)}")
    print(f"初步匹配数: {len(good_matches)}")
    print(f"几何验证后匹配数: {len(final_matches)}")

    cv2.imshow('SIFT Feature Matching', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sift_feature_matching()