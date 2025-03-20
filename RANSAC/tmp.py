import cv2
import numpy as np
import torch

def enhance_contrast(img):
    """图像对比度增强"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def compute_essential_matrix(x1, x2, weights=None):
    """可导的本质矩阵估计"""
    # 构建系数矩阵A
    x1 = x1.unsqueeze(1)  # [N, 1, 2]
    x2 = x2.unsqueeze(1)
    A = torch.cat([
        x1[..., 0] * x2[..., 0],
        x1[..., 0] * x2[..., 1],
        x1[..., 0],
        x1[..., 1] * x2[..., 0],
        x1[..., 1] * x2[..., 1],
        x1[..., 1],
        x2[..., 0],
        x2[..., 1],
        torch.ones_like(x1[..., 0])
    ], dim=2).permute(0, 2, 1)  # [N, 9, 1]
    
    if weights is not None:
        A = A * weights.view(-1, 1, 1)
    
    # 解方程并投影到本质矩阵空间
    U, S, V = torch.svd(A.squeeze(-1))
    E = V[:, -1].view(-1, 3, 3)
    
    # 对每个候选矩阵进行投影
    for i in range(E.shape[0]):
        U, S, Vt = torch.svd(E[i])
        S = torch.tensor([(S[0]+S[1])/2, (S[0]+S[1])/2, 0.0], dtype=torch.float32)
        E[i] = U @ torch.diag(S) @ Vt
    
    return E.mean(dim=0)  # 取平均作为最终估计

def robust_estimate_essential(x1, x2, iterations=5, threshold=0.001):
    """可导的鲁棒本质矩阵估计"""
    threshold_sq = threshold**2
    best_E = None
    best_score = -1
    
    # 多次采样初始化
    for _ in range(20):
        # 随机采样5个点
        indices = torch.randperm(x1.shape[0])[:5]
        sample_x1 = x1[indices]
        sample_x2 = x2[indices]
        
        # 计算初始估计
        with torch.no_grad():
            E_init = compute_essential_matrix(sample_x1, sample_x2)
        
        # 迭代优化
        E = E_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([E], lr=1e-3)
        
        for _ in range(iterations):
            # 计算Sampson误差
            x1_h = torch.cat([x1, torch.ones_like(x1[:, :1])], dim=1)
            x2_h = torch.cat([x2, torch.ones_like(x2[:, :1])], dim=1)
            Ex1 = torch.matmul(E, x1_h.t()).t()
            Etx2 = torch.matmul(E.t(), x2_h.t()).t()
            
            numerator = (x2_h * Ex1).sum(dim=1).square()
            denominator = Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 + Etx2[:, 1]**2 + 1e-8
            error = numerator / denominator
            
            # 计算软内点得分
            weights = torch.sigmoid(-error / threshold_sq)
            loss = -(weights * (1 - torch.sigmoid(error/阈值_sq - 5))).sum()
            
            # 优化步骤
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 投影到本质矩阵空间
            with torch.no_grad():
                U, S, Vt = torch.svd(E)
                S = torch.tensor([(S[0]+S[1])/2, (S[0]+S[1])/2, 0.0], dtype=torch.float32)
                E.copy_(U @ torch.diag(S) @ Vt)
        
        # 评估模型
        with torch.no_grad():
            inlier_score = torch.sum(torch.exp(-error / threshold_sq)).item()
            
        if inlier_score > best_score:
            best_score = inlier_score
            best_E = E.detach()
    
    return best_E

def sift_feature_matching():
    # 相机内参
    fx, fy, cx, cy = 211.949, 211.949, 127.933, 95.9333
    K = np.array([[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]], dtype=np.float32)
    
    # 图像加载与预处理
    img1 = cv2.imread("./41069021_305.377.png", cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread("./41069021_306.360.png", cv2.IMREAD_UNCHANGED)
    
    def preprocess(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return enhance_contrast(img)
    
    gray1 = preprocess(img1)
    gray2 = preprocess(img2)
    
    # 特征提取
    sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # 特征匹配
    flann = cv2.FlannBasedMatcher({'algorithm': 1}, {'checks': 100})
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 筛选匹配
    ratio_thresh = 0.6
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    
    if len(good_matches) < 4:
        print("Not enough matches")
        return
    
    # 坐标转换
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # 归一化坐标
    src_norm = cv2.undistortPoints(src_pts.reshape(-1,1,2), K, None)
    dst_norm = cv2.undistortPoints(dst_pts.reshape(-1,1,2), K, None)
    
    # 转换为PyTorch张量
    src_tensor = torch.from_numpy(src_norm.squeeze()).float()
    dst_tensor = torch.from_numpy(dst_norm.squeeze()).float()
    
    # 可微分RANSAC估计本质矩阵
    E = robust_estimate_essential(src_tensor, dst_tensor).numpy()
    
    # 计算内点掩码
    mask = np.ones(len(good_matches), dtype=np.uint8)
    if E is not None:
        # 恢复位姿
        _, R, t, mask = cv2.recoverPose(E, src_norm, dst_norm, mask=mask)
        mask = mask.ravel().astype(bool)
        print("Relative pose:\nRotation:", R, "\nTranslation:", t)
    
    # 可视化
    draw_params = dict(matchColor=(0,255,0), 
                       singlePointColor=(255,0,0),
                       matchesMask=mask.tolist(),
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
    
    cv2.imshow('Differentiable RANSAC', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sift_feature_matching()