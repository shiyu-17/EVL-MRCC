import cv2
import numpy as np
import torch
import kornia
import matplotlib.pyplot as plt

def enhance_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return enhance_contrast(img)

def estimate_pose_diff_ransac(src_pts, dst_pts, camera_matrix, threshold=0.001, max_iter=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K_tensor = torch.from_numpy(camera_matrix).float().unsqueeze(0).to(device)  # [1, 3, 3]
    
    # 确保输入点形状为 [1, N, 2]
    src_pts_tensor = torch.from_numpy(src_pts).float().view(1, -1, 2).to(device)
    dst_pts_tensor = torch.from_numpy(dst_pts).float().view(1, -1, 2).to(device)

    # 归一化坐标
    src_norm = kornia.geometry.normalize_points(src_pts_tensor, K_tensor[:, :3, :3])
    dst_norm = kornia.geometry.normalize_points(dst_pts_tensor, K_tensor[:, :3, :3])

    # 可微分RANSAC
    E, mask = kornia.geometry.find_essential_ransac(
        src_norm,
        dst_norm,
        threshold=threshold,
        max_iterations=max_iter
    )

    # 恢复位姿
    R_rel, t_rel = kornia.geometry.recover_relative_camera_pose(
        src_norm,
        dst_norm,
        E,
        mask,
        K_tensor[:, :3, :3]
    )

    # 构建变换矩阵
    T = torch.eye(4, device=device)
    T[:3, :3] = R_rel.squeeze(0)
    T[:3, 3] = t_rel.squeeze(0)
    return T.cpu().numpy(), mask.squeeze(0).cpu().numpy().astype(np.uint8)

def get_feature_matches(img1, img2):
    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    matches = flann.knnMatch(des1, des2, k=2)

    ratio_thresh = 0.6
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    return good_matches, kp1, kp2

def rotation_error(R1, R2):
    trace = np.trace(R1.T @ R2)
    return np.rad2deg(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))

def translation_direction_error(t1, t2):
    norm1 = np.linalg.norm(t1)
    norm2 = np.linalg.norm(t2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 180.0
    return np.rad2deg(np.arccos(np.clip(np.dot(t1/norm1, t2/norm2), -1.0, 1.0)))

def compute_reprojection_error(T, K, matches, kp1, kp2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    K_tensor = torch.from_numpy(K).float().unsqueeze(0).to(device)
    src_tensor = torch.from_numpy(src_pts).float().view(1, -1, 2).to(device)
    dst_tensor = torch.from_numpy(dst_pts).float().view(1, -1, 2).to(device)
    
    src_norm = kornia.geometry.normalize_points(src_tensor, K_tensor[:, :3, :3])
    dst_norm = kornia.geometry.normalize_points(dst_tensor, K_tensor[:, :3, :3])
    
    R = torch.from_numpy(T[:3, :3]).float().to(device)
    t = torch.from_numpy(T[:3, 3]).float().to(device)
    
    points_3d = kornia.geometry.triangulate_points(
        torch.eye(3, 4).unsqueeze(0).to(device),
        torch.cat([R, t.view(1, 3, 1)], dim=2),
        src_norm,
        dst_norm
    )[..., :3]
    
    projected = kornia.geometry.project_points(points_3d, R.unsqueeze(0), t.unsqueeze(0), K_tensor[:, :3, :3])
    errors = torch.norm(projected - dst_tensor, dim=-1)
    return errors.mean().item()

def main():
    camera_matrix = np.array([
        [211.949, 0, 127.933],
        [0, 211.949, 95.9333],
        [0, 0, 1]
    ], dtype=np.float32)

    img1 = cv2.imread("./41069021_305.377.png")
    img2 = cv2.imread("./41069021_306.360.png")
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像，请检查路径")

    gray1 = preprocess_image(img1)
    gray2 = preprocess_image(img2)

    matches, kp1, kp2 = get_feature_matches(gray1, gray2)
    print(f"初始匹配数量: {len(matches)}")

    if len(matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        T_est, mask = estimate_pose_diff_ransac(src_pts, dst_pts, camera_matrix)
        valid_matches = [m for m, flag in zip(matches, mask) if flag]
    else:
        print("匹配点不足")
        return

    T_gt = np.array([
        [0.99981151, 0.01853355, -0.00578381, 0.00737292],
        [-0.0145849, 0.91360526, 0.40634063, 0.02877406],
        [0.01281505, -0.40617969, 0.91370336, 0.37641721],
        [0, 0, 0, 1]
    ])

    rotation_err = rotation_error(T_est[:3, :3], T_gt[:3, :3])
    translation_err = translation_direction_error(T_est[:3, 3], T_gt[:3, 3])
    reproj_err = compute_reprojection_error(T_est, camera_matrix, valid_matches, kp1, kp2)
    
    print("\n=== 误差分析 ===")
    print(f"旋转误差: {rotation_err:.4f} 度")
    print(f"平移方向误差: {translation_err:.4f} 度")
    print(f"平均重投影误差: {reproj_err:.4f} 像素")

    draw_params = dict(matchColor=(0,255,0), 
                      singlePointColor=(255,0,0),
                      matchesMask=mask.tolist(),
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    vis = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Valid Matches: {sum(mask)}")
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()