import cv2
import numpy as np
from open3d import geometry, pipelines

def read_st2_intrinsic(filename):
    """
    读取特定格式的相机内参文件
    参数文件格式示例：
    1280 720 620.857 469.167 640.0 360.0
    """
    w, h, fx, fy, cx, cy = map(float, open(filename).read().split())
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])
    return K

def read_opencv_intrinsic(filename):
    """原始OpenCV内参读取函数"""
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    K = fs.getNode('camera_matrix').mat()
    dist = fs.getNode('distortion_coefficients').mat()
    fs.release()
    return K, dist

def read_intrinsics(file_path, format='opencv'):
    """统一内参读取接口"""
    if format == 'st2':
        K = read_st2_intrinsic(file_path)
        # 根据实际情况添加dist，默认假设无畸变
        dist = np.zeros((4,1))  # [k1, k2, p1, p2]
    elif format == 'opencv':
        K, dist = read_opencv_intrinsic(file_path)
    else:
        raise ValueError("Unsupported format")
    
    return K, dist

def undistort_point(u, v, K, dist):
    """对单个点进行去畸变处理"""
    points = np.array([[u, v]], dtype=np.float32)
    undistorted = cv2.undistortPoints(points, K, dist)
    return undistorted[0][0]

def pixel_to_3d(u, v, depth_value, K):
    """将像素坐标和深度值转换为3D坐标"""
    x = (u - K[0][2]) / K[0][0]
    y = (v - K[1][2]) / K[1][1]
    return np.array([depth_value * x, depth_value * y, depth_value])

def depth_to_pointcloud(depth_img, K):
    """将深度图转换为点云"""
    rows, cols = depth_img.shape
    points = []
    for v in range(rows):
        for u in range(cols):
            d = depth_img[v][u]
            if d == 0:
                continue
            pt = pixel_to_3d(u, v, d, K)
            points.append(pt)
    return geometry.PointCloud(np.array(points))

def main():
    
    # 加载 RGB 图像和深度图
    img1 = cv2.imread("./41069021_305.377.png")
    depth1 = cv2.imread("./41069021_305.377.src_depth.jpg", cv2.IMREAD_UNCHANGED)
    depth1 = depth1 [ : , : , 0 ] # 移除第三个维度
    img2 = cv2.imread("./41069021_306.360.png")
    depth2 = cv2.imread("./41069021_306.360.src_depth.jpg", cv2.IMREAD_UNCHANGED)
    depth2 = depth2 [ : , : , 0 ] # 移除第三个维度
    
    if not (img1 is not None and img2 is not None and depth1 is not None and depth2 is not None):
        raise FileNotFoundError("Input files not found")
    
    # 读取相机内参
    K, dist = read_intrinsics('./41069021_305.377.pincam', format='st2')  # 根据实际文件格式选择


    # 特征点提取与匹配
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)  # 正确返回 DMatch 列表

    # 打印调试信息
    print("Matches type:", type(matches))
    if matches:
        print("First match distance:", matches[0].distance)
    else:
        print("No matches found")
        return

    # RANSAC筛选最佳匹配点
    inlier_matches = []
    max_inliers = 200
    for m in matches:
        if len(inlier_matches) >= max_inliers:
            break
        inlier_matches.append(m)
    
    # 提取匹配点对
    rows1 , cols1 = depth1 . shape 
    rows2 , cols2 = depth2 . shape
    p1 = []
    p2 = []
    d1 = []
    d2 = []
    for m in inlier_matches:
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt
        if not ( 0 <= u1 < cols1 and 0 <= v1 < rows1 ) : 
            continue 
        if not ( 0 <= u2 < cols2 and 0 <= v2 < rows2 ) : 
            continue
        p1.append((u1, v1))
        p2.append((u2, v2))
        d1.append(depth1[int(v1), int(u1)])
        d2.append(depth2[int(v2), int(u2)])
    
    if len(p1) < 4:
        raise ValueError("Not enough inlier points for PnP")
    
    # 生成3D点云
    points1 = []
    for (u, v), d in zip(p1, d1):
        pt = pixel_to_3d(u, v, d, K)
        points1.append(pt)
    points1 = np.array(points1)
    
    # 去畸变处理
    img_pts = []
    for u, v in p1:
        xu, yu = undistort_point(u, v, K, dist)
        img_pts.append((xu, yu))
    img_pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
    
    # PnP求解初始位姿
    retval, rvec, tvec = cv2.solvePnP(points1, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    R_initial = cv2.Rodrigues(rvec)
    T_initial = tvec
    
    # ICP优化
    pcd1 = depth_to_pointcloud(depth1, K)
    pcd2 = depth_to_pointcloud(depth2, K)
    
    initial_transformation = np.eye(4)
    initial_transformation[:3, :3] = R_initial
    initial_transformation[:3, 3] = T_initial
    
    icp = pipelines.registration.ICPRegistration(
        pcd1,
        pcd2,
        max_correspondence_distance=0.1,  # 根据实际数据调整
        initial_transformation=initial_transformation,
        criteria=pipelines.registration.RegistrationCriteria(
            number_of_iterations=100,
            relative_tolerance=1e-6,
            absolute_tolerance=1e-6
        )
    )
    result = icp.register(pcd1, pcd2)
    final_R = result.rotation_matrix
    final_T = result.translation
    
    # 输出结果
    print("Final Rotation Matrix:\n", final_R)
    print("Final Translation Vector:\n", final_T)
    
    # 转换为欧拉角（可选）
    def rotation_to_euler(R):
        sy = np.sqrt(R[0][0]**2 + R[1][0]**2)
        cx = R[0][0]/sy if sy !=0 else 1
        cy = R[1][0]/sy if sy !=0 else 0
        cz = R[2][0]/sy if sy !=0 else 0
        
        angle_x = np.arctan2(R[2][1], R[2][2])
        angle_y = np.arctan2(-R[0][2], R[2][2])
        angle_z = np.arctan2(R[1][2], R[0][2])
        
        return np.degrees(angle_x), np.degrees(angle_y), np.degrees(angle_z)
    
    euler_angles = rotation_to_euler(final_R)
    print(f"Euler Angles (X, Y, Z): {euler_angles[0]:.2f}°, {euler_angles[1]:.2f}°, {euler_angles[2]:.2f}°")
    print(f"Translation: ({final_T[0]:.2f}, {final_T[1]:.2f}, {final_T[2]:.2f}) m")

if __name__ == "__main__":
    main()