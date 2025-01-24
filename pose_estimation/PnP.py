import numpy as np
import cv2

def st2_camera_intrinsics(filename):
    """从内参文件构建相机内参矩阵"""
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])

def generate_point(depth_image, intrinsic, subsample=1):
    """从深度图生成3D点"""

    # 检查 depth_image 是否为三维
    if len(depth_image.shape) == 3:
        print("警告: 深度图有 3 个通道，正在转换为单通道...")
        # 假设深度信息在第一个通道，或者转换为灰度图
        depth_image = depth_image[:, :, 0]  # 提取第一个通道
        # 或者使用 OpenCV 转换为灰度图：
        # depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)

    # 获取深度图的高和宽
    height, width = depth_image.shape
    print("深度图形状:", height, width)

    intrinsic_4x4 = np.identity(4)
    intrinsic_4x4[:3, :3] = intrinsic

    # 子采样深度图（如果 subsample > 1）
    if subsample > 1:
        depth_image = depth_image[::subsample, ::subsample]  # 子采样

    # 生成像素坐标网格
    u, v = np.meshgrid(
        np.arange(0, width),
        np.arange(0, height),
    )

    # 展平网格坐标和深度图
    u = u.flatten()
    v = v.flatten()
    d = depth_image.flatten()

    # 筛选有效深度值（非零的深度值）
    d_filter = d > 0
    u_valid = u[d_filter]
    v_valid = v[d_filter]
    d_valid = d[d_filter]

    # 构建齐次坐标矩阵
    mat = np.vstack((
        (u_valid - intrinsic[0, 2]) * d_valid / intrinsic[0, 0],
        (v_valid - intrinsic[1, 2]) * d_valid / intrinsic[1, 1],
        d_valid,
        np.ones_like(u_valid)
    ))

    # 转换到相机坐标系
    new_points_3d = np.dot(np.linalg.inv(intrinsic_4x4), mat)[:3]

    return new_points_3d.T, np.vstack((u_valid, v_valid)).T

def solve_pnp(points_3d_1, points_2d_1, intrinsic):
    """使用cv2.solvePnP计算旋转矩阵和平移向量"""
    success, rvec, tvec = cv2.solvePnP(points_3d_1, points_2d_1, intrinsic, None)
    if success:
        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec
    else:
        raise ValueError("PnP 求解失败")

def main():
    # 1. 读取相机内参
    intrinsic_filename = '41069021_305.377.pincam'  # 你的相机内参文件路径
    K = st2_camera_intrinsics(intrinsic_filename)
    print(f"相机内参矩阵 K:\n{K}")

    # 2. 读取两张深度图（例如：depth_image_1 和 depth_image_2）
    depth_image_1 = cv2.imread('./41069021_305.377.src_depth.jpg', cv2.IMREAD_UNCHANGED)  # 读取第一张深度图
    depth_image_2 = cv2.imread('./41069021_306.360.src_depth.jpg', cv2.IMREAD_UNCHANGED)  # 读取第二张深度图

    # print("depth_image 形状:", depth_image_1.shape)
    # print("depth_image 数据类型:", type(depth_image_1))

    # 3. 从深度图生成3D点和2D点
    points_3d_1, points_2d_1 = generate_point(depth_image_1, K)
    points_3d_2, points_2d_2 = generate_point(depth_image_2, K)

    points_3d_1 = np.array(points_3d_1, dtype=np.float32)
    points_2d_1 = np.array(points_2d_1, dtype=np.float32)

    # 确保它们的形状是符合要求的
    print(f"3D 点的形状: {points_3d_1.shape}, 2D 点的形状: {points_2d_1.shape}")

    # print(f"3D 点数量: {len(points_3d_1)}, 2D 点数量: {len(points_2d_1)}")
    # if len(points_3d_1) < 4 or len(points_2d_1) < 4:
    #     raise ValueError("3D 或 2D 点数量不足，solvePnP 至少需要 4 对点。")

    success, rvec, tvec = cv2.solvePnP(points_3d_1, points_2d_1, K, None)
    print(f"PnP 结果：success={success}, rvec={rvec}, tvec={tvec}")

    # 4. 使用PnP计算相机位姿（旋转矩阵和平移向量）
    R1, t1 = solve_pnp(points_3d_1, points_2d_1, K)
    R2, t2 = solve_pnp(points_3d_2, points_2d_2, K)

    # 5. 输出位姿信息
    print("\n第一帧相机位姿：")
    print("旋转矩阵 R1:\n", R1)
    print("平移向量 t1:\n", t1)

    print("\n第二帧相机位姿：")
    print("旋转矩阵 R2:\n", R2)
    print("平移向量 t2:\n", t2)

    # 6. 计算从第一帧到第二帧的相机位姿变换
    T1 = np.hstack((R1, t1))
    T1 = np.vstack((T1, np.array([0, 0, 0, 1])))
    T2 = np.hstack((R2, t2))
    T2 = np.vstack((T2, np.array([0, 0, 0, 1])))

    # 计算位姿变换矩阵 T12 (从帧1到帧2)
    T12 = np.dot(np.linalg.inv(T1), T2)
    print("\n从第一帧到第二帧的位姿变换矩阵 T12:\n", T12)

if __name__ == "__main__":
    main()
