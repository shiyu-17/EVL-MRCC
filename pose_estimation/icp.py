import open3d as o3d
import numpy as np
import cv2

# 从文件加载相机内参
def st2_camera_intrinsics(filename):
    """
    从文件加载相机内参并返回相机内参矩阵
    :param filename: 存储内参的文件路径
    :return: 相机内参矩阵 (3x3)
    """
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]]), int(w), int(h)

def depth_to_point_cloud(rgb_image, depth_image, intrinsics):
    """
    将深度图和 RGB 图像转换为点云。
    :param rgb_image: RGB 图像 (H, W, 3)
    :param depth_image: 深度图 (H, W)
    :param intrinsics: 相机内参矩阵 (3x3)
    :return: Open3D 点云对象
    """
    h, w = depth_image.shape
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    # 创建像素网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_image / 1000.0  # 假设深度图单位是毫米，转换为米
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 有效深度点掩码
    valid_mask = z > 0
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

    # 颜色信息
    rgb = rgb_image[valid_mask]

    # 创建点云
    points = np.stack((x, y, z), axis=-1)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb / 255.0)  # 归一化 RGB 到 [0,1]

    return point_cloud

def compute_icp(source_pcd, target_pcd):
    """
    使用 ICP 算法计算两帧点云之间的变换
    :param source_pcd: 源点云
    :param target_pcd: 目标点云
    :return: 变换矩阵和配准结果
    """
    # Step 1: 计算法向量
    print("计算目标点云法向量...")
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Step 2: 确保法向量方向与点云一致
    target_pcd.orient_normals_consistent_tangent_plane(k=30)

    # ICP 参数
    threshold = 0.02  # 配准的距离阈值
    trans_init = np.eye(4)  # 初始变换矩阵

    # Step 3: 执行 ICP 配准
    print("开始执行 ICP 配准...")
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return icp_result

def preprocess_point_cloud(pcd, voxel_size=0.01):
    # 降采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # 去噪
    pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd_clean

if __name__ == "__main__":
    # 从文件加载相机内参
    intrinsics_file = "./41069021_305.377.pincam"  # 替换为你的内参文件路径
    intrinsics, img_width, img_height = st2_camera_intrinsics(intrinsics_file)

    # 加载 RGB 图像和深度图
    rgb_image1 = cv2.imread("./41069021_305.377.png")
    depth_image1 = cv2.imread("./41069021_305.377.src_depth.jpg", cv2.IMREAD_UNCHANGED)
    rgb_image2 = cv2.imread("./41069021_306.360.png")
    depth_image2 = cv2.imread("./41069021_306.360.src_depth.jpg", cv2.IMREAD_UNCHANGED)

    # 检查深度图是否为单通道
    if len(depth_image1.shape) == 3 and depth_image1.shape[2] == 3:
        depth_image1 = cv2.cvtColor(depth_image1, cv2.COLOR_BGR2GRAY)
    if len(depth_image2.shape) == 3 and depth_image2.shape[2] == 3:
        depth_image2 = cv2.cvtColor(depth_image2, cv2.COLOR_BGR2GRAY)

    # 调整深度图分辨率
    if depth_image1.shape != (img_height, img_width):
        depth_image1 = cv2.resize(depth_image1, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    if depth_image2.shape != (img_height, img_width):
        depth_image2 = cv2.resize(depth_image2, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

    # 确保图像分辨率和内参一致
    assert rgb_image1.shape[:2] == (img_height, img_width)
    assert depth_image1.shape == (img_height, img_width)

    # 打印计算位姿所用的图像文件名
    # print(f"计算相对位姿，使用的两张图像: {rgb_image1}, {rgb_image2}")

    # 转换为点云
    pcd1 = depth_to_point_cloud(rgb_image1, depth_image1, intrinsics)
    pcd2 = depth_to_point_cloud(rgb_image2, depth_image2, intrinsics)

    pcd1 = preprocess_point_cloud(pcd1)
    pcd2 = preprocess_point_cloud(pcd2)

    # 可视化点云（可选）
    # o3d.visualization.draw_geometries([pcd1, pcd2])

    # 使用 ICP 计算相对位姿 
    icp_result = compute_icp(pcd1, pcd2)

    print("ICP 相对位姿矩阵:")
    print(icp_result.transformation)

    # 可视化配准结果
    # pcd1.transform(icp_result.transformation)
    # o3d.visualization.draw_geometries([pcd1, pcd2])
