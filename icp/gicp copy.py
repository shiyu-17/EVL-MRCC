#!/usr/bin/python3
import os
import time
import numpy as np
import PIL.Image
import small_gicp

# 从文件加载相机内参
def st2_camera_intrinsics(filename):
    """
    从文件加载相机内参并返回相机内参矩阵
    :param filename: 存储内参的文件路径
    :return: 相机内参矩阵 (3x3)
    """
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])

# 将深度图转换为点云
def depth_to_point_cloud(depth_img, intrinsics):
    """
    将深度图转换为3D点云
    :param depth_img: 深度图 (高度 x 宽度)
    :param intrinsics: 相机内参矩阵
    :return: 转换后的点云（N x 3）
    """
    # 确保深度图是单通道图像
    if len(depth_img.shape) == 3:
        depth_img = depth_img[:, :, 0]  # 只取第一个通道

    # print("Depth Image  shape:", depth_img.shape)

    h, w = depth_img.shape
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    points = []
    
    for v in range(h):
        for u in range(w):
            Z = depth_img[v, u]  # 深度值
            if Z == 0:  # 跳过无效深度
                continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])
    
    return np.array(points)

# 基于GICP的扫描-扫描配准估计
class ScanToScanMatchingOdometry(object):
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.T_last_current = np.identity(4)
        self.T_world_lidar = np.identity(4)
        self.target = None
    
    def estimate(self, raw_points0, raw_points):
        downsampled, tree = small_gicp.preprocess_points(raw_points, 0.1, num_threads=self.num_threads)
        
        self.target = small_gicp.preprocess_points(raw_points0, 0.1, num_threads=self.num_threads)

        result = small_gicp.align(self.target[0], downsampled, self.target[1], self.T_last_current, num_threads=self.num_threads)
        
        self.T_last_current = result.T_target_source
        self.T_world_lidar = self.T_world_lidar @ result.T_target_source
        self.target = (downsampled, tree)
        
        return self.T_world_lidar

def main():
    # 直接硬编码路径
    image1_path = './41069021_305.377.png'
    depth1_path = './41069021_305.377.src_depth.jpg'
    image2_path = './41069021_306.360.png'
    depth2_path = './41069021_306.360.src_depth.jpg'
    intrinsics_path = './41069021_305.377.pincam'

    num_threads = 4  # 默认线程数

    # 加载图像和深度图
    image1 = PIL.Image.open(image1_path)
    depth1 = np.array(PIL.Image.open(depth1_path), dtype=np.float32)
    image2 = PIL.Image.open(image2_path)
    depth2 = np.array(PIL.Image.open(depth2_path), dtype=np.float32)
    
    # 从文件加载相机内参
    intrinsics = st2_camera_intrinsics(intrinsics_path)
   
    # print("Camera Intrinsics Matrix:")
    # print(intrinsics)

    # 将深度图转换为点云
    raw_points1 = depth_to_point_cloud(depth1, intrinsics)
    raw_points2 = depth_to_point_cloud(depth2, intrinsics)

    print("Point Cloud 1 centroid:", np.mean(raw_points1, axis=0))
    print("Point Cloud 2 centroid:", np.mean(raw_points2, axis=0))


    # print(f"Raw points 1 count: {len(raw_points1)}")
    # print(f"Raw points 2 count: {len(raw_points2)}")

    # 初始化 GICP 配准
    odom = ScanToScanMatchingOdometry(num_threads)

    # 检查 GICP 配准前的点云数据
    # print(f"Initial point cloud 1 count: {len(raw_points1)}")
    # print(f"Initial point cloud 2 count: {len(raw_points2)}")
    
    # 进行配准，计算两张图片之间的相对位姿
    t1 = time.time()
    T = odom.estimate(raw_points1, raw_points2)  # 以第一张图为参考，估计第二张图的位置
    t2 = time.time()

    # 打印输出相对位姿
    print("相对位姿矩阵（从第一张图到第二张图）：")
    print(T)
    print(f"Time taken: {t2 - t1:.3f} seconds")
    
    # # 可视化
    # viewer = guik.viewer()
    # viewer.disable_vsync()

    # viewer.lookat(T[:3, 3])
    # viewer.update_drawable('points', glk.create_pointcloud_buffer(raw_points2), guik.FlatOrange(T).add('point_scale', 2.0))

    # # 显示位姿更新
    # viewer.update_drawable('pos', glk.primitives.coordinate_system(), guik.VertexColor(T))
    # viewer.append_text('Time: {:.3f} sec'.format(t2 - t1))

    # # 开始渲染
    # if not viewer.spin_once():
    #     return

if __name__ == '__main__':
    main()
