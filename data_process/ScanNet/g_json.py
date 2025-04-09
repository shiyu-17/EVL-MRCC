import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData
from scipy.spatial import ConvexHull

def load_point_cloud(ply_path):
    """加载PLY格式的点云数据"""
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    return np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

def compute_obb(points):
    """通过PCA计算OBB"""
    # 计算凸包避免异常点
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # 计算PCA
    cov_matrix = np.cov(hull_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 按特征值降序排序
    order = eigenvalues.argsort()[::-1]
    axes = eigenvectors[:, order]
    
    # 将点转换到新坐标系
    rotated = np.dot(hull_points, axes)
    
    # 计算边界范围
    min_vals = rotated.min(axis=0)
    max_vals = rotated.max(axis=0)
    size = max_vals - min_vals
    
    # 计算中心点（在新坐标系中）
    center = (min_vals + max_vals) / 2
    # 转换回原始坐标系
    centroid = np.dot(center, axes)
    
    return {
        "centroid": centroid.tolist(),
        "axesLengths": size.tolist(),
        "normalizedAxes": axes.flatten().tolist()
    }

def scannet_to_custom_annotation(scan_id, scannet_path, output_path):
    # 加载点云数据
    ply_path = os.path.join(scannet_path, f"{scan_id}_vh_clean_2.ply")
    point_cloud = load_point_cloud(ply_path)
    
    # 加载原始注释文件
    agg_file = os.path.join(scannet_path, f"{scan_id}.aggregation.json")
    seg_file = os.path.join(scannet_path, f"{scan_id}_vh_clean_2.0.010000.segs.json")
    
    with open(agg_file) as f:
        agg_data = json.load(f)
    with open(seg_file) as f:
        seg_data = json.load(f)
    
    # 生成点云索引映射
    seg_indices = np.array(seg_data['segIndices'])
    
    custom_annotations = {"data": []}
    for obj in agg_data['segGroups']:
        try:
            # 尝试获取原始OBB
            obb = obj['obb']
            centroid = np.array(obb['centroid'])
            axes_lengths = np.array(obb['axesLengths'])
            normalized_axes = R.from_quat(obb['normalizedAxes']).as_matrix().flatten().tolist()
        except KeyError:
            # 自动生成OBB
            print(f"为对象 {obj['id']} 生成OBB...")
            
            # 获取物体点云
            segments = []
            for seg_id in obj['segments']:
                segments.extend(np.where(seg_indices == seg_id)[0].tolist())
            obj_points = point_cloud[segments]
            
            # 计算OBB
            obb = compute_obb(obj_points)
            centroid = np.array(obb["centroid"])
            axes_lengths = np.array(obb["axesLengths"])
            normalized_axes = obb["normalizedAxes"]
        
        # 构建自定义格式
        custom_obj = {
            "uid": f"{scan_id}_{obj['id']}",
            "label": obj['label'],
            "segments": {
                "obb": {
                    "centroid": centroid.tolist(),
                    "axesLengths": axes_lengths.tolist(),
                    "normalizedAxes": normalized_axes
                },
                "segments": [int(i) for i in segments]  # 转换为Python原生int类型
            }
        }
        custom_annotations["data"].append(custom_obj)
    
    # 保存转换后的注释
    output_file = os.path.join(output_path, f"{scan_id}_3dod_annotation.json")
    with open(output_file, 'w') as f:
        json.dump(custom_annotations, f, indent=2)
    print(f"已保存到: {output_file}")

# 示例用法
scannet_to_custom_annotation(
    scan_id="scene0000_00", 
    scannet_path="/Users/shiyu/mycode/data/scans/scene0000_00/",
    output_path="/Users/shiyu/mycode/data/pic"
)