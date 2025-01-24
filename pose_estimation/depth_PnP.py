import numpy as np
import cv2

def vec_to_transform(rotation_vector, translation_vector):
    """将旋转向量和平移向量转换为4x4变换矩阵"""
    R, _ = cv2.Rodrigues(rotation_vector)  # 旋转向量转旋转矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation_vector
    return T

# 输入数据
rotation_vec1 = np.array([2.1562335342945733, -2.117006787798841, 0.24537857547787853])
translation_vec1 = np.array([0.0519758, 0.0541097, 0.0481458])

rotation_vec2 = np.array([2.1025062009661832, -1.9512261212443467, 0.4039279627048044])
translation_vec2 = np.array([0.0515506, 0.0759061, 0.197158])

# 转换为变换矩阵
T1 = vec_to_transform(rotation_vec1, translation_vec1)
T2 = vec_to_transform(rotation_vec2, translation_vec2)

# 计算相对变换矩阵
T1_inv = np.linalg.inv(T1)
T_rel = T1_inv @ T2

# 输出结果
print("相对变换矩阵T_rel:")
print(T_rel)
