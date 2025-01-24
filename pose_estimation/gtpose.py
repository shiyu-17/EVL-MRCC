import numpy as np
from scipy.spatial.transform import Rotation as R

# 第一帧数据 (时间戳，旋转向量，平移向量)
timestamp_1 = 305.37738108
rotation_vector_1 = np.array([2.1562335342945733, -2.117006787798841, 0.24537857547787853])
translation_vector_1 = np.array([0.0519758, 0.0541097, 0.0481458])

# 第二帧数据 (时间戳，旋转向量，平移向量)
timestamp_2 = 306.36031217
rotation_vector_2 = np.array([1.8937752059516033, -1.8785065570788495, 0.6343465703380557])
translation_vector_2 = np.array([0.0600633, 0.0970145, 0.399096])

# 将旋转向量转换为旋转矩阵
rotation_matrix_1 = R.from_rotvec(rotation_vector_1).as_matrix()
rotation_matrix_2 = R.from_rotvec(rotation_vector_2).as_matrix()

# 构造齐次变换矩阵 T1 和 T2
T1 = np.eye(4)
T1[:3, :3] = rotation_matrix_1
T1[:3, 3] = translation_vector_1

T2 = np.eye(4)
T2[:3, :3] = rotation_matrix_2
T2[:3, 3] = translation_vector_2

# 计算 T2 相对于 T1 的相对位姿矩阵 T_relative = T2 * T1^-1
T1_inv = np.linalg.inv(T1)
T_relative = T2 @ T1_inv

# 输出相对位姿的4x4矩阵
print("从第一帧到第二帧的相对位姿矩阵 (4x4):\n", T_relative)
