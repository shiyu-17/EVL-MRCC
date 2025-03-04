import numpy as np
from scipy.spatial.transform import Rotation as R

# 第一帧数据
timestamp_1 = 305.37738108
rotation_vector_1 = np.array([2.1562335342945733, -2.117006787798841, 0.24537857547787853])
translation_vector_1 = np.array([0.0519758, 0.0541097, 0.0481458])

# 第二帧数据
timestamp_2 = 306.36031217
rotation_vector_2 = np.array([1.8937752059516033, -1.8785065570788495, 0.6343465703380557])
translation_vector_2 = np.array([0.0600633, 0.0970145, 0.399096])

# 1. 将旋转向量转换为旋转矩阵
rot_1 = R.from_rotvec(rotation_vector_1).as_matrix()  # R1: 第一帧的旋转矩阵
rot_2 = R.from_rotvec(rotation_vector_2).as_matrix()  # R2: 第二帧的旋转矩阵

# 2. 计算相对旋转矩阵 (R_rel = R2 * R1^T)
R_rel = rot_2 @ rot_1.T  # 矩阵乘法

# 3. 计算相对平移向量 (t_rel = t2 - R_rel * t1)
t_rel = translation_vector_2 - R_rel @ translation_vector_1

# 4. 将相对旋转矩阵转换为旋转向量（可选）
rotation_vector_rel = R.from_matrix(R_rel).as_rotvec()

# 2. 构建4×4齐次变换矩阵 
T_rel = np.eye(4)
T_rel[:3, :3] = R_rel
T_rel[:3, 3] = t_rel  # 填充平移部分

# 输出结果
print("相对旋转矩阵 (R_rel):\n", R_rel)
print("\n相对平移向量 (t_rel):", t_rel)
print("\n相对旋转向量 (旋转向量表示):", rotation_vector_rel)
print ( "4×4 相对位姿矩阵 T_rel:\n", T_rel )