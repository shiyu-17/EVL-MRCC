import numpy as np
import cv2

# 给定的旋转向量和平移向量
rotation_vector_1 = np.array([2.327592958220677 ,2.0173072386484425, 0.18297520284276966])
translation_vector_1 = np.array([0.0859419, 0.1264 ,-0.168066])

rotation_vector_2 = np.array([2.3579397410461005 ,1.9944091778725408, 0.12878042999005843])
translation_vector_2 = np.array([0.100422, 0.122491, -0.127448])

# 将旋转向量转换为旋转矩阵
R1, _ = cv2.Rodrigues(rotation_vector_1)
R2, _ = cv2.Rodrigues(rotation_vector_2)

# 构造4x4的变换矩阵T1和T2
T1 = np.eye(4)
T1[:3, :3] = R1
T1[:3, 3] = translation_vector_1

T2 = np.eye(4)
T2[:3, :3] = R2
T2[:3, 3] = translation_vector_2

# 计算T1的逆矩阵
inv_T1 = np.eye(4)
inv_T1[:3, :3] = R1.T
inv_T1[:3, 3] = -R1.T @ translation_vector_1

# 计算相对位姿变换矩阵
relative_pose = T2 @ inv_T1

# 打印结果，保留六位小数
np.set_printoptions(precision=6, suppress=True)
print("相对位姿变换矩阵：\n", relative_pose)