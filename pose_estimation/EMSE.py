import numpy as np
from scipy.spatial.transform import Rotation as R

# 计算旋转矩阵的角度误差
def rotation_error(R1, R2):
    # 计算旋转矩阵之间的旋转角度误差
    # trace(R1^T * R2) 计算的是旋转角度的余弦值
    trace = np.trace(R1.T @ R2)
    return np.arccos((trace - 1) / 2)

# 计算位移向量的欧几里得距离
def translation_error(t1, t2):
    return np.linalg.norm(t1 - t2)

# 计算均方误差（MSE）或者均方根误差（RMSE）
def compute_error(T_relative, T_icp):
    # 从矩阵中提取旋转矩阵和位移向量
    R_relative = T_relative[:3, :3]
    t_relative = T_relative[:3, 3]
    
    R_icp = T_icp[:3, :3]
    t_icp = T_icp[:3, 3]
    
    # 计算旋转误差
    rotation_err = rotation_error(R_relative, R_icp)
    
    # 计算位移误差
    translation_err = translation_error(t_relative, t_icp)
    
    # 计算总的误差（可以使用加权的MSE或RMSE）
    rmse_rotation = rotation_err
    rmse_translation = translation_err
    
    # 输出误差
    print(f"旋转误差 : {rotation_err}")
    print(f"位移误差 : {translation_err}")
    print(f"旋转误差 RMSE: {rmse_rotation}")
    print(f"位移误差 RMSE: {rmse_translation}")

    return rmse_rotation, rmse_translation

# 示例数据
T_relative = np.array([[0.99981151, 0.01853355, -0.00578381, 0.00737292],
                       [-0.0145849, 0.91360526, 0.40634063, 0.02877406],
                       [0.01281505, -0.40617969, 0.91370336, 0.37641721],
                       [0, 0, 0, 1]])



# T_icp = np.array([[0.96094112, -0.26657813, 0.07435236, 0.00152355],
#                   [0.24352769, 0.94212209, 0.23043487, 0.00356736],
#                   [-0.1314779, -0.20332748, 0.97024301, 0.00726166],
#                   [0, 0, 0, 1]])
# # gicp
T_icp = np.array([[0.99778408204936, 0.01810645255084438, 0.06402407347974827, -0.5020424787397049],
                  [-0.02114779344518273, 0.9986639798241299, 0.04714897914302863, 0.01512423332018475],
                  [-0.06308483527216399, -0.04839846875526001, 0.9968339339031499, 0.8647107071774753],
                  [0, 0, 0, 1]])







# 计算误差
compute_error(T_relative, T_icp)
