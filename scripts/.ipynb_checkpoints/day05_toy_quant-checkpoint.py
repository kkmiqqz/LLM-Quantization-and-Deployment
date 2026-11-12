import torch
import numpy as np

def symmetric_quantize(w):
    print(f"\n--- 原始数据: {w} ---")
    # 1. 计算 Scale (对称量化，取绝对值的最大值除以 127)
    max_val = np.max(np.abs(w))
    # 防止全0数组除以0
    scale = max_val / 127.0 if max_val != 0 else 1.0
    
    print(f"计算出的 Scale: {scale:.6f}")
    
    # 2. 量化 (除以 scale，四舍五入，并限制在 -127 到 127 之间)
    # 相当于把浮点数挤进 8位整数 的盒子里
    int_w = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    print(f"INT8 整数数组: {int_w}")
    
    # 3. 反量化 (乘回 scale，恢复成浮点数)
    dequant_w = int_w * scale
    print(f"反量化后的浮点数组: {dequant_w}")
    
    # 4. 计算误差
    errors = np.abs(w - dequant_w)
    mean_abs_error = np.mean(errors)
    max_abs_error = np.max(errors)
    mse = np.mean(errors ** 2)
    
    print(f"平均绝对误差 (Mean Absolute Error): {mean_abs_error:.6f}")
    print(f"最大绝对误差 (Max Absolute Error): {max_abs_error:.6f}")
    print(f"均方误差 (MSE): {mse:.6f}")
    
    return mean_abs_error

# 测试数据 1：正常的数组
normal_w = np.array([-1.0, -0.4, 0, 0.3, 0.9], dtype=np.float32)
symmetric_quantize(normal_w)

# 测试数据 2：加入一个巨大的离群值 (Outlier) 10.0
outlier_w = np.array([-1.0, -0.4, 0, 0.3, 10.0], dtype=np.float32)
symmetric_quantize(outlier_w)