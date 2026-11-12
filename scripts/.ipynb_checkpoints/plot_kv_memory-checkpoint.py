import matplotlib.pyplot as plt
import os

# 确保 results 文件夹存在
os.makedirs("results", exist_ok=True)

# 1. 准备数据
seq_lengths = [128, 512, 1024, 2048]

# 理论 KV Cache 占用 (单位: MB)
theoretical_kv = [l * 28 / 1024 for l in seq_lengths]

# 你的实测数据
pure_kv_memory = [11.66, 22.16, 36.16, 64.16]    
peak_memory = [124.22, 469.03, 930.91, 1852.16]  

# 权重基础显存
base_weight_vram = 3393.29

# 2. 设置绘图参数
plt.figure(figsize=(10, 6), dpi=150) 

# 画三条线
plt.plot(seq_lengths, theoretical_kv, marker='^', linewidth=2, color='blue', linestyle=':', label='Theoretical KV Cache')
plt.plot(seq_lengths, pure_kv_memory, marker='o', linewidth=2, color='green', label='Measured Pure KV Cache')
plt.plot(seq_lengths, peak_memory, marker='s', linewidth=2, color='red', linestyle='--', label='Forward Peak Memory (Includes Temp)')

#强制横坐标只显示你指定的这些数据点
plt.xticks(seq_lengths, [str(s) for s in seq_lengths])

# 3. 美化图表
plt.title('Qwen2.5-1.5B: Memory Usage vs Input Sequence Length (FP16)', fontsize=14, fontweight='bold')
plt.xlabel('Sequence Length (Tokens)', fontsize=12)
plt.ylabel('GPU Memory Usage (MB)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)

# 4. 标注具体的数值点
for i, txt in enumerate(theoretical_kv):
    plt.annotate(f"{txt:.1f} MB", (seq_lengths[i], theoretical_kv[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='blue')
for i, txt in enumerate(pure_kv_memory):
    plt.annotate(f"{txt} MB", (seq_lengths[i], pure_kv_memory[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='green')

# 5. 保存图片
save_path = "results/kv_memory_curve.png"
plt.savefig(save_path, bbox_inches='tight')
print(f"✅ 图表已成功保存至: {save_path}")

# 6. 占比分析
print("\n--- 缓存占比分析 (128 和 2048 两个点) ---")
for i in [0, 3]:
    percent = (pure_kv_memory[i] / base_weight_vram) * 100
    print(f"序列长度 {seq_lengths[i]:4d} | 实测 KV Cache 占 FP16 权重的: {percent:.2f}%")