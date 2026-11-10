import matplotlib.pyplot as plt
import os

# 确保 results 文件夹存在
os.makedirs("results", exist_ok=True)

# 1. 准备数据 (来源于你刚才的实测数据)
seq_lengths = [128, 512, 1024, 2048]
pure_kv_memory = [11.66, 22.16, 36.16, 64.16]    # 纯 KV Cache 占用
peak_memory = [124.22, 469.03, 930.91, 1852.16]  # 前向传播峰值总增量 (含临时张量)

# 2. 设置绘图参数
plt.figure(figsize=(10, 6), dpi=150) # 设置画布大小和清晰度
# 为了兼容服务器上的中文字体，这里用英文作为标题和图例，如果你本地有中文字体可以换成中文
plt.plot(seq_lengths, pure_kv_memory, marker='o', linewidth=2, color='green', label='Pure KV Cache Memory')
plt.plot(seq_lengths, peak_memory, marker='s', linewidth=2, color='red', linestyle='--', label='Forward Peak Memory (Includes Temp Tensors)')

# 3. 美化图表
plt.title('Qwen2.5-1.5B: Memory Usage vs Input Sequence Length (FP16)', fontsize=14, fontweight='bold')
plt.xlabel('Sequence Length (Tokens)', fontsize=12)
plt.ylabel('GPU Memory Usage (MB)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)

# 4. 标注具体的数值点
for i, txt in enumerate(pure_kv_memory):
    plt.annotate(f"{txt} MB", (seq_lengths[i], pure_kv_memory[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='green')
for i, txt in enumerate(peak_memory):
    plt.annotate(f"{txt} MB", (seq_lengths[i], peak_memory[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=10, color='red')

# 5. 保存图片
save_path = "results/kv_memory_curve.png"
plt.savefig(save_path, bbox_inches='tight')
print(f"✅ 图表已成功保存至: {save_path}")