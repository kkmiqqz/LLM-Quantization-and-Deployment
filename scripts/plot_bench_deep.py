import json
import matplotlib.pyplot as plt

# 读取数据
with open("results/3090_bench_deep.json", "r") as f:
    data = json.load(f)

# 提取 Prefill 数据（只取 n_prompt > 0 的）
models = ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
prompt_lengths = [128, 256, 512, 1024, 2048]

# 初始化字典存储各模型的数据
prefill_data = {m: [] for m in models}

for item in data:
    if item["n_prompt"] > 0:
        model_name = item["model_filename"]
        if "f16" in model_name: name = "FP16"
        elif "q8" in model_name: name = "Q8_0"
        elif "q6" in model_name: name = "Q6_K"
        elif "q5" in model_name: name = "Q5_K_M"
        elif "q4" in model_name: name = "Q4_K_M"
        
        prefill_data[name].append(item["avg_ts"])

# 开始画折线图
plt.figure(figsize=(10, 6), dpi=150)

# 配置颜色和标记，让图表极其专业
colors = {"FP16": "black", "Q8_0": "blue", "Q6_K": "green", "Q5_K_M": "orange", "Q4_K_M": "red"}
markers = {"FP16": "o", "Q8_0": "s", "Q6_K": "^", "Q5_K_M": "v", "Q4_K_M": "D"}

for m in models:
    plt.plot(prompt_lengths, prefill_data[m], marker=markers[m], 
             color=colors[m], linewidth=2, label=f"{m} (Prefill)")

plt.title('Prefill Throughput vs Prompt Length (Qwen2.5-1.5B on RTX 3090)', fontsize=14, fontweight='bold')
plt.xlabel('Prompt Length (Tokens)', fontsize=12)
plt.ylabel('Prefill Throughput (tokens/sec)', fontsize=12)
plt.xticks(prompt_lengths) # 强制显示所有刻度
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)

# 保存
plt.savefig("results/bench_prefill_curve.png", bbox_inches='tight')
print("✅ 折线图已保存至: results/bench_prefill_curve.png")