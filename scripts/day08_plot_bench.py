import json
import matplotlib.pyplot as plt
import numpy as np

# 读取刚才 llama-bench 生成的数据
with open("results/3090_bench.json", "r") as f:
    data = json.load(f)

# 整理数据
models = []
prefill_128 = []
prefill_512 = []
decode_64 = []

for item in data:
    # 提取模型名，并简写
    model_name = item["model_filename"]
    if "f16" in model_name: name = "FP16"
    elif "q8" in model_name: name = "Q8_0"
    elif "q4" in model_name: name = "Q4_K_M"
    
    # n_prompt 是预填充长度，n_gen 是生成长度
    if item["n_prompt"] == 128 and name not in models:
        models.append(name)
        prefill_128.append(item["avg_ts"])
    elif item["n_prompt"] == 512:
        prefill_512.append(item["avg_ts"])
    elif item["n_gen"] == 64:
        decode_64.append(item["avg_ts"])

# 确保顺序是 FP16, Q8, Q4（按照预填充128的顺序排）
# 如果数据乱序，这里做个简单的映射对齐
# (llama-bench 默认按传入模型顺序输出，所以直接画即可)

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# 画三组柱子
rects1 = ax.bar(x - width, prefill_128, width, label='Prefill (128 tokens)', color='#4C72B0')
rects2 = ax.bar(x, prefill_512, width, label='Prefill (512 tokens)', color='#DD8452')
rects3 = ax.bar(x + width, decode_64, width, label='Decode (64 tokens)', color='#55A868')

# 添加文字标签
ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
ax.set_title('Qwen2.5-1.5B: Performance Benchmark on RTX 3090', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=11)

# 在柱子上标数字
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.grid(axis='y', linestyle='--', alpha=0.7)
fig.tight_layout()

# 保存图片
plt.savefig("results/bench_3090_bar.png")
print("✅ 跑分图表已保存至: results/bench_3090_bar.png")