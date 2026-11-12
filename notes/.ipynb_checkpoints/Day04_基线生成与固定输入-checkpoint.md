# Day 4: 固定测试集与基线生成

## 🧠 核心理论：科学实验的对照原则
在做量化对比之前，必须建立**标准的 Baseline（基线）**。
- 采用 `do_sample=False`（贪心解码），确保每次生成的绝对一致性。
- 截取新生成的 token：使用 `new_tokens = generated_ids[0][input_len:]`。
- 如果生成了终止符（EOS），模型会自然停止，长度可能小于 `max_new_tokens`。

## 🔬 实验记录与分析
- 我们固定了三个测试 Prompt（解释边缘计算、改写设备告警、区分内存与磁盘）。
- 生成了 JSONL 格式的基准文件，记录了输入长度、输出长度和具体的输出文本。这正是日后评判 GGUF、AWQ 模型“有没有变傻”的黄金标准。

> **代码入口**：[hf_generate.py](../scripts/hf_generate.py) 和 [day04_smoke.py](../scripts/day04_smoke.py)