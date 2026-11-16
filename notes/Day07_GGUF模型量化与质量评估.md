# Day 7: 生成并验证 Q8_0 与 Q4_K_M 模型

## 一、 实验目的
在 Day 6 中，我们得到了一个 2.9GB 的 FP16 GGUF 模型。对于端侧设备（如手机、车机），这个体积依然过大。今天的任务是使用 `llama-quantize` 工具，将其压缩为 Q8_0 和 Q4_K_M 格式，并验证在 GPU 环境下，压缩后的模型体积、推理速度和输出质量的变化。

## 二、 实验步骤与执行指令

### 1. 模型量化
使用纯 CPU 版编译的量化工具（避开损坏的 CUDA 库），以 FP16 模型为“母本”进行量化：
```bash
# 1. 生成 8-bit 量化模型
./llama.cpp/build_cpu/bin/llama-quantize models/qwen-f16.gguf models/qwen-q8.gguf Q8_0

# 2. 生成 4-bit 混合量化模型
./llama.cpp/build_cpu/bin/llama-quantize models/qwen-f16.gguf models/qwen-q4.gguf Q4_K_M
```
*(注：量化 Q8 耗时约 42 秒，Q4_K_M 耗时约 284 秒。Q4 混合量化需要更复杂的计算。)*

### 2. GPU 推理测速验证
使用编译好的 GPU 版 `llama-cli`，强制将所有层 offload 到显卡（`-ngl 99`），测试统一 Prompt 的生成速度：
```bash
# 测试 Q8_0 速度
./llama.cpp/build/bin/llama-cli -m models/qwen-q8.gguf -p "用一句话解释端侧推理" -n 32 -ngl 99

# 测试 Q4_K_M 速度
./llama.cpp/build/bin/llama-cli -m models/qwen-q4.gguf -p "用一句话解释端侧推理" -n 32 -ngl 99
```

## 三、 实验结果与深度分析

### 1. 量化数据双速率对比
| 模型版本 | 文件大小 | 平均位宽 (BPW) | Prefill 速度 (Prompt) | Decode 速度 (Generation) |
| :--- | :--- | :--- | :--- | :--- |
| **FP16 (原版)** | 2.9 GB | 16.00 | 475.5 t/s | 123.2 t/s |
| **Q8_0** | 1.5 GB | 8.50 | **1716.4 t/s** | **199.6 t/s** |
| **Q4_K_M** | **0.9 GB** | **5.08** | 1009.3 t/s | 185.7 t/s |

### 2. 深度解析：Prefill 与 Decode 的不同瓶颈
- **Prefill (Prompt) 是算力密集型**：从 FP16 到 Q8_0，Prompt 速度从 475 暴涨到 1716 t/s。因为低精度整数矩阵乘法极大提升了 GPU 的计算吞吐。
- **Decode (Generation) 是访存密集型**：从 FP16 到 Q4_K_M，生成速度从 123 提升到 185 t/s。因为模型体积缩小，每生成一个字搬运数据的带宽压力锐减。

### 3. 反常识现象背后的工程博弈
**现象**：为什么 Q8_0 的生成速度（199.6）比 Q4_K_M（185.7）快？为什么 Q8_0 的预填充速度（1716）远超 Q4_K_M（1009）？
**解答**：Q4_K_M 使用了复杂的混合量化块（K-quants），在计算时必须先将 4-bit 权重反量化还原为 16-bit。这种“反量化税”增加了计算开销。在 Prefill 这种纯算力场景下，Q4 多花的反量化时间抵消了其低带宽的优势。这说明：**量化不是越低越好，而是需要在“带宽红利”和“反量化开销”之间寻找平衡。**

## 四、 踩坑实录：libcuda.so.1 崩溃与修复
- **现象**：量化时报错 `error while loading shared libraries: /lib/x86_64-linux-gnu/libcuda.so.1: file too short`，甚至 `nvidia-smi` 命令也失效。
- **原因**：系统盘曾爆满，导致云容器环境下的宿主机驱动映射文件损坏。
- **解决**：在容器内无法直接修复映射文件。通过网页控制台执行“彻底关机 -> 重新开机”，云平台重新挂载了完好的驱动文件，GPU 瞬间复活。

---
> **代码与结果入口**：
> - 测速结果数据：[results/day07_quant_benchmark.txt](../results/day07_quant_benchmark.txt)
> - 生成模型：[models/qwen-q8.gguf](../models/) | [models/qwen-q4.gguf](../models/)
