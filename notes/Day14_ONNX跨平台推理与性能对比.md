# Day 14: ONNX Runtime 跨平台推理与性能深度对比

## 一、 实验目的
在 Day 13 中，我们将 HuggingFace 的 PyTorch 权重成功导出为了 ONNX 格式（包含静态计算图和外部权重）。今天的任务是：
1. 使用轻量级的 `onnxruntime-genai` 加载刚刚导出的模型，跑通自回归文本生成。
2. 在同一个 Prompt、相同精度（FP16）下，与原生 PyTorch (`model.generate`) 进行严格的性能对比，探究推理引擎底层的性能差异。

## 二、 实验步骤与指令

### 1. ORT GenAI 推理测试
新建 `scripts/day14_ort_generate.py`，核心 API 流程为：`og.Model` -> `og.Tokenizer` -> `og.GeneratorParams` -> `append_tokens` -> `generate_next_token`。
```bash
python3 scripts/day14_ort_generate.py
```

### 2. PyTorch 基线测速（控制变量）
使用 Day 4 编写并加入计时功能的原生生成脚本，读取相同的数据盘绝对路径模型，保证输入完全一致。
```bash
python3 scripts/hf_generate.py --model_dir /root/rivermind-data/models/qwen-hf --prompt "用一句话解释端侧推理" --max_new_tokens 32
```

---

## 三、 实验结果：震撼的 7.5 倍性能鸿沟

### 1. 生成质量对比
- **Prompt**："用一句话解释端侧推理"
- **PyTorch 输出**："端侧推理是指在设备本地进行的**模型推理**，无需将数据传输到云端。"
- **ORT 输出**："端侧推理是指在设备本地进行的**推理任务**，无需将数据传输到云端。"
*(结论：语义完全一致，仅存在微小的浮点精度差异，证明 ONNX 导出非常成功。)*

### 2. 性能与开销对比
| 推理后端 | 硬件设备 | 实际生成耗时 | 生成速度 | 环境依赖体量 |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch (原生)** | CUDA (FP16) | 1.29 秒 | **15.51 t/s** | 数 GB 的 PyTorch 框架环境 |
| **ONNX Runtime** | CUDA (FP16) | 0.17 秒 | **116.17 t/s** | 仅 90MB 的轻量级运行库 |

*(注：数据已同步保存至 `results/day14_ort_compare.csv`)*

---

## 四、 深度原理剖析：为什么原生 PyTorch 这么慢？

很多人会疑惑：PyTorch 明明是深度学习的老大，为什么在推理生成时被 ONNX Runtime 按在地上摩擦（15 t/s vs 116 t/s）？这背后是极其底层的工程差异：

**1. Python 解释器的“调度税”（Eager 模式劣势）**
HuggingFace 的 `model.generate()` 内部是一个巨大的 Python `while` 循环。每吐出一个字，CPU 都要执行大量的条件判断（是否遇到 EOS）、张量拼接、形状检查。GPU 算得极快，但 CPU 慢吞吞地发指令，导致 GPU 算力严重闲置。
而 ONNX Runtime 的生成循环**在底层的 C++ 中执行**，几乎消除了 CPU 调度开销，让 GPU 可以持续满负荷运转。

**2. 动态图 vs 静态图（算子融合）**
PyTorch 是动态图，每次计算都要启动几十个微小的 CUDA Kernel。
ONNX 是**静态计算图**，推理引擎在底层对算子进行了高度**融合（Operator Fusion）**。比如把 `LayerNorm`、`Add`、`MatMul` 合并成一个大的 Kernel 一次性算完，大幅减少了显存带宽的读取次数。

**3. 显存管理机制**
原生的 KV Cache 在不断追加新 Token 时，会进行频繁的动态显存申请和重新分配（导致显存碎片）。而 ORT GenAI 底层采用了预分配内存池机制，访问效率极高。

---

## 五、 全景总结：部署框架的“三足鼎立”

至此，我们完成了整个项目三条链路的探索。没有一个框架是万能的，它们各自在不同的场景称王：
1. **llama.cpp**：适合离线、无显卡、C++ 环境（极限压缩）。
2. **vLLM**：适合云端高并发、需要极限吞吐量的服务（PagedAttention）。
3. **ONNX Runtime**：适合跨平台（Windows/Mac/Linux）、要求小体积、无 Python 依赖的端侧交付。

---
> **代码与结果入口**：
> - ORT 推理脚本：[day14_ort_generate.py](../scripts/day14_ort_generate.py)
> - PyTorch 基线脚本：[hf_generate.py](../scripts/hf_generate.py)
> - 性能对比数据：[day14_ort_generate.txt](../results/day14_ort_generate.txt)