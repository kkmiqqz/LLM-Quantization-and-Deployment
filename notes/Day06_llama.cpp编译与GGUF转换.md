# Day 6: llama.cpp 编译与 GGUF 格式转换

## 一、 核心概念：为什么要离开 PyTorch？

在 Day 1 到 Day 5，我们一直在使用 HuggingFace 的 `transformers` 和 PyTorch。但如果你要把模型部署到手机、树莓派或者边缘设备上，PyTorch 庞大的体积和 Python 解释器开销就成了致命弱点。

**`llama.cpp` 就是答案。**
- 它是一个纯 C/C++ 编写的推理引擎，**零外部依赖**。
- 它底层使用 `mmap`（内存映射）技术加载模型，极大降低了内存峰值。
- 它针对不同的 CPU 指令集（如 AVX2）和 GPU 架构（CUDA/Metal）做了极致的底层汇编优化。

## 二、 核心概念：GGUF 格式到底是什么？

我们在 HuggingFace 下载的模型是“散装”的：`config.json`（配置）、`model.safetensors`（权重）、`tokenizer.json`（词表）。
**GGUF (GPT-Generated Unified Format)** 是 `llama.cpp` 独创的格式：
1. **All-in-One**：将权重、超参数、词表、甚至聊天模板全部打包进一个 `.gguf` 文件。
2. **单文件分发**：一个文件就是完整模型，极其适合端侧传输。
3. **天生为量化而生**：它的数据结构预留给各种混合精度量化（如 Q4_K_M）。

> **为什么转换成 FP16 GGUF 后，Day 8 还要进行 Q4 量化？**
> 转换为 FP16 只是将数据格式重新排列（类似压缩了“文件结构”，但内容还是 16 位浮点数，依然占 3.3GB）。只有通过 `llama-quantize`，把权重真正“切”成 4 位整数（INT4），才能实现物理体积的 4 倍缩小。

## 三、 实验过程与踩坑实录（工程经验）

这次实验可以说是“过五关斩六将”，真实的工程往往就是这样。

### 坑 1：Python 依赖地狱
- **现象**：在安装 `requirements.txt` 时报错 `triton` 缺失，且 `numpy` 版本冲突导致 `convert_hf_to_gguf.py` 崩溃（`Numpy is not available`）。
- **原因**：最新版的 `numpy (2.x)` 与 `torch 2.2.2` 和 `gguf` 包的接口不兼容。
- **解决**：强制降级 `pip install "numpy<2"`，绕过编译库的版本陷阱。

### 坑 2：源码克隆被墙
- **现象**：`git clone` 一直卡在 70% 或者报错 `HTTP/2 framing layer`。
- **解决**：取消本地代理，使用国内镜像站或 `--depth=1` 浅克隆。

### 坑 3：CUDA 编译报错（最致命）
- **现象**：编译到 14% 时，报错 `class "ncclConfig_v21400" has no member "maxCTAs"`。
- **原因**：`llama.cpp` 最新代码使用了新版 NCCL（NVIDIA 多卡通信库）的特性，而服务器上的 NCCL 版本较旧。
- **解决**：由于我们只有单卡 3090，根本用不到多卡通信。在 CMake 配置时加上 `-DGGML_CUDA_NCCL=OFF`，强行关闭 NCCL 功能，编译直接一路绿灯到 100%。

## 四、 实验结果与深度分析

### 1. 体积对比
- HuggingFace BF16 权重：约 3.1 GB
- 转换出的 FP16 GGUF：**2.9 GB**
（因为部分 LayerNorm 层变成了 FP32，所以体积并非完美的倍数关系，但完全符合预期）。

### 2. 速度对比（震撼时刻）
- **Day 1 (PyTorch 方案)**：生成 32 个 token 耗时 1.38 秒，约 **23 tokens/s**。
- **Day 7 (llama.cpp + CUDA 方案)**：生成速度达到 **123.2 tokens/s**。
- **结论**：**整整快了 5 倍多！** 这证明了 Python 的解释器开销和内存管理在推理场景中极为致命。纯 C++ 引擎配合底层 CUDA kernel，才能榨干显卡的性能。

## 五、 重点问题

**Q1：GGUF 格式和 HuggingFace 的 safetensors 有什么区别？**
> “safetensors 只是纯粹的权重矩阵，需要额外读取 json 配置；而 GGUF 是端侧推理引擎专用的单文件格式，它把权重、元数据和词表全部打包，文件头自带元信息，配合 `mmap` 内存映射，能做到秒级加载，对边缘设备极其友好。”

**Q2：为什么纯 C++ 写的 llama.cpp 推理比 Python 的 PyTorch 快那么多？**
> “第一，消除了 Python 解释器的动态开销和 GIL 锁；第二，llama.cpp 底层针对不同硬件架构写了极致的 SIMD 汇编指令和 CUDA Kernel；第三，它使用了内存映射（mmap）直接按需读取硬盘，极大减少了内存拷贝的损耗。”

**Q3：如果你要把这个模型部署到手机上，下一步该怎么做？**
> “首先，我会使用 `llama-quantize` 工具，把 FP16 的 GGUF 压缩成 Q4_K_M（4位混合量化）格式，把 2.9GB 的体积压缩到 1GB 以内。然后，我会使用 Android 端的 MLC-LLM 或 llama.cpp 的 Android 编译版本，跑通端侧推理链路。”

---
> **代码入口**：
> - 模型转换脚本：[convert_hf_to_gguf.py](../llama.cpp/convert_hf_to_gguf.py)
> - 推理测试命令见 [results/day06_gguf_test.txt](../results/day06_gguf_test.txt)