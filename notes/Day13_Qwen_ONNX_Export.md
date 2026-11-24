# Day 13 Qwen 导出 ONNX 与 ORT GenAI 环境适配

**目标**　将本地 Qwen 导出为 FP16 CUDA ONNX 模型，理解计算图、外部权重、KV 输入输出和 Execution Provider。

## 一、核心理论前置

### 1.1 ONNX 计算图是什么

ONNX 全称为 **Open Neural Network Exchange**。它把模型计算表示为由算子节点和张量连接构成的图，并保存输入输出、数据类型、形状及参数信息。

例如，Qwen 的矩阵乘法、归一化、Attention 等计算可以通过 ONNX 图表达，再由 ONNX Runtime 加载执行。导出工具可能融合算子或使用运行时专用算子，部署时需要匹配相应的运行环境。

与 Day 12 的知识衔接起来，**ONNX 图负责表达模型计算，CUDA Graph 负责捕获和重放 GPU 执行流程**。

### 1.2 外部权重为什么要单独保存

本次导出采用 **External Data，外部张量数据存储**。

- `model.onnx` 保存图结构、张量描述及外部权重引用。
- `model.onnx.data` 保存大量参数数值。

图中的引用记录了权重所在文件及数据位置。这种方式支持大体积模型的保存与加载，分发时需要保持两个文件的相对位置和名称一致。

本次图文件只有 **167K**，主要参数保存在 **2.9G** 的外部文件中，因此模型体积需要结合两者计算。

### 1.3 KV Cache 为什么是图的输入输出

自回归生成需要反复调用模型。图接收历史 **past KV**，计算当前 token，并输出更新后的 **present KV**，供下一次调用复用。

| 阶段 | 当前输入 | 缓存变化 |
| --- | --- | --- |
| Prefill | 通常为完整 Prompt | 建立 Prompt 对应的 KV Cache |
| Decode | 通常为新 token 和历史 KV | 复用历史 K、V，加入当前 token 的 K、V |

例如，已有 128 个 token 的缓存，再处理 1 个 token，有效缓存长度就增长到 129。历史 K、V 得以复用，Attention 仍需读取相关历史状态。

KV 的常见布局为 `[batch_size, num_kv_heads, sequence_length, head_dim]`，具体布局和接口名称以本次导出的图为准。检查时可以关注 `past_key_values`、`present` 等常见命名。

### 1.4 Execution Provider 和生成配置

**Execution Provider，EP** 是 ONNX Runtime 对接硬件执行库的接口。运行时把受支持的节点或子图交给相应后端执行。

| 配置或组件 | 作用 |
| --- | --- |
| `-e cuda` | 导出时选择 CUDA 目标 |
| `CUDAExecutionProvider` | 运行时通过 CUDA 执行受支持的计算 |
| `genai_config.json` | 描述模型文件、结构参数、输入输出映射、生成及会话配置 |
| 分词器文件 | 完成文本与 token ID 的相互转换 |

ORT GenAI 在模型调用之外，还负责组织 token 选择、停止条件和 KV 状态管理。本次完成了面向 CUDA 的导出，实际 GPU 生成需要在后续运行中验证。

---

## 二、实验环境与步骤

### 2.1 关键版本

本次使用云端 Linux 实例已有的 `/opt/conda` Python 3.10 环境，工作目录为 `/root/rivermind-fs`。

| 组件 | 本次记录的版本 |
| --- | --- |
| `onnxruntime-genai-cuda` | 0.11.4 |
| `onnxruntime-gpu` | 1.23.2 |
| `onnx` / `onnx-ir` | 1.23.2 / 1.1.0 |
| `numpy` / `ml_dtypes` | 1.26.4 / 0.5.4 |
| `gguf` | 0.18.0 |
| `transformers` | 修复指令指定为 4.51.3，归档时核对 |

依赖修复后，`pip check` 实际输出为 `No broken requirements found.`。

### 2.2 安装与检查入口

最初通过清华镜像安装 CUDA 版本。

```bash
python -m pip install onnxruntime-genai-cuda \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

主要下载了 90.0 MB 的 GenAI 包和 300.5 MB 的 GPU Runtime 包。主包导入检查输出为 `0.11.4`。

```bash
python -c "import onnxruntime_genai as og; print(og.__version__)"
```

转换前可使用 `--help` 检查参数及 builder 的模块加载情况。

```bash
python -m onnxruntime_genai.models.builder --help
```

### 2.3 导出 FP16 CUDA 模型

本次成功执行的命令如下，输入输出使用绝对路径。

```bash
python -m onnxruntime_genai.models.builder \
  -i /root/rivermind-data/models/qwen-hf \
  -o /root/rivermind-data/models/qwen-ort-cuda \
  -p fp16 -e cuda
```

| 参数 | 含义 |
| --- | --- |
| `-i` | 本地 Hugging Face 模型目录 |
| `-o` | 导出产物目录 |
| `-p fp16` | 指定 FP16 精度 |
| `-e cuda` | 指定 CUDA 执行目标 |

终端出现以下完成日志，随后正常返回提示符。

```text
Saving ONNX model in /root/rivermind-data/models/qwen-ort-cuda
Saving GenAI config in /root/rivermind-data/models/qwen-ort-cuda
Saving processing files in /root/rivermind-data/models/qwen-ort-cuda for GenAI
```

这份 FP16 导出模型可作为后续 ORT 推理与量化实验的基线。

---

## 三、实验结果与文件说明

通过以下命令检查产物。

```bash
ls -lh /root/rivermind-data/models/qwen-ort-cuda/
```

| 文件 | 显示大小 | 作用 |
| --- | --- | --- |
| `model.onnx` | 167K | ONNX 图及外部权重引用 |
| `model.onnx.data` | 2.9G | 外部权重数据 |
| `genai_config.json` | 1.8K | 模型与生成配置 |
| `tokenizer.json` | 11M | 分词器序列化数据 |
| `tokenizer_config.json` | 7.2K | 分词器行为及相关配置 |
| `vocab.json` | 2.7M | 词表映射 |
| `merges.txt` | 1.6M | BPE 合并规则 |
| `special_tokens_map.json` | 613 字节 | 特殊 token 的角色映射 |
| `added_tokens.json` | 605 字节 | 附加 token 信息 |

目录总量约 **2.9G**，沿用 `ls -lh` 的显示值。后续比较不同模型体积时，需要统一容量单位。

目前已验证主包及 ONNX 相关模块能够导入、声明依赖检查通过、builder 完成导出，并生成了上述文件。实际文本生成和 GPU 执行情况仍待验证。

---

## 四、踩坑实录

| 问题 | 原因或现象 | 本次处理 |
| --- | --- | --- |
| 大包下载缓慢 | 清华源约 27.7 kB/s，官方 CDN 测速约 27.6 kB/s | 当时已下载约 84%，继续原下载，最终完成 |
| 缺少 `onnx_ir` | 主包能导入，builder 还需要额外的转换依赖 | 补装 `onnx-ir` |
| NumPy 冲突及安装记录损坏 | 实际导入 2.2.6，已有 vLLM、outlines 要求 NumPy 小于 2，旧版 `METADATA` 又缺失 | 先下载指定 wheel，备份 NumPy 文件及安装记录，再重装 NumPy 1.26.4、`ml_dtypes` 0.5.4 |
| `gguf` 版本冲突 | 0.19.0 在当前 Python 下要求 NumPy 至少为 2.2.6 | 将 `gguf` 调整为 0.18.0 |
| Transformers 接口缺失 | 4.45.0 缺少 builder 要导入的 `Qwen2_5_VLForConditionalGeneration` | 修复指令指定 Transformers 4.51.3，随后完成导出 |

补装 `onnx-ir` 时，最初没有约束关联依赖，触发了 NumPy 等组件的连带升级。这次实践需要记住两点。

1. **已有环境中安装新组件，要同时考虑版本约束。** 本次围绕 `numpy==1.26.4`、`ml_dtypes==0.5.4`、`gguf==0.18.0` 修复兼容性。
2. **分层验证环境。** `pip check` 核对声明依赖，模块导入检查验证接口，实际转换验证 builder 的执行路径，GPU 推理再验证运行过程。

---

## 五、当天验收与记录归档

| 验收项 | 当前状态 |
| --- | --- |
| 本地 Qwen 导出为 FP16 CUDA ONNX | 已完成 |
| 图、外部权重、生成配置和分词器文件 | 已生成并检查存在 |
| NumPy 修复及声明依赖检查 | 已完成 |
| 图接口、配置内容和模型结构检查 | 待补充 |
| 完整构建日志、最终版本及硬件资源信息 | 待归档 |
| 独立环境建设 | 待补充，当前使用已有 Conda 环境 |
| CUDA 模型加载与文本生成 | 待验证 |

导出完成后，可用以下命令查看配置并进行模型结构检查，执行结果再补入记录。

```bash
python -m json.tool \
  /root/rivermind-data/models/qwen-ort-cuda/genai_config.json

python -c "import onnx; onnx.checker.check_model('/root/rivermind-data/models/qwen-ort-cuda/model.onnx'); print('ONNX 结构检查通过')"
```

重点核对模型文件名、输入输出映射、KV 相关字段及执行后端。对于超过 2GB 的外部数据模型，按 ONNX 官方建议使用文件路径进行 checker 检查。

版本与资源记录可保存到项目的 `logs` 目录。

```bash
mkdir -p logs
python -m pip freeze > logs/day13_requirements.txt
python -m pip check > logs/day13_pip_check.txt 2>&1
nvidia-smi > logs/day13_nvidia_smi.txt 2>&1
free -h > logs/day13_host_ram.txt
df -h /root/rivermind-data /opt/conda > logs/day13_disk.txt
```

CUDA、cuDNN 版本及库加载来源需要结合实际环境补充。本笔记保留了关键构建片段，完整构建日志可从现有终端记录保存。

---

> **模型入口**
>
> - 本地 HF 权重目录为 `/root/rivermind-data/models/qwen-hf`。
> - FP16 CUDA 导出目录为 `/root/rivermind-data/models/qwen-ort-cuda`。
> - 后续先验证文本生成，再基于同一 ORT 部署路径进行量化对照。

## 参考资料

- [Microsoft Model Builder 官方入口](https://onnxruntime.ai/docs/genai/howto/build-model.html)
- [ONNX 计算图基础](https://onnx.ai/onnx/intro/concepts.html)
- [ONNX 外部张量数据与检查方法](https://onnx.ai/onnx/repo-docs/ExternalData.html)
- [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [ORT GenAI 配置说明](https://onnxruntime.ai/docs/genai/reference/config.html)
