import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/qwen-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)

# 记录模型加载完成后的基础显存
torch.cuda.empty_cache()
torch.cuda.synchronize()
base_vram = torch.cuda.memory_allocated() / (1024**2) # MB
print(f"✅ 模型权重加载完毕，纯权重显存占用: {base_vram:.2f} MB")

# 准备一段超长文本用于测试 (重复一段话凑长度)
long_text = "端侧推理是指在设备本地进行模型计算。" * 200
inputs = tokenizer(long_text, return_tensors="pt").to("cuda")

seq_lengths = [128, 512, 1024, 2048]
results = []

for seq_len in seq_lengths:
    # 截断到指定长度
    input_ids = inputs.input_ids[:, :seq_len]
    
    # 清理显存，重置峰值统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # 只做一次前向传播 (Prefill阶段)，生成 KV Cache
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
    
    torch.cuda.synchronize()
    
    # 计算显存增量
    current_vram = torch.cuda.memory_allocated() / (1024**2)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    
    kv_cache_vram = current_vram - base_vram
    peak_delta = peak_vram - base_vram
    
    print(f"序列长度 {seq_len:4d} | 缓存占用: {kv_cache_vram:6.2f} MB | 前向峰值: {peak_delta:6.2f} MB")
    
    # 清理中间变量和 KV Cache，防止影响下一次循环
    del outputs
    torch.cuda.empty_cache()
