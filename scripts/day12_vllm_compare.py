import time
import subprocess
import os
from vllm import LLM, SamplingParams

# 定义一个后台采样显存的函数（用 nvidia-smi）
def get_gpu_memory():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    return int(result.strip().split('\n')[0])

def benchmark_model(model_path, is_awq=False):
    print(f"\n==================================================")
    print(f"⏳ 正在加载模型: {model_path}")
    
    # 强制使用 eager 模式，这样显存测量更准，且避免 CUDA Graph 的额外显存开销
    llm = LLM(
        model=model_path, 
        quantization="awq" if is_awq else None,
        enforce_eager=True, 
        gpu_memory_utilization=0.8,
        max_model_len=512 # 限制上下文长度
    )
    
    # 预热一次
    prompt = "请写一段快速排序的 Python 代码。"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    llm.generate([prompt], sampling_params)
    
    # 记录运行时的显存
    time.sleep(1) # 等待显存稳定
    mem_used = get_gpu_memory()
    
    # 正式测速
    print(f"🚀 开始测速 (生成 32 tokens)...")
    start_time = time.time()
    outputs = llm.generate([prompt] * 4, sampling_params) # 稍微加点并发测得更准
    end_time = time.time()
    
    total_tokens = 32 * 4
    time_cost = end_time - start_time
    speed = total_tokens / time_cost
    
    print(f"✅ {model_path} 测试结果:")
    print(f"   - 运行显存占用: {mem_used} MiB")
    print(f"   - 生成速度: {speed:.2f} tokens/s")
    
    # 释放显存
    del llm
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    time.sleep(2)
    
    return mem_used, speed

if __name__ == "__main__":
    # 需要确保你的 models/qwen-hf 是原来的 FP16 模型
    fp16_mem, fp16_speed = benchmark_model("/root/rivermind-data/models/qwen-hf", is_awq=False)
    awq_mem, awq_speed = benchmark_model("/root/rivermind-data/models/qwen-awq", is_awq=True)
    print("\n================ 最终对比结果 ================")
    print(f"FP16: 显存 {fp16_mem} MiB | 速度 {fp16_speed:.2f} t/s")
    print(f"AWQ : 显存 {awq_mem} MiB | 速度 {awq_speed:.2f} t/s")
    print(f"显存节省: {((fp16_mem - awq_mem) / fp16_mem) * 100:.1f}%")