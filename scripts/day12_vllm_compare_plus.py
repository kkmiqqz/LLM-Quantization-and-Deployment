import time
import subprocess
from vllm import LLM, SamplingParams
import gc
import torch

def get_gpu_memory():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    return int(result.strip().split('\n')[0])

def benchmark_model(model_path, quant_method=None):
    print(f"\n==================================================")
    print(f"⏳ 正在加载模型: {model_path}")
    print(f"配置: quantization={quant_method}, enforce_eager=False (开启CUDA Graph)")
    
    # 核心修改：
    # 1. 如果是 awq，使用 awq_marlin 内核
    # 2. enforce_eager=False，让 vLLM 使用 CUDA Graph 加速
    # 3. gpu_memory_utilization=0.4，防止吃满显存，方便看清纯权重差距
    llm = LLM(
        model=model_path, 
        quantization=quant_method,
        enforce_eager=False, 
        gpu_memory_utilization=0.4,
        max_model_len=512
    )
    
    prompt = "请写一段快速排序的 Python 代码。"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    
    # 预热
    llm.generate([prompt], sampling_params)
    
    time.sleep(1)
    mem_used = get_gpu_memory()
    
    print(f"🚀 开始极限测速 (生成 32 tokens)...")
    start_time = time.time()
    outputs = llm.generate([prompt] * 4, sampling_params) 
    end_time = time.time()
    
    total_tokens = 32 * 4
    speed = total_tokens / (end_time - start_time)
    
    print(f"✅ 测试结果:")
    print(f"   - 运行显存占用: {mem_used} MiB")
    print(f"   - 生成速度: {speed:.2f} tokens/s")
    
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    
    return mem_used, speed

if __name__ == "__main__":
    # 测 FP16
    fp16_mem, fp16_speed = benchmark_model("/root/rivermind-data/models/qwen-hf", quant_method=None)
    # 测 AWQ Marlin
    awq_mem, awq_speed = benchmark_model("/root/rivermind-data/models/qwen-awq", quant_method="awq_marlin")
    
    print("\n================ 最终对比结果 (开启优化内核) ================")
    print(f"FP16      : 显存 {fp16_mem} MiB | 速度 {fp16_speed:.2f} t/s")
    print(f"AWQ_Marlin: 显存 {awq_mem} MiB | 速度 {awq_speed:.2f} t/s")