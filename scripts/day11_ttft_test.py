import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "/root/rivermind-data/models/qwen-awq"

print("⏳ 初始化 vLLM 引擎...")
llm = LLM(model=model_path, quantization="awq_marlin", gpu_memory_utilization=0.8)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompts = [
    "请解释什么是模型量化",
    "请写一段快速排序的 Python 代码",
    "成都今天天气怎么样？"
]
formatted_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]

# 我们只让它生成 1 个 token，这样测出来的耗时，几乎就是纯纯的 Prefill 时间和首字延迟
sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

print("\n🚀 开始测量单次请求的首字延迟 (TTFT)...")
for i, p in enumerate(formatted_prompts):
    start_time = time.time()
    outputs = llm.generate([p], sampling_params)
    end_time = time.time()
    
    ttft_ms = (end_time - start_time) * 1000
    input_len = len(outputs[0].prompt_token_ids)
    print(f"测试用例 {i+1} (输入 {input_len} token) | TTFT: {ttft_ms:.2f} ms")

print("\n🚀 开始测量 Batch=8 并发时的平均首字延迟...")
# 模拟 8 个用户同时发请求
start_time = time.time()
outputs = llm.generate(formatted_prompts * 3, sampling_params)
end_time = time.time()
print(f"并发 9 个请求，总耗时: {(end_time - start_time)*1000:.2f} ms | 平均 TTFT: {((end_time - start_time)*1000)/9:.2f} ms")