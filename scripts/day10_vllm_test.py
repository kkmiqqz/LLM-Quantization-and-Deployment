import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "models/qwen-awq"

print("⏳ vLLM 正在加载 AWQ 模型并初始化引擎...")
# 尝试开启 awq_marlin 加速内核（如果报错，改回 "awq" 即可）
llm = LLM(model=model_path, quantization="awq_marlin", gpu_memory_utilization=0.8)

# 加载 Tokenizer 来套用 Chat 模板
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 准备 16 条不同的 Prompt 用于并发测试
prompts = ["用一句话解释端侧推理"] * 16
formatted_prompts = []
for p in prompts:
    messages = [{"role": "user", "content": p}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    formatted_prompts.append(text)

sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

print("🚀 开始单条延迟测试...")
start_time = time.time()
outputs = llm.generate([formatted_prompts[0]], sampling_params)
end_time = time.time()

print("\n--- 单条模型输出 ---")
print(outputs[0].outputs[0].text)
single_time = end_time - start_time
print(f"⏱️ 单条生成 64 个 token 耗时: {single_time:.2f} 秒")
print(f"🔥 单条速度: {64 / single_time:.2f} tokens/s")


print("\n🚀 开始 Batch=16 并发吞吐量极限压测...")
start_time = time.time()
outputs = llm.generate(formatted_prompts, sampling_params)
end_time = time.time()

total_time = end_time - start_time
total_tokens = 64 * 16 # 16条请求，每条64个token

print(f"\n⏱️ 并发生成 16 * 64 = {total_tokens} 个 token 总耗时: {total_time:.2f} 秒")
print(f"🔥 vLLM 并发总吞吐量: {total_tokens / total_time:.2f} tokens/s")