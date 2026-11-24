import onnxruntime_genai as og
from transformers import AutoTokenizer
import time

# 根据实际导出路径修改
model_dir = "/root/rivermind-data/models/qwen-ort-cuda"
# 如果用的是 CPU 导出，改为 "/root/rivermind-data/models/qwen-ort-cpu"

# 加载 ORT 模型
print(f"⏳ 加载 ORT 模型: {model_dir}")
model = og.Model(model_dir)
print(f"✅ 设备: {model.device_type} | 架构: {model.type}")

# 用原始 HF tokenizer 套 Chat Template（与 Day 4 保持一致）
hf_tokenizer = AutoTokenizer.from_pretrained("/root/rivermind-data/models/qwen-hf", trust_remote_code=True)
ort_tokenizer = og.Tokenizer(model)

prompt = "用一句话解释端侧推理"
messages = [{"role": "user", "content": prompt}]
chat_text = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ORT 分词
input_tokens = ort_tokenizer.encode(chat_text)

# 配置生成参数（贪心解码）
params = og.GeneratorParams(model)
params.set_search_options(max_length=len(input_tokens) + 32, do_sample=False)

generator = og.Generator(model, params)
generator.append_tokens(input_tokens)

print("\n--- 生成中 ---")
start = time.time()
new_tokens = []
while not generator.is_done():
    generator.generate_next_token()
    new_token = generator.get_next_tokens()[0]
    new_tokens.append(new_token)

end = time.time()

# 解码新增 token
output_text = ort_tokenizer.decode(new_tokens)
print(f"\n--- 模型输出 ---\n{output_text}")
print(f"\n⏱️ 生成 {len(new_tokens)} 个 token 耗时: {end - start:.2f} 秒")
print(f"🔥 速度: {len(new_tokens) / (end - start):.2f} tokens/s")