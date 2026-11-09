import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/root/rivermind-fs/models/Qwen/Qwen2.5-1.5B-Instruct"

print("⏳ 正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

messages = [{"role": "user", "content": "你好，请简单介绍一下你自己。"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=50)
response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

print("\n--- 模型输出 ---")
print(response)