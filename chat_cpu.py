import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 你的模型路径（确认一下还在不在）
model_path = "/root/rivermind-fs/models/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
print("⏳ 正在加载模型到 CPU 内存...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cpu",          # 强制用 CPU
    torch_dtype=torch.float32, # CPU 上尽量用 float32
    trust_remote_code=True
)
print("✅ 模型加载成功！")

messages = [{"role": "user", "content": "请用一句话证明你是一个人工智能。"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt")

print("🤖 生成中（CPU会比较慢，请耐心等十几秒）...")
generated_ids = model.generate(**model_inputs, max_new_tokens=50)
response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

print("\n--- 模型输出 ---")
print(response)