import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "models/qwen-hf"

print("⏳ 正在加载 Tokenizer 和 Model 到 GPU...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.float16, # Day1 明确要求用 FP16
    trust_remote_code=True
)
print(f"✅ 模型加载成功！当前使用的设备: {model.device}")

# 输入 prompt
prompt = "用一句话解释什么是端侧推理"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("\n--- 验证数据 ---")
print(f"input_ids 形状: {model_inputs.input_ids.shape}")
print(f"前 10 个 token ID: {model_inputs.input_ids[0][:10].tolist()}")
print(f"模型数据类型 (dtype): {model.dtype}")

# 生成
start = time.time()
generated_ids = model.generate(**model_inputs, max_new_tokens=32, do_sample=False)
end = time.time()

# 提取新增的 token
new_tokens = generated_ids[0][model_inputs.input_ids.shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"\n--- 模型输出 ---\n{response}")
print(f"\n⏱️ 生成 32 个 token 耗时: {end - start:.2f} 秒")

# Day1 任务 4：张量字节计算验证
print("\n--- 张量字节计算 ---")
tensor_2x3 = torch.randn(2, 3, dtype=torch.float16)
bytes_manual = 2 * 3 * 2 # 形状 2x3，FP16占2字节
bytes_calc = tensor_2x3.numel() * tensor_2x3.element_size()
print(f"手算字节数: {bytes_manual}, 代码计算字节数: {bytes_calc}")