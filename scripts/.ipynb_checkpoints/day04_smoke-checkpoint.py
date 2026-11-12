import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 确保 results 目录存在
os.makedirs("results", exist_ok=True)

model_path = "models/qwen-hf"
print("⏳ 加载模型 (只需加载一次，循环跑三个提示)...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)

# 读取固定测试集
with open("data/prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

results = []

for prompt in prompts:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    input_len = model_inputs.input_ids.shape[1]
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=32, do_sample=False)
        
    new_tokens = generated_ids[0][input_len:]
    output_len = len(new_tokens)
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    res_data = {
        "prompt": prompt,
        "input_len": input_len,
        "output_len": output_len,
        "output": response
    }
    results.append(res_data)
    print(f"✅ 完成 Prompt: {prompt}")

# 写入 jsonl 文件 (每行一个 JSON)
with open("results/hf_smoke.jsonl", "w", encoding="utf-8") as f:
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print("\n✅ 验收物已生成: results/hf_smoke.jsonl")