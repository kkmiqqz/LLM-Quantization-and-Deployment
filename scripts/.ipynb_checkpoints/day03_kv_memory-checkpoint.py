import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/qwen-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

torch.cuda.empty_cache()
torch.cuda.synchronize()
base_vram = torch.cuda.memory_allocated() / (1024**2)
print(f"✅ 纯权重显存: {base_vram:.2f} MB")

long_text = "端侧推理是指在设备本地进行模型计算。" * 200
inputs = tokenizer(long_text, return_tensors="pt").to("cuda")

seq_lengths = [128, 512, 1024, 2048]

for seq_len in seq_lengths:
    input_ids = inputs.input_ids[:, :seq_len]
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
    
    # 🔥 关键动作：提取 KV Cache，并彻底删除其他所有东西
    past_key_values = outputs.past_key_values
    del outputs  # 删掉庞大的 logits
    del input_ids
    torch.cuda.empty_cache() # 强制清理缓存
    torch.cuda.synchronize()
    
    # 此时显存里只剩下模型权重和纯正的 KV Cache
    cache_vram = torch.cuda.memory_allocated() / (1024**2) - base_vram
    
    print(f"序列长度 {seq_len:4d} | 纯 KV Cache 占用: {cache_vram:6.2f} MB")
    
    del past_key_values
    torch.cuda.empty_cache()