from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

model_path = "models/qwen-hf" # 你之前的 HF 模型路径
quant_path = "models/qwen-awq"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

print("⏳ 正在加载 FP16 模型进行 AWQ 量化...")
# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 执行量化
print("🚀 开始量化...")
model.quantize(tokenizer, quant_config=quant_config)

# 保存量化后的模型
print("💾 保存量化模型...")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f"✅ 量化完成，模型已保存至: {quant_path}")