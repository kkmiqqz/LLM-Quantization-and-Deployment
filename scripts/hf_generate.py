import argparse
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="通用大模型推理脚本")
    parser.add_argument("--model_dir", type=str, default="models/qwen-hf", help="模型路径")
    parser.add_argument("--prompt", type=str, required=True, help="输入的提示词")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="最大生成token数")
    args = parser.parse_args()

    print(f"⏳ 正在加载模型: {args.model_dir}...")
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    load_end = time.time()
    print(f"✅ 模型加载耗时: {load_end - load_start:.2f} 秒")

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    input_len = model_inputs.input_ids.shape[1]

    # 正式测速（不包含加载时间）
    start_time = time.time()
    
    # 贪心生成 (do_sample=False)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=args.max_new_tokens, 
            do_sample=False
        )
        
    end_time = time.time()

    # 只截取新生成的 token
    new_tokens = generated_ids[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # 计算真实生成的速度
    output_len = len(new_tokens)
    time_cost = end_time - start_time
    speed = output_len / time_cost if time_cost > 0 else 0

    print("\n" + "="*40)
    print(f"Prompt: {args.prompt}")
    print(f"输出: {response}")
    print("-" * 40)
    print(f"生成长度: {output_len} tokens")
    print(f"生成耗时: {time_cost:.2f} 秒")
    print(f"生成速度: {speed:.2f} tokens/s")
    print("="*40)

if __name__ == "__main__":
    main()