import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="通用大模型推理脚本")
    parser.add_argument("--model_dir", type=str, default="models/qwen-hf", help="模型路径")
    parser.add_argument("--prompt", type=str, required=True, help="输入的提示词")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="最大生成token数")
    args = parser.parse_args()

    print(f"⏳ 正在加载模型: {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    input_len = model_inputs.input_ids.shape[1]

    # 贪心生成 (do_sample=False)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=args.max_new_tokens, 
            do_sample=False # 关闭随机采样
        )

    # 只截取新生成的 token
    new_tokens = generated_ids[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print("\n" + "="*40)
    print(f"Prompt: {args.prompt}")
    print(f"输出: {response}")
    print("="*40)

if __name__ == "__main__":
    main()