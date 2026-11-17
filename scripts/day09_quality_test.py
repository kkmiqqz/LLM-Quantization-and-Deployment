import os

models = ["qwen-f16.gguf", "qwen-q8.gguf", "qwen-q4.gguf"]
prompt = "你现在是一个设备日志解析器。请从以下日志中提取设备ID、告警级别和故障原因，并严格以JSON格式输出，不要包含任何其他废话。日志：'设备ID: 8848, 级别: CRITICAL, 故障原因: 温度过高引发自动关机'"

os.makedirs("results", exist_ok=True)
output_file = "results/day09_quality_compare.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("================ Day 9: JSON 抽取测试结果 ================\n\n")

for model in models:
    print(f"\n================ 正在测试模型: {model} ================")
    
    # 构造 shell 命令，强制捕获 stderr
    cmd = f"./llama.cpp/build/bin/llama-cli -m models/{model} -p \"{prompt}\" -n 64 -ngl 99 --no-display-prompt -st 2>&1"
    
    # 直接执行并抓取所有输出
    result = os.popen(cmd).read()
    
    # 简单粗暴地提取 JSON 部分（寻找 ```json 到 ``` 之间的内容）
    if "```json" in result:
        json_output = result.split("```json")[1].split("```")[0].strip()
    else:
        # 如果没有 markdown 标记，直接取最后几行
        json_output = result.strip().split('\n')[-5:]
        json_output = "\n".join(json_output)
    
    print(json_output)
    
    # 追加写入结果文件
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"--- Model: {model} ---\n")
        f.write(json_output + "\n\n")

print(f"\n✅ 完美！所有真实 JSON 结果已保存至 {output_file}")