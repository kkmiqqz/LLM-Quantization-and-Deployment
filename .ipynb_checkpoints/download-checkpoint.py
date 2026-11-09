from modelscope import snapshot_download

# 注意确认一下这个路径是不是你当前的持久化路径
model_dir = snapshot_download(
    'Qwen/Qwen2.5-1.5B-Instruct', 
    cache_dir='/root/rivermind-fs/models'
)
print(f"✅ 下载完成：{model_dir}")