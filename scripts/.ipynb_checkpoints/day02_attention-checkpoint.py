import torch
import torch.nn.functional as F

# 1. 定义三个向量 [1,0], [0,1], [1,1] (简化，Q=K=V)
X = torch.tensor([[1.0, 0.0], 
                  [0.0, 1.0], 
                  [1.0, 1.0]])

Q = K = V = X 
d_k = Q.size(-1)

# 2. 计算 Q 和 K 的点积 (Scaled Dot-Product)
scores = torch.matmul(Q, K.T) / (d_k ** 0.5)
print("--- 原始注意力分数 ---")
print(scores)

# 3. 加上因果掩码 (Causal Mask) - 只能看前面，不能看未来
mask = torch.triu(torch.ones(scores.shape), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
print("\n--- 加上因果掩码后 (上三角变成 -inf) ---")
print(scores)

# 4. Softmax 归一化
attn_weights = F.softmax(scores, dim=-1)
print("\n--- 注意力权重矩阵 ---")
print(attn_weights)

# 5. 乘以 V 得到最终输出
output = torch.matmul(attn_weights, V)
print("\n--- 最终输出 ---")
print(output)