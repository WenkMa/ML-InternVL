import torch
import torch.nn as nn
import torch.nn.functional as F


class BiCrossAttention(nn.Module):
    def __init__(self,
                 query_dim=512,
                 kv_dim=512,
                 num_heads=4,
                 mlp_dim=512,
                 dropout=0.1):
        super().__init__()

        # 新增线性层，将输入的128维提升到query_dim
        self.query_up_proj = nn.Linear(128, query_dim)  # 根据实际输入维度调整

        # 第一层交叉注意力
        self.cross_attn_1 = CrossAttention(
            dim=query_dim,
            context_dim=kv_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 第二层交叉注意力
        self.cross_attn_2 = CrossAttention(
            dim=query_dim,
            context_dim=kv_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # MLP模块
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, query_dim)
        )

        # 归一化层
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)

    def forward(self,
                user_query,
                learnable_token,
                document_features,
                query_pos_emb=None,
                key_pos_emb=None):
        # 合并用户Query与可学习Token
        # 调整learnable_token的重复次数以匹配user_query的序列长度
        learnable_token = learnable_token.repeat(1, user_query.shape[1], 1)
        print("user_query shape:", user_query.shape)  # [1, 10, 128]
        print("learnable_token shape after repeat:", learnable_token.shape)  # [1, 10, 128]
        q = torch.cat([user_query, learnable_token], dim=1)
        print("q shape before projection:", q.shape)  # [1, 20, 128]

        # 将q的维度从128扩展到query_dim (512)
        q = self.query_up_proj(q)
        print("q shape after projection:", q.shape)  # [1, 20, 512]

        # 第一层交叉注意力
        attn_output1, attn_weights1 = self.cross_attn_1(
            q=q,
            context=document_features,
            query_pos_emb=query_pos_emb,
            key_pos_emb=key_pos_emb
        )
        x = self.norm1(attn_output1 + q)

        # 第二层交叉注意力
        attn_output2, attn_weights2 = self.cross_attn_2(
            q=x,
            context=document_features,
            query_pos_emb=query_pos_emb,
            key_pos_emb=key_pos_emb
        )
        x = self.norm2(attn_output2 + x)

        # MLP层
        x = self.mlp(x)
        x = self.norm3(x + x)

        return x, attn_weights1, attn_weights2


class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 context_dim,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()

        # 使用单个线性层处理输入，移除冗余的投影层
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.scale = dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(self,
                q,
                context,
                query_pos_emb=None,
                key_pos_emb=None):

        # 处理位置编码
        if query_pos_emb is not None:
            q = q + query_pos_emb
        if key_pos_emb is not None:
            context = context + key_pos_emb

        # 线性变换
        q = self.to_q(q)
        k = self.to_k(context)
        v = self.to_v(context)

        # 分头处理
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 计算注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context_vec = (attn_weights @ v).transpose(1, 2).contiguous()
        context_vec = context_vec.view(context_vec.size(0), -1, self.num_heads * self.head_dim)

        return context_vec, attn_weights

    def _split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)


if __name__ == '__main__':
    # 确保query_dim与输入维度匹配
    bi_ca = BiCrossAttention(query_dim=512, kv_dim=512)

    # 输入数据
    user_query = torch.randn(1, 10, 128)  # [batch, seq_len, input_dim=128]
    learnable_token = torch.randn(1, 1, 128)  # [batch, 1, input_dim=128]
    document_features = torch.randn(1, 32, 512)  # [batch, img_h*img_w, dim]

    # 前向传播
    output, attn1, attn2 = bi_ca(
        user_query=user_query,
        learnable_token=learnable_token,
        document_features=document_features
    )
    print("Output shape:", output.shape)  # [1, 20, 512]