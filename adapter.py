import torch
from torch import nn
import torch.nn.functional as F


class ResidualAdapter(nn.Module):
    def __init__(
            self,
            input_dim: int,
            bottleneck_dim: int = None,
            adapter_scalar: float = 0.1,
            use_bn: bool = True
    ):
        super().__init__()

        # 如果没有指定bottleneck_dim，默认使用input_dim的1/4
        if bottleneck_dim is None:
            bottleneck_dim = input_dim // 4

        self.adapter_scalar = adapter_scalar
        self.use_bn = use_bn

        # 定义Adapter的主要组件
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.non_linear = nn.GELU()  # 使用GELU激活函数，这是Transformer中常用的
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(input_dim)

        # 如果使用Batch Normalization
        if use_bn:
            self.bn = nn.BatchNorm1d(input_dim)

        # 初始化方法
        self.init_weights()

    def init_weights(self):
        # 使用较小的初始值以确保残差连接的稳定性
        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        # 保存输入形状
        original_shape = x.shape

        # 保存输入用于残差连接
        residual = x

        # 应用Layer Normalization
        x = self.layer_norm(x)

        # Adapter的主要前向传播
        x = self.down_proj(x)
        x = self.non_linear(x)
        x = self.up_proj(x)

        # 如果使用Batch Normalization
        if self.use_bn:
            shape = x.shape
            x = x.view(-1, shape[-1])
            x = self.bn(x)
            x = x.view(*shape)

        # 应用残差连接和缩放因子
        x = self.adapter_scalar * x + residual

        # 确保输出形状与输入形状一致
        assert x.shape == original_shape, f"输出形状 {x.shape} 与输入形状 {original_shape} 不一致"

        return x


class MultiHeadAdapter(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_heads: int = 8,
            bottleneck_dim: int = None,
            adapter_scalar: float = 0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        # 为每个头创建一个Adapter
        self.adapters = nn.ModuleList([
            ResidualAdapter(
                input_dim=self.head_dim,
                bottleneck_dim=bottleneck_dim // num_heads if bottleneck_dim else None,
                adapter_scalar=adapter_scalar
            ) for _ in range(num_heads)
        ])

    def forward(self, x):
        # 输入形状: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape

        # 将输入分割成多个头
        x_reshaped = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 对每个头应用相应的Adapter
        outputs = []
        for i in range(self.num_heads):
            head_output = self.adapters[i](x_reshaped[..., i, :])
            outputs.append(head_output)

        # 合并所有头的输出
        x = torch.stack(outputs, dim=2)
        x = x.view(batch_size, seq_len, -1)

        # 确保输出维度与输入维度匹配
        assert x.shape[0] == batch_size and x.shape[1] == seq_len and x.shape[2] == self.input_dim, \
            f"输出形状 {x.shape} 与预期 ({batch_size}, {seq_len}, {self.input_dim}) 不一致"

        return x


class VisualAdapter(nn.Module):
    def __init__(
            self,
            input_dim: int,
            bottleneck_dim: int = None,
            num_heads: int = 8,
            adapter_scalar: float = 0.1,
            use_multi_head: bool = True
    ):
        super().__init__()

        self.use_multi_head = use_multi_head

        if use_multi_head:
            self.adapter = MultiHeadAdapter(
                input_dim=input_dim,
                num_heads=num_heads,
                bottleneck_dim=bottleneck_dim,
                adapter_scalar=adapter_scalar
            )
        else:
            self.adapter = ResidualAdapter(
                input_dim=input_dim,
                bottleneck_dim=bottleneck_dim,
                adapter_scalar=adapter_scalar
            )

    def forward(self, x, is_test=False):
        # 保存原始形状
        original_shape = x.shape

        if not is_test:
            # 在训练时添加噪声增强
            noise = torch.normal(0, 0.05, x.shape, device=x.device)
            x = x + noise

        # 处理不同维度的输入
        if len(original_shape) == 4:  # [batch, height, width, channels]
            N, H, W, C = original_shape
            x = x.view(N, H * W, C)  # 展平空间维度
            x = self.adapter(x)
            # 确保输出与输入具有相同的通道数
            x = x.view(N, H, W, C)  # 恢复原始形状
        else:  # [batch, sequence_length, channels]
            x = self.adapter(x)
            # 确保输出维度与输入维度一致
            if x.shape != original_shape:
                x = F.interpolate(x.transpose(1, 2), size=original_shape[1], mode='linear').transpose(1, 2)

        # 确保输出形状与输入形状完全一致
        assert x.shape == original_shape, f"输出形状 {x.shape} 与输入形状 {original_shape} 不一致"

        return x