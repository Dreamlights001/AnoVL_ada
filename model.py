from torch import Tensor, nn
import torch
from torch.nn import functional as F
from sklearn.metrics import pairwise
import numpy as np
from adapter import ResidualAdapter, MultiHeadAdapter, VisualAdapter


class TextAdapter(nn.Module):
    def __init__(self, text_embeddings, label=None, beta=5.5):
        super(TextAdapter, self).__init__()
        text_embeddings = torch.cat((text_embeddings[..., 0], text_embeddings[..., 1]), dim=0)
        input_dim = text_embeddings.shape[1]

        # 使用新的Adapter模块
        self.adapter = ResidualAdapter(
            input_dim=input_dim,
            bottleneck_dim=input_dim // 4,
            adapter_scalar=0.1
        )

        self.text_embeddings = text_embeddings
        self.noise_level = 1
        self.mask_ratio = 0.25
        self.beta = beta

    def adapter_forward(self, img):
        img = img / img.norm(dim=-1, keepdim=True)
        output = self.adapter(img)

        # 确保输出形状与text_embeddings兼容 - 输出是N×C维度
        # 通过归一化和重塑操作确保维度匹配
        output = output / output.norm(dim=-1, keepdim=True)

        # 按照原始AnoVL的处理方式，进行线性投影
        affinity = F.linear(output, self.text_embeddings)
        affinity = torch.tanh(affinity)  # 使用tanh激活，与原始代码保持一致

        # 线性投影回原始空间
        output = F.linear(affinity, self.text_embeddings.t())

        return output

    def mask_aug(self, true_feats):
        N, H, W, C = true_feats.shape

        ids_noise = torch.rand(N, H * W, device=true_feats.device)
        ids_shuffle = torch.argsort(ids_noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_mask = int(H * W * self.mask_ratio)

        noise = torch.normal(0, 0.05 * 1.1 ** 2, true_feats.shape).to(true_feats.device)
        fake_feats = [true_feats]
        noise_masks = []
        for i in range(int(1 / self.mask_ratio)):
            mask = torch.zeros([N, H * W], device=true_feats.device)
            if i != int(1 / self.mask_ratio):
                mask[:, i * len_mask:(i + 1) * len_mask] = 1
            else:
                mask[:, i * len_mask:] = 1
            noise_mask = torch.gather(mask, dim=1, index=ids_restore)
            noise_masks.append(noise_mask)
            fake_feat = true_feats + noise * noise_mask.view(N, H, W, 1)
            fake_feats.append(fake_feat)
        return torch.stack(fake_feats, dim=0).view(-1, H, W, C), torch.stack(noise_masks, dim=0).view(-1, H, W, 1)

    def aug(self, true_feat):
        N, H, W, C = true_feat.shape
        feat_list = [true_feat]
        for n in range(self.noise_level):
            noise = torch.normal(0, 0.05 * 1.1 ** (n + 1), true_feat.shape).to(true_feat.device)
            fake_feat = true_feat + noise
            feat_list.append(fake_feat)
        return torch.stack(feat_list, dim=0).view(-1, H, W, C)

    def forward(self, x, is_test=False, scale=0.1):
        if not is_test:
            x = self.aug(x)
        if len(x.shape) == 4:
            N, H, W, C = x.shape
            x = x.view(N, H * W, C)
            x = 0.5 * x + 0.5 * self.adapter_forward(x)
            x = x.view(N, H, W, C)
        else:
            x = 0.5 * x + 0.5 * self.adapter_forward(x)
        return x


class Adapter(nn.Module):
    def __init__(self, text_embeddings, label=None, beta=5.5):
        super(Adapter, self).__init__()
        text_embeddings = torch.cat((text_embeddings[..., 0], text_embeddings[..., 1]), dim=0)
        input_dim = text_embeddings.shape[1]

        # 使用新的VisualAdapter模块
        self.adapter = VisualAdapter(
            input_dim=input_dim,
            bottleneck_dim=input_dim // 4,
            num_heads=8,
            adapter_scalar=0.1,
            use_multi_head=True
        )

        self.noise_level = 1
        self.mask_ratio = 0.25
        self.beta = beta

    def forward(self, x, is_test=False, scale=0.1):
        # 直接使用VisualAdapter的forward方法
        return self.adapter(x, is_test)


class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model_name, model):
        super(LinearLayer, self).__init__()
        if 'ViT' in model_name:
            self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
        else:
            self.fc = nn.ModuleList([nn.Linear(dim_in * 2 ** (i + 2), dim_out) for i in range(k)])
        self.ln = model.visual.ln_post
        self.proj = model.visual.proj

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                # tokens[i] = self.fc[i](tokens[i][:, 1:, :])
                tokens[i] = self.ln(tokens[i][:, 1:, :]) @ self.proj
            else:
                assert 1 == 2, "Not completed!"
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens
