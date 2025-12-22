import torch
import torch.nn as nn
import os
import sys
from transformers import CLIPTextModel, CLIPModel, CLIPVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPVisionEmbeddings, CLIPEncoder
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import math
from typing import Union, Optional
from torch.nn import init
import copy
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf


sys.path.append(os.getcwd())
from utils.functionals import batch_attention, attention, try_load
from model.BasicModules.transformer import BasicTransformer
from model.BasicModules.VectorQuantize import VectorQuantize
from model.BasicModules.mlp import BasicMlp
from model.oned_tokenizer.modeling.titok import TiTok


class PartialConvolution(nn.Module):
    def __init__(self, in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int,
                 padding: int=1,
                 bias=True,
                 epsilon=1e-7,
                 device='cpu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.device = device
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.zeros([self.out_channels, self.in_channels, self.kernel_size, self.kernel_size]))
        if bias:
            self.bias= nn.Parameter(torch.zeros([1, self.out_channels, 1]))
        else:
            self.bias = None

        self.reset_parameters()
        self.to(device)
        return

    def unfold(self, x: torch.Tensor, mask: torch.Tensor = None):
        assert x.ndim == 4

        if mask is not None:
            x *= mask
            mask = mask.unfold(-2, self.kernel_size, self.stride).unfold(-2, self.kernel_size, self.stride)
            mask = mask.reshape(*mask.shape[:2], -1, *mask.shape[-2:])
            mask_sum = mask.sum((-1, -2)).squeeze(-1)

        x = x.unfold(-2, self.kernel_size, self.stride).unfold(-2, self.kernel_size, self.stride)
        x = x.reshape(*x.shape[:2], -1, *x.shape[-2:])

        x = x.reshape(*x.shape[:3], -1)
        x = x.transpose(-1, -2)
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if mask is not None:
            return x, mask_sum

        return x
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        if self.padding is not None and self.padding > 0:
            x = nn.functional.pad(x, [self.padding] * 4)
            mask = mask if mask is None else nn.functional.pad(mask, [self.padding] * 4)

        b, c, h, w = x.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        kernel = self.weight.reshape(self.weight.shape[0], -1).unsqueeze(0)

        bias = 0 if self.bias is None else self.bias
        if mask is not None:
            unfolded_x, mask_sum = self.unfold(x, mask)
            weight = torch.zeros_like(mask_sum)
            idx = torch.where(mask_sum > self.epsilon)
            weight[idx] = (self.kernel_size ** 2) / mask_sum[idx]
            x = torch.matmul(kernel, unfolded_x) + bias
            # x[torch.where(mask_sum <= self.epsilon)] = 0.
            x *= (mask_sum > self.epsilon).float()
        else:
            unfolded_x = self.unfold(x)
            x = torch.matmul(kernel, unfolded_x) + bias
        
        x = x.reshape(b, self.out_channels, out_h, out_w)

        return x

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
        
        return

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        """
        初始化 Positional Embedding 类。

        参数:
            d_model (int): 嵌入维度（即每个词向量的维度）。
            max_len (int): 最大序列长度，默认为 512。
        """
        super(PositionalEmbedding, self).__init__()
        
        # 创建一个位置编码矩阵
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model // 2)
        
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        
        # 将位置编码矩阵注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。

        返回:
            torch.Tensor: 添加了位置编码的张量，形状为 (batch_size, seq_len, d_model)。
        """
        # 将位置编码加到输入张量上
        x = x + self.pe[:x.size(1), :]  # 只取前 seq_len 个位置编码
        return x
    

class VisionEmbedding(nn.Module):
    def __init__(self, in_channels: int, image_size: int, config: CLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = image_size
        self.patch_size = int((self.image_size // 4) ** 2)
        self.in_channels = in_channels

        # self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # downsample 4 times
        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(
        #     in_channels=self.in_channels,
        #     out_channels=self.in_channels * 4,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     bias=False,
        # ),
        # nn.Conv2d(
        #     in_channels=self.in_channels * 4,
        #     out_channels=self.embed_dim,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     bias=False,
        # ))

        self.patch_embedding = nn.Sequential(
            PartialConvolution(in_channels=self.in_channels,
                               out_channels=self.in_channels * 4,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False),
            PartialConvolution(in_channels=self.in_channels * 4,
                               out_channels=self.embed_dim,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False))

        self.position_embedding = PositionalEmbedding(self.embed_dim, self.patch_size)
        return
    
    def forward(self, image: torch.Tensor, **kwargs):
        batch_size = image.shape[0]
        embeddings = self.patch_embedding(image).reshape(batch_size, self.embed_dim, -1).transpose(-1, -2)
        # class_embedding = self.class_embedding.expand(batch_size, 1, -1)
        # embeddings = torch.cat([class_embedding, embeddings], dim=1)
        embeddings = self.position_embedding(embeddings)

        return embeddings


class VisionTransformer(nn.Module):
    replace_modules = ('embeddings',)

    def __init__(
            self, 
            num_channels: int, 
            patch_size: int,
            config: CLIPVisionConfig,
            image_size=64
        ):
        super().__init__()
        embed_dim = config.hidden_size

        config.num_channels = num_channels
        config.patch_size = patch_size
        config.image_size = image_size
        self.config = config
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    
    @classmethod
    def from_pretrained(cls, num_channels, patch_size, fp: str, image_size=64):
        clip_model: CLIPVisionModel = CLIPVisionModel.from_pretrained(fp)
        clip_model = clip_model.vision_model

        model = cls(num_channels, patch_size, clip_model.config, image_size)

        for name, sub_module in model.named_children():
            if name not in model.replace_modules:
                sub_module.load_state_dict(getattr(clip_model, name).state_dict())

        return model

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ):
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        self.path_a = nn.ModuleList([nn.Linear(in_dim, in_dim),
                                     nn.Linear(in_dim, in_dim)])
        self.path_main = nn.ModuleList([nn.Linear(in_dim, in_dim),
                                        nn.Linear(in_dim * 2, in_dim * 2)],
                                        nn.Linear(in_dim * 3, out_dim))
        self.path_b = nn.ModuleList([nn.Linear(in_dim, in_dim),
                                     nn.Linear(in_dim * 2, in_dim * 2)])
        
        return
    
    def forward(self, x):
        x_a = x
        x_main = x
        for i in range(2):
            x_a = self.path_a[i](x_a)
            x_b = self.path_b[i](x_main)
            x_main = self.path_main[i](x_main) + x_b
            x_main = torch.concat([x_main, x_a], dim=-1)

        return self.path_main[2](x_main)


class BatchAttention(nn.Module):
    def __init__(self, q_dim: int, k_dim: int, v_dim: int, heads: int = 8):
        super().__init__()

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.heads = heads

        self.to_q = nn.Linear(q_dim, q_dim, bias=False)
        self.to_k = nn.Linear(k_dim, k_dim, bias=False)
        self.to_v = nn.Linear(v_dim, v_dim, bias=False)

        self.layer_norm_q = nn.LayerNorm(q_dim)
        self.layer_norm_k = nn.LayerNorm(k_dim)
        self.layer_norm_v = nn.LayerNorm(v_dim)
        
        self.to_out = nn.LayerNorm(q_dim)

        return
    
    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor,
                use_batch_attn=True):
        residual = q
        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        if use_batch_attn:
            hidden_states = batch_attention(q, k, v, self.heads).squeeze(1)
        else:
            hidden_states = attention(q, k, v, self.heads).squeeze(1)

        return self.to_out(hidden_states) + residual


class Embedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        image_size: int,
        token_len: int=32,
        transformer_num: int=4,
        use_attention_mask=False,
        mlp_ratio=4.0
    ):
        super().__init__()

        self.seq_len = (image_size // patch_size) ** 2
        self.token_len = token_len

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_attention_mask = use_attention_mask
        self.transformer_num = transformer_num

        self.class_embedding = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.class_embedding, std=0.2)

        self.token_embedding = nn.Parameter(torch.zeros(1, token_len, self.embed_dim))
        nn.init.normal_(self.token_embedding, std=0.2)

        self.position_embedding = nn.Parameter(torch.zeros(1 + self.seq_len, self.embed_dim))
        nn.init.normal_(self.position_embedding, std=0.2)

        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.ln_pre = nn.LayerNorm(embed_dim)

        self.transformers = nn.ModuleList([
            BasicTransformer(self.embed_dim, heads=1) for _ in range(transformer_num)
        ])

        self.mlps = nn.ModuleList([
            BasicMlp(embed_dim, mlp_ratio=mlp_ratio) for _ in range(transformer_num)
        ])

        self.ln_post = nn.LayerNorm(embed_dim)

        return
    
    def down_sample(self, x: torch.Tensor, scale=0.5):
        image_size = int(math.sqrt(x.shape[1]))
        x = x.transpose(-1, -2)
        x = x.reshape(*x.shape[:2], image_size, image_size)
        x = nn.functional.interpolate(x, scale_factor=scale)
        x = x.reshape(*x.shape[:2], -1)
        x = x.transpose(-1, -2)

        return x

    def forward(self, x, attention_mask=None):
        batch_size = x.shape[0]

        x = self.patch_embedding(x)
        x = x.reshape(batch_size, x.shape[1], -1)
        x = x.transpose(-1, -2)

        class_embedding = self.class_embedding.repeat(batch_size, 1, 1)
        token_embedding = self.token_embedding.repeat(batch_size, 1, 1)
        embedding_len = class_embedding.shape[1] + self.token_embedding.shape[1]
        x = torch.concat([class_embedding, x], dim=1) + self.position_embedding
        x = torch.concat([token_embedding, x], dim=1)

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size, -1)
            attention_mask = torch.concat([torch.zeros(batch_size, embedding_len, device=x.device), attention_mask], dim=1)
            attention_mask = attention_mask * -1e5
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat(1, attention_mask.shape[-1], 1).unsqueeze(1)

        x = self.ln_pre(x)

        for i in range(self.transformer_num):
            # print(i, torch.cuda.memory_allocated() // 1024 ** 2)
            # if self.training:
            #     x = checkpoint(self.transformers[i], x, None, attention_mask, use_reentrant=True)
            # else:
            x = self.transformers[i](x, attention_mask=attention_mask) + x
            x = self.mlps[i](x) + x

        x = self.ln_post(x[:, :self.token_len, :])

        return x


class DeEmbedding(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        patch_size: int
    ):
        super().__init__()
        self.de_conv = nn.ConvTranspose2d(in_channels, out_channels, patch_size, patch_size)

        return

    def forward(self, x: torch.Tensor):
        image_size = int(math.sqrt(x.shape[1]))

        x = x.transpose(-1, -2)
        x = x.reshape(*x.shape[:2], image_size, image_size)
        x = self.de_conv(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        token_len=32,
        transformer_num=4,
        embedding_transformer_num=2,
        use_attention_mask=False,
        mlp_ratio=4.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.transformer_num = transformer_num

        self.patch_embed = Embedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            image_size=image_size,
            token_len=token_len,
            transformer_num=embedding_transformer_num,
            use_attention_mask=use_attention_mask,
            mlp_ratio=mlp_ratio
        )

        self.position_embedding = nn.Parameter(torch.zeros(1, token_len, embed_dim))
        nn.init.normal_(self.position_embedding)

        self.ln_pre = nn.LayerNorm(embed_dim)

        self.mlps = nn.ModuleList([
            BasicMlp(embed_dim, mlp_ratio) for _ in range(transformer_num)
        ])

        self.transformers = nn.ModuleList([
            BasicTransformer(self.embed_dim, heads=8) for _ in range(transformer_num)
            # nn.MultiheadAttention(self.embed_dim, num_heads=8) for _ in range(transformer_num)
        ])

        self.ln_post = nn.LayerNorm(embed_dim)
    
        return

    def forward(self, x: torch.Tensor, attention_mask=None):
        hidden_states = self.patch_embed(x, attention_mask)
        hidden_states = hidden_states + self.position_embedding

        hidden_states = self.ln_pre(hidden_states)

        for i in range(self.transformer_num):
            # if self.training:
            #     hidden_states = checkpoint(self.transformers[i], hidden_states, use_reentrant=True)
            #     hidden_states = checkpoint(self.mlps[i], hidden_states, use_reentrant=True)
            # else:
            hidden_states = self.transformers[i](hidden_states) + hidden_states
            # hidden_states = self.transformers[i](hidden_states, hidden_states, hidden_states)[0] + hidden_states
            hidden_states = self.mlps[i](hidden_states) + hidden_states

        hidden_states = self.ln_post(hidden_states)

        return hidden_states
    

class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        token_len=32,
        transformer_num=4,
        mlp_ratio=4.0
    ):
        super().__init__()

        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.token_len = token_len
        self.transformer_num = transformer_num
        self.image_size = image_size

        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.position_embedding = nn.Parameter(torch.zeros(1, image_size ** 2, embed_dim))
        nn.init.normal_(self.position_embedding)

        self.ln_pre = nn.LayerNorm(embed_dim)

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                nn.SiLU(),
                nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
            ) for _ in range(transformer_num)
        ])

        self.transformers = nn.ModuleList([
            BasicTransformer(self.embed_dim, heads=8) for _ in range(transformer_num)
            # nn.MultiheadAttention(self.embed_dim, num_heads=8) for _ in range(transformer_num)
        ])

        self.ln_post = nn.LayerNorm(embed_dim)

        self.conv_out = nn.Conv2d(embed_dim, out_channels, 1, 1)
        
        return

    def forward(self, x):
        bsz = x.shape[0]

        head_embeddings = x
        null_token = self.null_token.repeat(bsz, self.image_size ** 2, 1)
        null_token = null_token + self.position_embedding

        x = torch.concat([head_embeddings, null_token], dim=1)

        x = self.ln_pre(x)
        for i in range(self.transformer_num):
            # if self.training:
            #     x = checkpoint(self.transformers[i], x, use_reentrant=True)
            #     x = checkpoint(self.mlps[i], x, use_reentrant=True)
            # else:
            # x = self.transformers[i](x, x, x)[0] + x
            x = self.transformers[i](x) + x
            x = self.mlps[i](x) + x

        x = x[:, head_embeddings.shape[1]:, ...]
        x = self.ln_post(x)
        image_size = int(math.sqrt(x.shape[1]))
        x = x.transpose(-1, -2)
        x = x.reshape(*x.shape[:2], image_size, image_size).contiguous()
        x = self.conv_out(x)

        return x


class EmbeddingMatcher(nn.Module):
    trainable_module = ('proj_out', 'batch_attn', 'embeddings')
    fine_module = ('pixel_decoder', 'pixel_quantize', 'pixel_encoder')
    def __init__(
        self,
        backbone: TiTok,
    ):
        super().__init__()
        self.backbone: TiTok = backbone

        self.fine = dict()

        # self.proj_out = nn.Linear(self.embeddings.embed_dim, self.hidden_size)

        return

    def forward(
        self, 
        image_masked: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor=None
    ):
        decoded, result_dict = self.backbone(image_masked, attention_mask, image)
        
        return decoded, result_dict
    
    def encode(self, x: torch.Tensor, attention_mask=None):
        z_quantized, result_dict = self.backbone.encode(x, attention_mask=attention_mask)

        return z_quantized, result_dict
    
    def decode(self, x: torch.Tensor):
        x = self.backbone.decode(x)

        return x
    
    def register_fine_paramters(self, state_dict: dict):
        self.fine = dict()

        for name, param in self.named_parameters():
            name_ = name.split('.')
            name_ = '.'.join(name_[1:])

            if name_ in state_dict.keys() and param.shape == state_dict[name_]:
                self.fine[name] = param
            
        return
        
    def raw_parameters(self):
        params = list()

        for name, child in self.backbone.named_children():
            if name not in ['pixel_decoder', 'pixel_quantize', 'pixel_encoder']:
                params.extend(list(child.parameters()))
            else:
                print(f'{name} \'s param is ignored')

        # for name, param in self.named_parameters():
        #     if name not in self.fine.keys():
        #         params.append(param)
                
        return params
    
    def fine_parameters(self):
        params = list()

        for name, child in self.named_children():
            if name not in ['pixel_decoder', 'pixel_quantize']:
                params.extend(list(child.parameters()))
            else:
                print(name)

        # for name, param in self.named_parameters():
        #     if name in self.fine.keys():
        #         params.append(param)
                
        return params
    
    def train_encoder(self):
        for name, child in self.backbone.named_children():
            if name in self.fine_module:
                child.eval()
                child.requires_grad_(False)
            else:
                child.train()
                child.requires_grad_(True)
        return
    
    def train_decoder(self):
        for name, child in self.backbone.named_children():
            if name in self.fine_module:
                child.train()
                child.requires_grad_(True)
            else:
                child.eval()
                child.requires_grad_(False)
        return

    def train_(self):
        self.train()

        # for module in self.modules():
        #     for param in module.parameters():
        #         param.requires_grad = True
        
        # for name, module in self.named_children():
        #     if not name.endswith(self.trainable_module):
        #         for param in module.parameters():
        #             param.requires_grad = False
        #     else:
        #         for param in module.parameters():
        #             param.requires_grad = True
        
        return
    
    @classmethod
    def from_pretrained(cls, config_path: str, ckpt_path: str):
        config = OmegaConf.load(config_path)
        pretrained = TiTok.from_pretrained(ckpt_path)

        model = TiTok(config)
        model = try_load(model, pretrained)

        return cls(model)

    

if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.getcwd())

    # from model.EmbeddingManager.EmbeddingManager import VisionTextEmbedding

    # vision_text_embedding = VisionTextEmbedding('pretrained/stable-diffusion-v1-4/tokenizer', 'pretrained/stable-diffusion-v1-4/text_encoder')
    # vision_text_embedding.add_tokens(['*'])
    
    # bsz = 2
    # prompt = ['*'] * bsz
    embedding_matcher = EmbeddingMatcher(4, 64, 768, './pretrained/clip-vit-large-patch14')
    # image = torch.rand([bsz, 4, 64, 64])

    # image_embeds = embedding_matcher(image, all_features=True)

    # result = vision_text_embedding(prompt, image_embeds)

    # print(result.shape)
