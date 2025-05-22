import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import WindowAttMaskWithRegister
from math import sqrt, log
from dataclasses import dataclass, fields
from typing import Optional
from torch import Tensor
from zeta.nn import MambaBlock
from einops import repeat

@dataclass
class ModelArgs:
    seq_len: int = 512 
    pred_len: int = 336
    dim: int = 16
    head_size: int = 4
    n_block_layers: int = 2
    ffn_multiplier: Optional[float] = 3.
    patch_len: int = 16
    stride: int = 16
    patches_window_len: int = 4
    att_dropout: Optional[float] = 0.1
    dropout: Optional[float] = 0.2
    num_register_tokens: int = 32
    d_state: int = 16,
    d_conv: int = 2,
    expend_ratio_scaler: int = 16
    proj_expend_ratio: int = 4
    proj_squeeze_ratio: int = 2
    dropout_proj: float = 0.05

class AbsulotePositionalEncoder(nn.Module):
    def __init__(self, d_model, num_patches):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(num_patches, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, num_patches).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(log(500.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

class ParallelTimeAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads, bias=False)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads, bias=False)
        self.value_projection = nn.Linear(d_model, d_values * n_heads, bias=False)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, registers: Tensor, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        if registers is not None:
            _, R, _ = registers.shape
        else:
            R = 0
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        if registers is not None:
            queries = torch.cat((registers.view(B, R, H, -1), queries), dim=1)
            keys = torch.cat((registers.view(B, R, H, -1), keys), dim=1)
            values = torch.cat((registers.view(B, R, H, -1), values), dim=1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        )
        out = out.view(B, R + L, -1)
        if registers is not None:
            registers, out = out[:, :R], out[:, R:]

        return self.out_projection(out), registers
        
class FlashAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.attention_dropout = attention_dropout

    def forward(self, queries, keys, values, attn_mask):
        queries, keys, values = queries.transpose(1,2), keys.transpose(1,2), values.transpose(1,2)
        out = F.scaled_dot_product_attention(queries, keys, values, attn_mask=attn_mask.mask, dropout_p=self.attention_dropout)

        return out.transpose(1,2).contiguous()

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class ParallelTimeFFN(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, inter_dim)
        self.linear_2 = nn.Linear(dim, inter_dim)
        self.linear_3 = nn.Linear(inter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_3(F.silu(self.linear_1(x)) * self.linear_2(x))

class ParallelTimeWeighter(nn.Module):
    def __init__(self, num_channels, dim, expend_ratio=16):
        super().__init__()
        num_channels_expend = num_channels * expend_ratio
        self.num_channels= num_channels
        small_dim = int(sqrt(dim))
        self.fc1 = nn.Linear(small_dim * num_channels, num_channels_expend, bias=True)
        self.fc2 = nn.Linear(num_channels_expend, num_channels, bias=True)
        self.squeeze_att = nn.Linear(dim, small_dim, bias=True)
        self.squeeze_ssm = nn.Linear(dim, small_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_att: Tensor, x_ssm: Tensor, return_weights: bool):
        bs, num_patches, _ = x_att.shape
        # x_att.shape, x_ssm.shape: (batch_size, num_patches, dim)
        squeeze_x_att = self.squeeze_att(x_att)
        squeeze_x_ssm = self.squeeze_ssm(x_ssm)
        # -> (batch_size, num_patches, dim, 2)
        input_tensor = torch.stack((x_att, x_ssm), dim=-1)
        # -> (batch_size, num_patches, 1, 2)
        expended_tensor = torch.stack((squeeze_x_att, squeeze_x_ssm), dim=-1).view(bs, num_patches, 1, -1)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(expended_tensor))
        weights = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, weights)
        if return_weights:
            return output_tensor.mean(dim=-1), weights # -> (batch_size, num_patches, dim)
        else:
            return output_tensor.mean(dim=-1), None # -> (batch_size, num_patches, dim)
    
class ParallelTimeDecoder(nn.Module):
    def __init__(self, attention, d_model, ffn_multiplier=4., dropout=0.1, 
                 activation="relu", d_state: int = 16, d_conv: int = 2, 
                 expend_ratio_scaler: int = 16):
        super().__init__()
        d_ff = int(d_model * ffn_multiplier)
        
        self.attention = attention
        self.ssm = MambaBlock(d_model, depth=1, d_state=d_state,expand=1,d_conv=d_conv)
        self.weigther = ParallelTimeWeighter(2, dim=d_model, expend_ratio=expend_ratio_scaler)

        self.norm_ssm = RMSNorm(d_model)
        self.norm_att = RMSNorm(d_model)
        
        self.feed_forward = ParallelTimeFFN(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.dropout_ff = nn.Dropout(dropout) if dropout else nn.Identity()
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: Tensor, registers: Optional[Tensor], attn_mask=None, return_weights=False):
        # x: (-1, num_patches, dim)
        x_norm = self.norm1(x)
        
        # Attention
        x_att, _ = self.attention(
            x_norm, x_norm, x_norm,
            registers=registers,
            attn_mask=attn_mask,
        )
        # Mamba
        x_ssm = self.ssm(x_norm)

        # ParallelTime Weigther
        new_x, weights = self.weigther(self.norm_att(x_att), self.norm_ssm(x_ssm), return_weights)

        x = x + self.dropout(new_x)

        y = self.norm2(x)
        y = self.dropout_ff(self.feed_forward(y))
        return x + y, weights

class ParallelTimeEmbed(nn.Module):
    def __init__(self, patch_len: int, dim: int, n_kernels: int = 16):
        super().__init__()
        self.embed = nn.Linear(patch_len, dim, bias=False)
        self.kernel_size = int(sqrt(patch_len))
        self.feature_map_size = patch_len // self.kernel_size
        self.conv = nn.Conv1d(1, n_kernels, self.kernel_size, self.kernel_size)
        self.linear_conv = nn.Linear(self.feature_map_size, dim)
        self.dim = dim

    def forward(self, x: Tensor):
        # x.shape: (bs, nvars , num_patches, patch_len)
        bs, nvars, num_patches, patch_len = x.shape
        conv_out = self.linear_conv(self.conv(x.view(-1, 1, patch_len)))
        conv = conv_out.mean(dim=-2).view(bs, nvars, num_patches, self.dim)
        return self.embed(x) + conv # -> (bs, nvars , num_patches, patch_len)
    
def params_to_model_args(params: dict):
    model_args_fields = {field.name for field in fields(ModelArgs)}
    filtered_params = {k: v for k, v in params.items() if k in model_args_fields}
    return ModelArgs(**filtered_params)

class ParallelTimeExpandCompressProject(nn.Module):
    def __init__(self, num_patches: int, dim: int, pred_len: int, squeeze_ratio: int=2, 
                 expend_ratio: int=4, dropout: float=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.higher_dim = int(dim * expend_ratio)
        self.lower_dim = dim // squeeze_ratio
        self.expand = nn.Linear(dim, self.higher_dim, bias=False)
        self.compress = nn.Linear(self.higher_dim, self.lower_dim, bias=False)
        self.project = nn.Linear(self.lower_dim * num_patches, pred_len, bias=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor):
        # x.shape: (bs, nvars, num_patches, dim)
        x = self.norm(x)
        bs, nvars, _, _ = x.shape
        x = self.compress(self.expand(x))
        # x -> (bs, nvars, num_patches, lower_dim)
        x = self.project(x.view(bs, nvars, -1))
        return self.dropout(x)
    
class Model(nn.Module):
    """
    ParallelTime
    """
    def __init__(self, params):
        super().__init__()
        params : ModelArgs= params_to_model_args(params)
        self.pred_len = params.pred_len
        self.dim = params.dim
        n_heads = params.dim // params.head_size

        # Patching
        self.patch_len = params.patch_len
        self.stride = params.stride
        self.patches_window_len = params.patches_window_len
        self.num_patches = (params.seq_len - self.patch_len) // self.stride + 1
        assert self.num_patches * self.patch_len == params.seq_len, 'seq_len need to be divisible by patch_len.'
        assert params.dim % params.head_size == 0, 'dim must be divisible by head_size.'

        # Embedding
        self.embed = ParallelTimeEmbed(params.patch_len, params.dim)

        # Positional encoder
        self.pe = AbsulotePositionalEncoder(params.dim, self.num_patches)

        # Registers
        self.n_registers = params.num_register_tokens
        if self.n_registers > 0:
            self.registers = nn.Parameter(
                torch.randn(self.n_registers, params.dim)
            )

        # Parallel time decoder layers
        self.decoder_layers = nn.ModuleList([
            ParallelTimeDecoder(
                ParallelTimeAttentionLayer(
                    FlashAttention(attention_dropout=params.att_dropout),
                    params.dim, 
                    n_heads),
                d_model=params.dim,
                ffn_multiplier=params.ffn_multiplier,
                dropout=params.dropout,
                activation='gelu',
                d_state=params.d_state,
                d_conv=params.d_conv,
                expend_ratio_scaler=params.expend_ratio_scaler,
            ) for _ in range(params.n_block_layers)]
        )

        # Projection
        self.projector = ParallelTimeExpandCompressProject(self.num_patches, params.dim, params.pred_len, 
                                               params.proj_squeeze_ratio, params.proj_expend_ratio, 
                                               dropout=params.dropout_proj)

    def patching(self, x: Tensor):
        # x.shape: (bs, nvars, seq_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)   
        return x # -> (bs, nvars, num_patches, patch_len)
    
    def forecast(self, x: Tensor, return_weights=False):
        # x.shape: (bs, seq_len, nvars)
        batch_size, _, nvars = x.shape 
        # x -> (bs, nvars, seq_len)
        x = x.permute(0, 2, 1).contiguous()   
        
        # Patching
        # x -> (bs, nvars, num_patches, patch_len)
        x = self.patching(x)
        
        # Embedding
        # x -> (bs, nvars, num_patches, dim)
        x = self.embed(x) 

        # Positional encoding
        x = self.pe(x)

        # Registers
        if self.n_registers > 0:
            registers = repeat(
                self.registers.to(x.device), 
                'n d -> b n d', 
                b=batch_size*nvars,
            )
        else:
            registers = None
        attention_mask = WindowAttMaskWithRegister(self.num_patches, self.patches_window_len, 
                                                   x.device, self.n_registers)
        
        x = x.view(-1, self.num_patches, self.dim) # x -> (bs * nvars, num_patches, dim)
        
        weights_list = []
        for decoder_layer in self.decoder_layers:
            x, weights = decoder_layer(x, registers, attn_mask=attention_mask, return_weights=return_weights) 
            if return_weights:
                weights_list.append(weights)

        x = x.view(batch_size, nvars, self.num_patches, self.dim)

        # x -> (bs, pred_len, nvars)
        x = self.projector(x).permute(0, 2, 1)
        
        if return_weights:
            return x, weights_list
        return x

    def forward(self, x, return_weights=False):
        # x.shape: (bs, pred_len, nvars)
        # norm
        series_mean = torch.mean(x, dim=1, keepdim=True)
        series_stdev = torch.sqrt(torch.var(x - series_mean, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - series_mean) / series_stdev
        
        # forecast
        if return_weights:
            x, weight = self.forecast(x, return_weights)
        else:
            x = self.forecast(x)
        # x.shape: (bs, pred_len, nvars)
            
        # denorm
        x = (x * series_stdev) + series_mean
        
        if return_weights:
            return x[:, -self.pred_len:, :], weight
        return x[:, -self.pred_len:, :]