import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from soft_moe_pytorch import DynamicSlotsSoftMoE


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):

        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class MoEEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_experts = 4 ):
        super(MoEEncoderLayer, self).__init__()
        self.attention = attention
        self.moe = DynamicSlotsSoftMoE(
            dim = d_model,         # model dimensions
            geglu = True,
            num_experts = num_experts    # number of experts - (they suggest number of experts should be high enough that each of them get only 1 slot. wonder if that is the weakness of the paper?)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):

        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.moe(y)))

        return self.norm2(x + y), attn


class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim #768
        self.num_blocks = 4 
        self.block_size = self.hidden_size // self.num_blocks 
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * self.scale)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size )

        x = torch.fft.fft2(x, dim=(1,2), norm='ortho') # FFT on N dimension

        x_real_1 = F.relu(self.multiply(x.real, self.complex_weight_1[0]) - self.multiply(x.imag, self.complex_weight_1[1]) + self.complex_bias_1[0])
        x_imag_1 = F.relu(self.multiply(x.real, self.complex_weight_1[1]) + self.multiply(x.imag, self.complex_weight_1[0]) + self.complex_bias_1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_2[0]) - self.multiply(x_imag_1, self.complex_weight_2[1]) + self.complex_bias_2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_2[1]) + self.multiply(x_imag_1, self.complex_weight_2[0]) + self.complex_bias_2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1,2), norm="ortho")
        
        # RuntimeError: "fused_dropout" not implemented for 'ComplexFloat'
        x = x.to(torch.float32)
        x = x.reshape(B, N, C)
        return x
    

class EinFFTEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_experts = 4 ):
        super(EinFFTEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.moe = EinFFT(
            dim = d_model,         # model dimensions
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):

        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.moe(y)))

        return self.norm2(x + y), attn


def blockdiag_matmul (x , w ) :
    return torch . einsum (
    "bnm ,... bm -> ... bn", w , x . view (* x . shape [: -1] , w . shape [0] , w . shape [ -1])) . reshape (* x . shape )

class MonarchMatrix ( nn . Module ) :
    def __init__ ( self , sqrt_n : int ) :
        super () . __init__ ()
        self.sqrt_n = sqrt_n
        self.L = nn.Parameter(torch.randn((sqrt_n, sqrt_n, sqrt_n)))
        self.R = nn.Parameter(torch.randn((sqrt_n, sqrt_n, sqrt_n)))
    def forward ( self , x ) :
        x = rearrange (x , "... (m n) -> ... (n m)", n = self.sqrt_n )
        x = blockdiag_matmul (x , self.L)
        x = rearrange (x , "... (m n) -> ... (n m)", n = self.sqrt_n )
        x = blockdiag_matmul (x , self.R )
        return rearrange (x , " ... (m n) -> ... (n m)", n = self.sqrt_n )


class MonarchFMixerLayer ( nn . Module ) :
    def __init__ ( self , sqrt_d : int ) :
        super () . __init__ ()
        self.m3 = MonarchMatrix ( sqrt_d )
        self.m4 = MonarchMatrix ( sqrt_d )
        self.d_kernel = nn.Parameter ( torch.randn (1 , sqrt_d ** 2) )
        self.layer_norm = nn.LayerNorm ( sqrt_d ** 2)
    def forward ( self , x : torch.Tensor ) : # x.shape = (b, n, d)
        x = self.m4 ( torch.relu (self.d_kernel * self.m3 (x) ) ) # mix features
        return x # skip connection
    

class MonarchEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_experts = 4 ):
        super(MonarchEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.moe = MonarchFMixerLayer(
            sqrt_d = int(d_model ** 0.5),         # model dimensions
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):

        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.moe(y))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]

        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

#######################
### Seq mixers layers
#######################


class MonarchMixerLayer ( nn.Module ) :     # Number of tokens effects the number of parameters which is not good!
    def __init__ ( self , sqrt_n : int , sqrt_d : int ) :
        super () . __init__ ()
        self.m1 = MonarchMatrix ( sqrt_n )
        self.m2 = MonarchMatrix ( sqrt_n )
        self.n_kernel = nn.Parameter ( torch.randn ( sqrt_d ** 2 , sqrt_n ** 2) )
    def forward ( self , x : torch.Tensor, k = None, v = None, attn_mask=None, tau=None, delta=None ) : # x.shape = (b, n, d)
        x = self.m2 ( torch.relu ( self.n_kernel * self.m1 (x.transpose ( -1 , -2) ) )
            ).transpose ( -1 , -2) # mix sequence
        return x , None # skip connection


class PoolMixerLayer (nn.Module):
    def __init__ (self, pool_size = 3) :        # x.shape = (b, n, d))
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
    def forward (self, x : torch.Tensor, k = None, v = None, attn_mask=None, tau=None, delta=None ) : # x.shape = (b, n, d)
        x = x.permute(0,2,1)
        x = self.pool(x) - x
        x = x.permute(0,2,1)
        return x, None

### Diff Attention

def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    


class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        model_parallel_size = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads // model_parallel_size
        self.num_kv_heads = num_heads // model_parallel_size
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
    
    def forward(
        self, x : torch.Tensor, k = None, v = None,  # x.shape = (b, n, d)
        attn_mask=None, tau=None, delta=None 
        ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask   
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn, None

##################

class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(
                x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
