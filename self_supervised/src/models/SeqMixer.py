import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
    MoEEncoderLayer,
    EinFFTEncoderLayer,
    MonarchEncoderLayer,
    PoolMixerLayer,
    MultiheadDiffAttn,

)
from ..models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from ..models.layers.Embed import PositionalEmbeddingXBiot
from ..models.hydra.matrix_mixer import MatrixMixer



class SeqMixer(nn.Module):

    def __init__(self, d_model, dropout, max_len, n_heads, d_ff, e_layers, moe = 'mlp', 
                 num_experts = 4, postional_embed = True, seq_mixer = 'transformer',
                 pool_size = 3):
        super(SeqMixer, self).__init__()
        # Embedding
        self.postional_embed = postional_embed
        self.positional_embedding = PositionalEmbeddingXBiot(
            d_model,
            dropout,
            max_len
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        # Encoder
        if seq_mixer == 'transformer':
            if moe == 'moe':
                self.encoder = Encoder(
                    [
                        MoEEncoderLayer(
                            AttentionLayer(
                                FullAttention(
                                    False,
                                    attention_dropout= dropout,
                                    output_attention= False,
                                ),
                                d_model,
                                n_heads,
                            ),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                            num_experts= num_experts
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'mlp' :
                self.encoder = Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention(
                                    False,
                                    attention_dropout= dropout,
                                    output_attention= False,
                                ),
                                d_model,
                                n_heads,
                            ),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'einfft' :
                self.encoder = Encoder(
                    [
                        EinFFTEncoderLayer(
                            AttentionLayer(
                                FullAttention(
                                    False,
                                    attention_dropout= dropout,
                                    output_attention= False,
                                ),
                                d_model,
                                n_heads,
                            ),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'monarch':
                self.encoder = Encoder(
                    [
                        MonarchEncoderLayer(
                            AttentionLayer(
                                FullAttention(
                                    False,
                                    attention_dropout= dropout,
                                    output_attention= False,
                                ),
                                d_model,
                                n_heads,
                            ),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
        elif seq_mixer == 'poolformer':
            if moe == 'mlp':
                self.encoder = Encoder(
                    [
                        EncoderLayer(
                            PoolMixerLayer(pool_size = pool_size),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'moe':
                self.encoder = Encoder(
                    [
                        MoEEncoderLayer(
                            PoolMixerLayer(pool_size = pool_size),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'einfft':
                self.encoder = Encoder(
                    [
                        EinFFTEncoderLayer(
                            PoolMixerLayer(pool_size = pool_size),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'monarch':
                self.encoder = Encoder(
                    [
                        MonarchEncoderLayer(
                            PoolMixerLayer(pool_size = pool_size),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
        elif seq_mixer == 'diffattn':
            if moe == 'mlp':
                self.encoder = Encoder(
                    [
                        EncoderLayer(
                            MultiheadDiffAttn(embed_dim = d_model,
                            depth = 1,
                            num_heads = n_heads,
                            model_parallel_size = 1,),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'moe':
                self.encoder = Encoder(
                    [
                        MoEEncoderLayer(
                            MultiheadDiffAttn(embed_dim = d_model,
                            depth = 1,
                            num_heads = n_heads,
                            model_parallel_size = 1,),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'einfft':
                self.encoder = Encoder(
                    [
                        EinFFTEncoderLayer(
                            MultiheadDiffAttn(embed_dim = d_model,
                            depth = 1,
                            num_heads = n_heads,
                            model_parallel_size = 1,),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'monarch':
                self.encoder = Encoder(
                    [
                        MonarchEncoderLayer(
                            MultiheadDiffAttn(embed_dim = d_model,
                            depth = 1,
                            num_heads = n_heads,
                            model_parallel_size = 1,),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
        else:
            if moe == 'mlp':
                self.encoder = Encoder(
                    [
                        EncoderLayer(
                            MatrixMixer( # {'toeplitz', 'vandermonde', 'cauchy', 'low_rank', 'attention', 'quasiseparable'}
                                matrix_mixer_type= seq_mixer,
                                is_data_dependent= True,
                                d_model= d_model,    # Model dimension d_model
                                qk_dim= d_model // n_heads,  # dimension for QK
                                expand= 2,
                                headdim= 2 * d_model // n_heads, # expand * d_model // n_heads
                            ),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'moe':
                self.encoder = Encoder(
                    [
                        MoEEncoderLayer(
                            MatrixMixer( # {'toeplitz', 'vandermonde', 'cauchy', 'low_rank', 'attention', 'quasiseparable'}
                                matrix_mixer_type= seq_mixer,
                                is_data_dependent= True,
                                d_model= d_model,    # Model dimension d_model
                                qk_dim= d_model // n_heads,  # dimension for QK
                                expand= 2,
                                headdim= 2 * d_model // n_heads, # expand * d_model // n_heads
                            ),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'einfft':
                self.encoder = Encoder(
                    [
                        EinFFTEncoderLayer(
                            MatrixMixer( # {'toeplitz', 'vandermonde', 'cauchy', 'low_rank', 'attention', 'quasiseparable'}
                                matrix_mixer_type= seq_mixer,
                                is_data_dependent= True,
                                d_model= d_model,    # Model dimension d_model
                                qk_dim= d_model // n_heads,  # dimension for QK
                                expand= 2,
                                headdim= 2 * d_model // n_heads, # expand * d_model // n_heads
                            ),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
            elif moe == 'monarch':
                self.encoder = Encoder(
                    [
                        MonarchEncoderLayer(
                            MatrixMixer( # {'toeplitz', 'vandermonde', 'cauchy', 'low_rank', 'attention', 'quasiseparable'}
                                matrix_mixer_type= seq_mixer,
                                is_data_dependent= True,
                                d_model= d_model,    # Model dimension d_model
                                qk_dim= d_model // n_heads,  # dimension for QK
                                expand= 2,
                                headdim= 2 * d_model // n_heads, # expand * d_model // n_heads
                            ),
                            d_model,
                            d_ff,
                            dropout= dropout,
                            activation= "gelu",
                        )
                        for l in range(e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(d_model),
                )




    def forward(self, x_enc): # input should be b,n,c
        # Embedding
        if self.postional_embed:
            enc_out = self.positional_embedding(x_enc)
        else:
            enc_out = x_enc
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(
            enc_out
        )  
        output = self.dropout(output)
        return output