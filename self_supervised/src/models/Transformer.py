import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
    MoEEncoderLayer,
    EinFFTEncoderLayer,
    MonarchEncoderLayer,
)
from ..models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from ..models.layers.Embed import PositionalEmbeddingXBiot



class Transformer(nn.Module):

    def __init__(self, d_model, dropout, max_len, n_heads, d_ff, e_layers, moe = 'mlp', num_experts = 4, postional_embed = True):
        super(Transformer, self).__init__()
        # Embedding
        self.postional_embed = postional_embed
        self.positional_embedding = PositionalEmbeddingXBiot(
            d_model,
            dropout,
            max_len
        )
        # Encoder
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
        self.act = F.gelu


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