
__all__ = ['PatchTST','PITS']

# Cell
from typing import Optional
import torch
from torch import nn
from torch import Tensor

from models.layers.basics import *
from models.losses import *
# from ..models.Transformer import Transformer
from models.SeqMixer import SeqMixer
import torch_dct as dct


import numpy as np

class Model(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x num_class] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, configs):

        super(Model, self).__init__()

        individual = False

        assert configs.task_name in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'
        
        # Backbone
        self.backbone = DeepMLPencoder(c_in = configs.enc_in, patch_len = configs.patch_len,
                                        stride = configs.stride, features_domain = configs.features_domain,
                                        d_model= configs.d_model, shared_embedding= configs.shared_embedding,
                                        hidden_depth= configs.hidden_depth, h_mode = configs.h_mode, seq_mixing = configs.seq_mixing,
                                        dropout = configs.dropout, max_len = configs.max_len, n_heads = configs.n_heads, 
                                        d_ff = configs.d_ff, e_layers = configs.e_layers,
                                        num_experts = configs.num_experts , moe = configs.moe,
                                        postional_embed = configs.postional_embed, pool_size = configs.pool_size,
                                        )
        
        # Head
        self.n_vars = configs.enc_in
        self.head_type = configs.task_name
        self.pretrain_task = 'PI'
        self.mean_norm_pretrain = 0
        self.mean_norm_for_cls = configs.mean_norm_for_cls

        
        self.instance_CL = configs.instance_CL
        self.temporal_CL = configs.temporal_CL

        if self.instance_CL & self.temporal_CL :
            self.contrastive_loss = hard_inst_hard_temp
        elif self.instance_CL :
            self.contrastive_loss = hard_inst
        elif self.temporal_CL :
            self.contrastive_loss = hard_temp

        if self.head_type == "pretrain":
            # y : [bs x num_patch x nvars x patch_len]
            self.head = PretrainHead(configs.d_model, configs.patch_len, configs.head_dropout) 
        elif self.head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, configs.d_model, configs.num_patch, configs.target_dim, configs.head_dropout)
        elif self.head_type == "regression":
            # y: [bs x output_dim]
            self.head = RegressionHead(self.n_vars, configs.d_model, configs.target_dim, configs.head_dropout)
        elif self.head_type == "classification":
            # y: [bs x n_classes]
            if configs.aggregate == 'max':
                self.head = ClassificationHead_max(self.n_vars, configs.d_model, configs.num_class, configs.head_dropout)
            elif configs.aggregate == 'avg':
                self.head = ClassificationHead_avg(self.n_vars, configs.d_model, configs.num_class, configs.head_dropout)
            elif configs.aggregate == 'concat':
                self.head = ClassificationHead_concat(self.n_vars, configs.d_model, configs.num_patch,  configs.num_class, configs.head_dropout)

    def forward(self, z, x_mark_enc, x_dec, x_mark_dec, mask = None):        
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        if self.head_type == 'classification':
            if self.mean_norm_for_cls:
                z_mean = z.mean(axis=1).mean(axis=-1).unsqueeze(1)
                z = z-z_mean.unsqueeze(-1)
            _, z = self.backbone(z) 
            out = self.head(z) 
            return out
        


    
class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: 
            y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        x, _ = torch.max(x.squeeze(1),dim=2) # (64,1,128,125) -> (64,128,125) -> (64,128)
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y

class ClassificationHead_max(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

    def forward(self, x):
        x = self.flatten(x)
        x, _ = torch.max(x,dim=2) # (64,1,128,125) -> (64,128,125) -> (64,128)
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y
    
class ClassificationHead_avg(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.mean(x,dim=2) # (64,1,128,125) -> (64,128,125) -> (64,128)
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y
    
class ClassificationHead_concat(nn.Module):
    def __init__(self, n_vars, d_model, num_patch_new, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model*num_patch_new, n_classes)
        self.flatten = nn.Flatten(start_dim=1,end_dim=3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y

class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                     
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x


class MLPencoder(nn.Module):
    def __init__(self, c_in, patch_len,  d_model=128, shared_embedding=True, **kwargs):
        super().__init__()
        self.n_vars = c_in
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        
        self.act = nn.ReLU(inplace=True)
        if not shared_embedding: 
            self.W_P1 = nn.ModuleList()
            self.W_P2 = nn.ModuleList()
            for _ in range(self.n_vars): 
                self.W_P1.append(nn.Linear(patch_len, d_model))
                self.W_P2.append(nn.Linear(d_model, d_model))
        else:
            self.W_P1 = nn.Linear(patch_len, d_model)      
            self.W_P2 = nn.Linear(d_model, d_model)      

    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out1 = []
            x_out2 = []
            for i in range(n_vars):
                z = self.W_P1[i](x[:,:,i,:])
                x_out1.append(z)
                z = self.act(z)
                z = self.W_P2[i](z) 
                x_out2.append(z)
            x1 = torch.stack(x_out1, dim=2)
            x2 = torch.stack(x_out2, dim=2)
        else:
            x1 = self.W_P1(x)                                                      # x: [bs x num_patch x nvars x d_model]
            x2 = self.act(x1)
            x2 = self.W_P2(x2)                                                      # x: [bs x num_patch x nvars x d_model]
        x1 = x1.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        
        x2 = x2.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        
        x1 = x1.permute(0,1,3,2)
        x2 = x2.permute(0,1,3,2)
        return x1,x2
    
    

class DeepMLPencoder(nn.Module):
    def __init__(self, c_in, patch_len,
                 stride, features_domain,
                 dropout, max_len, n_heads, d_ff, e_layers,
                 d_model=128, shared_embedding=True, 
                 hidden_depth= 0, h_mode ='fix', seq_mixing = None,
                 num_experts = 4 , moe = None,
                 postional_embed = True, pool_size = 3,
                 **kwargs):
        super().__init__()
        self.n_vars = c_in
        self.patch_len = patch_len
        self.stride = stride
        self.features_domain = features_domain
        self.d_model = d_model
        self.shared_embedding = shared_embedding        
        self.act = nn.ReLU(inplace=True)
    
        self.dropout = nn.Dropout(p= dropout)
        self.hidden_depth = hidden_depth
        if h_mode == 'logscale': # old fashioned - do not use  
            self.hidden_sizes = np.ceil(np.logspace(np.log10(patch_len), np.log10(d_model), num = self.hidden_depth + 2, endpoint = True)).astype(int)
        elif h_mode == 'linscale': # old fashioned - do not use  
            self.hidden_sizes = np.ceil(np.linspace(patch_len,d_model, num = self.hidden_depth + 2, endpoint = True).astype(int))
        elif h_mode == 'fix':
            self.hidden_sizes = np.ones(self.hidden_depth + 2).astype(int) * d_model
            self.hidden_sizes[0] = patch_len
        self.hidden_layers = nn.ModuleList()
        if not shared_embedding: 
            for _ in self.hidden_sizes[:-1]:
                self.hidden_layers.append(nn.ModuleList)
            for hid_id in range(len(self.hidden_sizes) - 1):
                for _ in range(self.n_vars): 
                    self.hidden_layers[hid_id].append(nn.Linear(self.hidden_sizes[hid_id], self.hidden_sizes[hid_id + 1]))
        else:
            for hid_id in range(len(self.hidden_sizes) - 1):
                self.hidden_layers.append(nn.Linear(self.hidden_sizes[hid_id], self.hidden_sizes[hid_id + 1]))
        if not shared_embedding: 
            self.outputlayer = nn.ModuleList()
            for _ in range(self.n_vars): 
                self.outputlayer.append(nn.Linear(d_model, d_model))
        else:
           self.outputlayer = nn.Linear(d_model, d_model) 

        # seq mixer
        if seq_mixing == None or seq_mixing == 'None':
            self.seq_mixer = nn.Identity()
        else:
            self.seq_mixer = SeqMixer(d_model, dropout, max_len, n_heads, d_ff, e_layers, moe = moe, 
                                         num_experts = num_experts, postional_embed = postional_embed, 
                                         seq_mixer = seq_mixing, pool_size = pool_size)        

    def create_patch(self, xb, patch_len, stride, features_domain):

        """
        xb: [bs x seq_len x n_vars]
        """
        seq_len = xb.shape[1]
        num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        # tgt_len = patch_len  + stride*(num_patch-1)
        tgt_len = patch_len  + stride * num_patch
        pd = tgt_len - seq_len
        pad1 = (0, pd)
        xb = xb.permute(0,2,1)
        xb = F.pad(xb, pad1, "constant", 0)
        xb = xb.permute(0,2,1)

        xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]

        if features_domain == 'dft_mean_rm':
            xb = torch.abs(torch.fft.fft(xb- torch.mean(xb, dim = -1, keepdim=True), norm = 'ortho'))
        elif features_domain == 'dftfuse':
            xb = torch.abs(torch.fft.fft(xb- torch.mean(xb, dim = -1, keepdim=True), norm = 'ortho')) + xb
        elif features_domain == 'dft':
            xb = torch.abs(torch.fft.fft(xb, norm = 'ortho'))
        elif features_domain == 'dct_mean_rm':
            xb = dct.dct(xb- torch.mean(xb, dim = -1, keepdim=True))
        elif features_domain == 'dctfuse':
            xb = dct.dct(xb- torch.mean(xb, dim = -1, keepdim=True)) + xb
        elif features_domain == 'dct':
            xb = dct.dct(xb)
        elif features_domain == 'time_mean_rm':
            xb = xb - torch.mean(xb, dim = -1, keepdim=True)
        elif features_domain == 'time':
            xb = xb

        return xb

    def forward(self, x) -> Tensor:          
        
        x = self.create_patch(x, self.patch_len, self.stride, self.features_domain)

        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out1 = []
            x_out2 = []
            for i in range(n_vars):
                z = x[:,:,i,:]
                for l_id, layer in enumerate(self.hidden_layers[:,i]):
                    if l_id == len(self.hidden_layers[:,i]) - 1:
                        if l_id == 0:
                            z =self.dropout(layer(z))
                        else:
                            z= self.dropout(layer(z) + z)
                    else:
                        z = self.dropout(self.act(layer(z) + z))
                x_out1.append(z)
                z = self.act(z)
                z = self.outputlayer[i](z)
                x_out2.append(z)
            x1 = torch.stack(x_out1, dim=2)
            x2 = torch.stack(x_out2, dim=2)
        else:
            x1 = x
            for l_id, layer in enumerate(self.hidden_layers):
                    if l_id == len(self.hidden_layers) - 1:
                        if l_id == 0:                   # for the first layer no residual because of size mismatch
                            x1 = self.dropout(layer(x1))
                        else:
                            x1= self.dropout(layer(x1) + x1)
                    else:
                        if l_id == 0:                   # for the first layer no residual because of size mismatch
                            x1 = self.dropout(layer(x1))
                        else:
                            x1 = self.dropout(self.act(layer(x1) + x1))                                   # x: [bs x num_patch x nvars x d_model]
            if self.hidden_depth > 0:
                x2 = self.act(x1)
            else:
                x2 = x1
            x2 = self.outputlayer(x2)                                               # x: [bs x num_patch x nvars x d_model]

        x1 = x1.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        
        x2 = x2.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]

        # # # # # # # # # #
        # sequence mixer  #
        # # # # # # # # # #
        a1, b1, c1, d1 = x1.shape
        a2, b2, c2, d2 = x2.shape

        x1 = torch.reshape(x1, (a1 * b1, c1, d1))
        x2 = torch.reshape(x2, (a2 * b2, c2, d2))

        x1 = self.seq_mixer(x1)
        x2 = self.seq_mixer(x2)

        x1 = torch.reshape(x1, (a1, b1, c1, d1))
        x2 = torch.reshape(x2, (a2, b2, c2, d2))


        x1 = x1.permute(0,1,3,2)                                                    # x: [bs x nvars x d_model x num_patch ]
        x2 = x2.permute(0,1,3,2)                                                    # x: [bs x nvars x d_model x num_patch ]
        return x1,x2
    


