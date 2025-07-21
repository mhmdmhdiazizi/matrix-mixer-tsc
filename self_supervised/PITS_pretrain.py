### Code from: https://github.com/seunghan96/pits

import numpy as np
import pandas as pd
import os
import torch
import random
import sys

from src.models.PITS import PITS
from src.learner import Learner
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *

import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--cls', type=int, default=0, help='classification or not')
parser.add_argument('--pretrain_task', type=str, default='PI', help='PI vs. PD')
parser.add_argument('--CI', type=int, default=1, help='channel independence or not')

parser.add_argument('--instance_CL', type=int, default=0, help='Instance-wise contrastive learning')
parser.add_argument('--temporal_CL', type=int, default=1, help='Temporal contrastive learning')

parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
parser.add_argument('--dset_pretrain', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')

# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')

# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
parser.add_argument('--mean_norm', type=int, default=0, help='reversible instance normalization')

# Model args
parser.add_argument('--d_model', type=int, default=128, help='hidden dimension of MLP')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='masking ratio for the input')
parser.add_argument('--mask_schedule', type=float, default=0, help='mask_schedule')

# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=100, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')

# Device Id
parser.add_argument('--device_id', type=int, default=2, help='Device ID')
parser.add_argument('--seed', type=int, default=1, help='Random Seed')

# Deep embeddings arguments
parser.add_argument('--hidden_depth', type= int, default= 0, help= 'Depth of deep embedding layer')
parser.add_argument('--h_mode', type=str, default= 'fix', help= 'fix, logscael, and linscale for size increase in MLP')
parser.add_argument('--feature_domain', type = str, default= 'dftfuse', help= ' dftfuse, time/mean_rm, dft/mean_rm, dct/mean_rm, dctfuse')
parser.add_argument('--dropout', type = float, default= 0.1, help= 'dropout probablity in encoder')
parser.add_argument('--max_len', type = int, default= 10000, help= 'max number of token')
parser.add_argument('--n_heads', type = int, default= 8, help= 'number of heads in sequence mixer')
parser.add_argument('--d_ff', type = int, default= 512, help='dimension of mlp channel mixer')
parser.add_argument('--e_layers', type = int, default= 4, help= 'number of encoder layers')
parser.add_argument('--seq_mixing', type = str, default= None, help = 'transformer, poolformer')
parser.add_argument('--num_experts', type =  int, default= 4, help='number of experts in sequence mixer layers')
parser.add_argument('--moe', type= str, default = 'mlp', help= 'feature mixer : moe, mlp, einfft, monarch')
parser.add_argument('--postional_embed', type = int, default= 1, help = 'positional embedding (1) or not (0)')
parser.add_argument('--pool_size', type = int, default= 3, help= 'pool size in poolformer')

args = parser.parse_args()

print('args:', args)

if args.pretrain_task=='PI':
    path1 = 'saved_models/' + args.dset_pretrain + '/PITS_PI/' + args.model_type
elif args.pretrain_task=='PD':
    path1 = 'saved_models/' + args.dset_pretrain + '/PITS_PD/' + args.model_type
else:
    print('Choose either PI or PD task!') 
    sys.exit(0)
    
path2 = 'PITS_pretrained' + '_D' + str(args.d_model) + '_cw' +str(args.context_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride)
path2 += '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.pretrained_model_id)

if args.CI:
    path2 += '_CI'
else:
    path2 += '_CD' 

if args.mean_norm:
    path2 += '_mean_norm' 

args.save_path = path1 + '/' + path2 +'/'
args.pretrained_model = path2

if not os.path.exists(args.save_path): 
    os.makedirs(args.save_path)

set_device(args.device_id)

random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def get_model(c_in, args):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('input TS length:', args.context_points)
    print('patch size:', args.patch_len)
    print('number of patches:', num_patch)
    if args.patch_len > args.stride:
        print('Overlapping patches...')
    else:
        print('Non-Overlapping patches...')
    print('-'*80)
    # get model
    model = PITS(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                d_model=args.d_model,
                shared_embedding=args.CI,
                head_type='pretrain',
                head_dropout=args.head_dropout,
                mean_norm  = args.mean_norm,
                instance_CL = args.instance_CL,
                temporal_CL = args.temporal_CL,
                hidden_depth= args.hidden_depth, 
                h_mode = args.h_mode,
                dropout = args.dropout , max_len = args.max_len , 
                n_heads = args.n_heads, d_ff = args.d_ff, 
                e_layers = args.e_layers,
                num_experts = args.num_experts , moe = args.moe,
                postional_embed = args.postional_embed,
                pool_size = args.pool_size,
                seq_mixing= args.seq_mixing
                )        
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr():
    print("="*50)
    print("Loading DataLoaders")
    print("="*50)
    dls = get_dls(args)    
    
    print("="*50)
    print("Loading Models")
    print("="*50)
    model = get_model(dls.vars, args)
    
    # get loss
    loss_func = None
    
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, 
                        mask_ratio=args.mask_ratio, mask_schedule=args.mask_schedule,
                        overlap = None, features_domain = args.feature_domain)]        
    
    # define learner
    learn = Learner(args, dls, model, 
                        loss_func, 
                        lr=args.lr, 
                        cbs=cbs,
                        ft = False
                        )                 
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def pretrain_func(lr=args.lr):
    
    print("="*50)
    print("Loading DataLoaders")
    print("="*50)
    dls = get_dls(args)
    
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        PatchMaskCB(patch_len=args.patch_len, stride=args.stride, 
                        mask_ratio=args.mask_ratio, mask_schedule=args.mask_schedule,
                        overlap = None, features_domain = args.feature_domain),
        SaveModelCB(monitor='valid_loss', fname=args.pretrained_model, every_epoch=10,                       
                        path=args.save_path)
        ]    
    print("="*50)
    print("Loading Models")
    print("="*50)
    model = get_model(dls.vars, args)
    
    loss_func = None
    learn = Learner(args, dls, model, loss_func, 
                    lr=lr, cbs=cbs, ft= False)                         
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)
    
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.pretrained_model + '_losses.csv', float_format='%.6f', index=False)


if __name__ == '__main__':
    args.dset = args.dset_pretrain
    print('-'*100,'\n','-'*100)
    print('Finding the best Learning Rate')
    suggested_lr = find_lr()

    print('-'*100,'\n','-'*100)
    print('Start Pretraining')
    pretrain_func(suggested_lr)

    print('-'*100,'\n','-'*100)
    print('pretraining completed')