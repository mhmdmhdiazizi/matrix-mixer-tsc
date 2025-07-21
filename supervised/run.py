import argparse
import os
import torch
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=0, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_in', type=int, default=0, help='input channels size') ### PITS ### ### unnecessary kuz enc_in in exp classifiacation ###
    parser.add_argument('--target_dim', type=int, default=0, help='number of target points') ### PITS ### ### unnecessary kuz num_class in exp classifiacation ###
    parser.add_argument('--stride', type=int, default=7, help='stride size') ### PITS ###
    parser.add_argument('--features_domain', type=str, default='dct', ### PITS ###
                        help='features_domain in tokenization')
    parser.add_argument('--num_patch', type=int, default=0, help='number of patches') ### PITS ###
    parser.add_argument('--context_points_fn', type=int, default=7, help='number of context points') ### PITS ### seq_len
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model') ### also for PITS ###
    parser.add_argument('--shared_embedding', type=int, default=1, help='shared_embedding(channel independence) or not')  ### also for PITS ### ### default is okay ###
    parser.add_argument('--head_type', type=str, default='classification', ### PITS ### ### unnecessary kuz tassk_name in exp classifiacation ###
                        help='head(task) type')
    parser.add_argument('--aggregate', type=str, default='avg', ### PITS ### ### default is okay ###
                        help='aggregation type')
    parser.add_argument('--instance_CL', type=int, default=0, help='Instance-wise contrastive learning') ### PITS ###
    parser.add_argument('--temporal_CL', type=int, default=0, help='Temporal contrastive learning') ### PITS ### 
    parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout') ### PITS ### ### default is okay ###
    parser.add_argument('--mean_norm_for_cls', type=int, default=0) ### PITS ### ### default is okay ###
    parser.add_argument('--hidden_depth', type=int, default=0, help= 'number of layers with non-linearity in tokenixation step') ### PITS ###
    parser.add_argument('--h_mode', type=str, default= 'fix', help= 'fix, logscael, and linscale for size increase in MLP') ### PITS ### ### default is okay ###
    parser.add_argument('--seq_mixing', type = str, default= None, help = 'transformer, poolformer') ### PITS ### 
    parser.add_argument('--max_len', type = int, default= 2000, help= 'max number of token') ### PITS ### ### default is okay ###
    parser.add_argument('--num_experts', type =  int, default= 4, help='number of experts in sequence mixer layers') ### PITS ### ### default is okay ###
    parser.add_argument('--moe', type= str, default = 'mlp', help= 'feature mixer : moe, mlp, einfft, monarch') ### PITS ### ### default is okay ###
    parser.add_argument('--postional_embed', type = int, default= 1, help = 'positional embedding (1) or not (0)') ### PITS ### ### default is okay ###
    parser.add_argument('--pool_size', type = int, default= 3, help= 'pool size in poolformer') ### PITS ### ### default is okay ###

    parser.add_argument('--n_heads', type=int, default=8, help='num of heads') ### also for PITS ### ### default is okay ###
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers') ### also for PITS ###
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn') ### also for PITS ###
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout') ### also for PITS ### ### default is okay ###
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0, 1', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length') ### also for PITS ###

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    

    print('Args in experiment:')
    # print_args(args)
    print(args)

    if args.task_name == 'classification':
        Exp = Exp_Classification


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = 'tn={}_mi={}_m={}_d={}_dm={}_s={}_pl={}_stride={}_fd={}_hd={}_sm={}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.d_model, 
                ii, 
                args.patch_len, 
                args.stride, 
                args.features_domain, 
                args.hidden_depth, 
                args.seq_mixing, 
                )  

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = 'tn={}_mi={}_m={}_d={}_dm={}_s={}_pl={}_stride={}_fd={}_hd={}_sm={}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.d_model, 
                ii, 
                args.patch_len, 
                args.stride, 
                args.features_domain, 
                args.hidden_depth, 
                args.seq_mixing, 
                ) 

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
