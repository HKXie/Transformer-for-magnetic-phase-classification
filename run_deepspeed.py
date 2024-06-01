import deepspeed
import torch
# import torchvision
from torch import nn
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from einops import rearrange
# from torchvision import datasets, transforms
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

import argparse

import h5py
import os
import copy

import numpy as np
import matplotlib.pylab as plt
import Transformer_2nd
from dataloader import data_provider, Dataset_heisenberg_Vit
import train


def add_deep_argment():
        parser = argparse.ArgumentParser(description='Vit_heisenberg_pahses_classification')

        
        parser.add_argument('--imgage-size', type=int, default=20,help='input image size')
        parser.add_argument('--imgage-patch', type=int, default=4,help='segmentation image patch')
        parser.add_argument('--channels', type=int, default=2, help='_')
        # parser.add_argument('--num-class', type=int, default=2,help='num class')
        # parser.add_argument('--mlp-dim', type=int, default=128, help='_')
        # parser.add_argument('--num_token',type=int, default=96, help='_')
        # parser.add_argument('--dropout-rate', type=int, default=0.3, help='_')
        parser.add_argument('--data_path', type=str, default='data/magnetic_phase_data',help='')
        parser.add_argument('--train_data_name', type=str,
                        default=['124_MT_20size_1.5T_jiangede.hdf5',],help='')

        parser.add_argument('--test_data_name', type=str,
                        default=['124_MT_20size_2T_jiangede.hdf5',],help='')



        parser.add_argument('--with_cuda', default=False, action='store_true',
                            help='use CPU in case there is no GPU support')
        parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

        # train
        # parser.add_argument('-b',
        #                         '--batch_size',
        #                         default=32,
        #                         type=int,
        #                         help='mini-batch size (default: 32)')
        parser.add_argument('-e',
                                '--epochs',
                                default=30,
                                type=int,
                                help='number of total epochs (default: 30)')
        parser.add_argument('--local_rank',
                                type=int,
                                default=-1,
                                help='local rank passed from distributed launcher')

        parser.add_argument('--log-interval',
                                type=int,
                                default=2000,
                                help="output logging information at a given interval")

        # parser.add_argument('--moe',
        #                         default=False,
        #                         action='store_true',
        #                         help='use deepspeed mixture of experts (moe)')

        # parser.add_argument('--ep-world-size',
        #                         default=1,
        #                         type=int,
        #                         help='(moe) expert parallel world size')
        # parser.add_argument('--num-experts',
        #                         type=int,
        #                         nargs='+',
        #                         default=[
        #                         1,
        #                         ],
        #                         help='number of experts list, MoE related.')
        # parser.add_argument(
        #         '--mlp-type',
        #         type=str,
        #         default='standard',
        #         help=
        #         'Only applicable when num-experts > 1, accepts [standard, residual]')
        # parser.add_argument('--top-k',
        #                         default=1,
        #                         type=int,
        #                         help='(moe) gating top 1 and 2 supported')
        # parser.add_argument(
        #         '--min-capacity',
        #         default=0,
        #         type=int,
        #         help=
        #         '(moe) minimum capacity of an expert regardless of the capacity_factor'
        # )
        # parser.add_argument(
        #         '--noisy-gate-policy',
        #         default=None,
        #         type=str,
        #         help=
        #         '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
        # )
        # parser.add_argument(
        #         '--moe-param-group',
        #         default=False,
        #         action='store_true',
        #         help=
        #         '(moe) create separate moe param groups, required when using ZeRO w. MoE'
        # )
        # Include DeepSpeed configuration arguments

        parser = deepspeed.add_config_arguments(parser=parser)


        args = parser.parse_args()
        return args

# def add_argments():
#         parser = argparse.ArgumentParser(description='origin')

#         parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
#         parser.add_argument('--data_path', type=str, default='data',help='')
#         parser.add_argument('--train_data_name', type=str,
#                         default=['124_MT_20size_1.5T_jiangede.hdf5',],help='')

#         parser.add_argument('--test_data_name', type=str,
#                         default=['124_MT_20size_2T_jiangede.hdf5',],help='')


#         args = parser.parse_args()

#         return args


# args.cuda = not args.no_cuda and torch.cuda.is_available()
# args.device = torch.device("cuda" if args.cuda else "cpu")

deepspeed.init_distributed()



deep_args = add_deep_argment()
# args = add_argments()

# print(f'****************************{deep_args.local_rank}, and {deep_args.batch_size}')

Model_vit = Transformer_2nd.ViT(image_size=20, patch_size=4, num_classes=2, channels=2,
        dim=24, depth=4, heads=8, mlp_dim=128, dropout_rate=0.4,token_class=128)

parameters = filter(lambda p: p.requires_grad, Model_vit.parameters())



data_train_set = Dataset_heisenberg_Vit(deep_args,flag='train')
data_test_set = Dataset_heisenberg_Vit(deep_args,flag='test')


model_engine, optimizer, trainloader, _ = deepspeed.initialize(
    args=deep_args, model=Model_vit,model_parameters=parameters,training_data=data_train_set)

# exp = train.Vit_train_task(Model_vit, args)
# exp.train_valid()

# def train_valid(self, ):

print(f'****************************{model_engine.local_rank}, and {trainloader.batch_size}')

criterion = nn.CrossEntropyLoss()

for epoch in range(deep_args.epochs):

        The_loss = 0

        for i, data in enumerate(trainloader):

                optimizer.zero_grad()
                inputs = data[0].to(model_engine.local_rank)
                target = data[1].to(model_engine.local_rank)
                outputs = model_engine(inputs)

                loss = criterion(outputs, target)

                # loss.backward()
                model_engine.backward(loss)

                # optimizer.step()
                model_engine.step()

                The_loss += loss.item()
                # if i % 50 == 0:
                #     print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                #           ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                #           '{:6.4f}'.format(loss.item()))
                # loss_history.append(The_loss / len(data_loader))  # loss.item()

        if epoch % 5 == 0 or epoch <= 5:
                print('Epoch {}: Average train loss : {:.5f}'.format(epoch, The_loss / len(trainloader)))
                
print('Finish Training')
        #     vali_loss, vali_accuracy, vali_accuracy_sk, recall_sk, f1_sk = self.vali(test_load, criterion)
        #     if epoch % 5 == 0 or epoch <= 5:
        #         print('Epoch {}: '.format(epoch) + 'Average valid loss: ' + '{:.5f}'.format(vali_loss) +
        #               ' Accuracy:'+'{:4.2f}'.format(vali_accuracy) + ' Accuracy_sk:'+'{:4.2f}'.format(vali_accuracy_sk) +
        #               ' Recall_sk:'+f'{recall_sk}' +
        #               f' f1_sk:{f1_sk}'+'\n')

