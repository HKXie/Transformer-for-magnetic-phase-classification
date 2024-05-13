import torch
# import torchvision
from torch import nn
import time
import torch.nn.functional as F
from torch import optim
from einops import rearrange
# from torchvision import datasets, transforms
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

import argparse

import numpy as np
import matplotlib.pylab as plt
import Transformer_2nd
from dataloader import data_provider
import train

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Vit_heisenberg_pahses_classification')

    parser.add_argument('--data_path', type=str, default='data',help='')
    parser.add_argument('--train_data_name', type=str,
                        default=['124_MT_20size_1.5T_jiangede.hdf5',],help='')

    parser.add_argument('--test_data_name', type=str,
                        default=['124_MT_20size_2T_jiangede.hdf5',],help='')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--imgage-size', type=int, default=20,help='input image size')
    parser.add_argument('--imgage-patch', type=int, default=4,help='segmentation image patch')
    parser.add_argument('--channels', type=int, default=2, help='_')
    parser.add_argument('--num-class', type=int, default=2,help='num class')
    parser.add_argument('--mlp-dim', type=int, default=128, help='_')
    parser.add_argument('--num_token',type=int, default=96, help='_')
    parser.add_argument('--dropout-rate', type=int, default=0.3, help='_')


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    Model_vit = Transformer_2nd.ViT(image_size=args.imgage_size, patch_size=args.imgage_patch, num_classes=args.num_class, channels=args.channels,
            dim=24, depth=4, heads=8, mlp_dim=args.mlp_dim, dropout_rate=0.4,token_class=128)

    # data_train_loader = data_provider(args, train_flag=True)[1]
    # data_test_loader = data_provider(args, train_flag=False)[1]

    exp = train.Vit_train_task(Model_vit, args)
    exp.train_valid()



