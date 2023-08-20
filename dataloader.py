import torch
from torch import nn
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import h5py
import os
import copy

import numpy as np
import matplotlib.pylab as plt

#
# class Heisenberg_dataset(Dataset):
#     def __init__(self, x_list):
#         """ initialize the class instance
#         Args:
#             x_list: data with list type
#         Returns:
#             None
#         """
#         if not isinstance(x_list, list):
#             raise ValueError("input x_list is not a list type")
#         self.data = x_list
#         print("intialize success")
#
#     def __getitem__(self, idx):
#         # print("__getitem__ is called")
#         return self.data[idx]
#
#     def __len__(self):
#         # print("__len__ is called")
#         return len(self.data)


# data_train = data_provider(data_all_closeTc + data_all_2_closeTc)#+data_all_test_closeTc
# data_test = data_provider(data_all_test)#data_all_beyondTc+data_all_2_beyondTc+
#
#
# train_loader = DataLoader(data_train, batch_size = args.batch_size, shuffle=True)
# test_loader = DataLoader(data_test, batch_size = args.batch_size, shuffle=True )



def data_provider(configs, train_flag=True):
    if train_flag:
        flag = 'train'
    else:
        flag = 'test'

    data_set = Dataset_heisenberg_Vit(configs=configs, flag=flag)

    if train_flag:

        data_loader = DataLoader(
            data_set,
            batch_size=configs.batch_size,
            shuffle=True
            # num_workers=args.num_workers,
            # drop_last=drop_last,
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=configs.batch_size,
            shuffle=True,
            drop_last=True#False
            # num_workers=args.num_workers,
            # drop_last=drop_last,
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )

    return data_set, data_loader


class Dataset_heisenberg_Vit(Dataset):
    def __init__(self, configs, flag='train'):

        # size [seq_len, label_len, pred_len]
        # info
        self.configs = configs
        self.flag = flag

        self.root_path = configs.data_path#root_path
        self.data_train = configs.train_data_name
        self.data_test = configs.test_data_name


        self.__read_data__()

    def __read_data__(self):

        self.data_set = []
        self.Phase_label = []
        data_list = self.data_train

        if self.flag == 'train':
            data_list = self.data_train
            data_key = 'data_train'

        elif self.flag == 'test':
            data_list = self.data_test
            data_key = 'data_test'

        else:
            pass

        file_path = os.path.join(self.root_path, data_list[0])
        # 载入数据
        data = []

        with h5py.File(file_path, 'r') as f:

            for i in np.arange(1, 41):
                dataset = f[str(i) + '.0K']

                data.append(dataset[:])

        data = np.array(data)

        sample_num, data_num = data.shape[1], data.shape[0]

        data = data.reshape(-1, self.configs.channels, self.configs.imgage_size, self.configs.imgage_size)

        T = np.linspace(1, 40, data_num)
        Tc = 19

        Phase_label = []

        [Phase_label.append(0) if i <= Tc else Phase_label.append(1) for i in T for j in range(sample_num)]


        # 数据做变形,处于0到1之间,并将其和独热码转为tensor
        data_set = (data + 1) / 2

        self.data_set = torch.tensor(data_set).to(torch.float32)                #数据
        self.Phase_label = torch.tensor(Phase_label).to(torch.long)             #标签

    def __getitem__(self, idx):
        # print("__getitem__ is called")
        return [self.data_set[idx], self.Phase_label[idx]]

    def __len__(self):
        # print("__len__ is called")
        return len(self.data_set)
