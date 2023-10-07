import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

import h5py
import copy
import numpy as np
from einops import rearrange
import torchvision
from dataloader import data_provider
import metrics



class Vit_train_task():
    def __init__(self, model, configs):
        self.configs = configs
        self.model = model
        self.device = configs.device#torch.device("cuda" if configs.cuda else "cpu")

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion


    def _get_data(self, train_flag):
        data_set, data_loader = data_provider(self.configs, train_flag=train_flag)
        return data_set, data_loader

    def train_valid(self, ):
        self.model.train()
        _, train_load = self._get_data(train_flag=True)#[1]
        _, test_load = self._get_data(train_flag=False)#[1]
        total_samples = len(train_load) * train_load.batch_size  # len(data_loader.dataset)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.configs.epochs):

            The_loss = 0

            for i, (data, target) in enumerate(train_load):
                optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)

                # output = F.log_softmax(self.model(data), dim=1)
                # loss = F.nll_loss(output, target)
                loss_f = criterion
                loss = loss_f(self.model.forward(data, mask=False, classification=True), target)

                The_loss += loss.detach().item()

                loss.backward()
                optimizer.step()

                # if i % 50 == 0:
                #     print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                #           ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                #           '{:6.4f}'.format(loss.item()))
            # loss_history.append(The_loss / len(data_loader))  # loss.item()

            if epoch % 5 == 0 or epoch <= 5:
                print('Epoch {}: Average train loss : {:.5f}'.format(epoch, The_loss / len(train_load)))

            vali_loss, vali_accuracy, vali_accuracy_sk, recall_sk, f1_sk = self.vali(test_load, criterion)
            if epoch % 5 == 0 or epoch <= 5:
                print('Epoch {}: '.format(epoch) + 'Average valid loss: ' + '{:.5f}'.format(vali_loss) +
                      ' Accuracy:'+'{:4.2f}'.format(vali_accuracy) + ' Accuracy_sk:'+'{:4.2f}'.format(vali_accuracy_sk) +
                      ' Recall_sk:'+f'{recall_sk}' +
                      f' f1_sk:{f1_sk}'+'\n')

    def vali(self, vali_loader, criterion):
    # def evaluate(self, data_loader, loss_history, Accuracy_rate, epoch):
        self.model.eval()

        total_samples = len(vali_loader) * vali_loader.batch_size  # len(data_loader.dataset)
        correct_samples = 0
        total_loss = 0
        Accuracy_sk = 0
        Recall_sk = 0
        f1_sk = 0

        # T_error=[]

        with torch.no_grad():
            for data, target in vali_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                ###
                # output = F.log_softmax(self.(data), dim=1)
                # loss = F.nll_loss(output, target, reduction='sum')
                loss_f = criterion
                output = self.model.forward(data, mask=False, classification=True)
                loss = loss_f(output, target)
                _, pred = torch.max(output, dim=-1)

                # _, pred = torch.max(output, dim=0)#

                total_loss += loss.item()
                # correct_samples += pred.eq(target).sum()
                ###
                # predicted = model.forward(data).argmax(axis=1)
                # print(predicted.shape)

                # total += target.size(0)
                # correct_samples += predicted.eq(target.argmax(axis=1)).sum().item()
                correct_samples += pred.eq(target).sum()

                Accuracy_sk += metrics.accuracy_score(pred.cpu().numpy(), target.cpu().numpy())
                Recall_sk += metrics.Recall_sk(pred.cpu().numpy(), target.cpu().numpy())
                f1_sk += metrics.f1_score_sk(pred.cpu().numpy(), target.cpu().numpy())

                # if predicted.eq(target.argmax(axis=1)).sum().item() < target.size(0):
                #     cha=abs(predicted-target.argmax(axis=1))
                #     for i in cha*T_tensor:
                #             if i!=0:
                #                 T_error.append(i.item())

        avg_loss = total_loss / len(vali_loader)  # total_samples
        Accuracy = (100.0 * correct_samples / total_samples).item()
        Accuracy_sk = Accuracy_sk/len(vali_loader)
        Recall_sk = Recall_sk/len(vali_loader)
        f1_sk = f1_sk/len(vali_loader)
        # loss_history.append(avg_loss)
        # Accuracy_rate.append(Accuracy)

        # if epoch % 10 == 0 or epoch <= 5:
        #     print('Epoch {}: '.format(epoch) + 'Average test loss: ' + '{:.5f}'.format(avg_loss) +
        #           '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
        #           '{:5}'.format(total_samples) + ' (' +
        #           '{:4.2f}'.format(Accuracy) + '%)\n')
            # print(T_error)

        return avg_loss, Accuracy, Accuracy_sk, Recall_sk, f1_sk

