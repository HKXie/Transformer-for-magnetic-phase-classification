import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

import h5py
import copy
import numpy as np
from einops import rearrange
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#残差模块，放在每个前馈网络和注意力之后
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

#layernorm归一化,放在多头注意力层和激活函数层。用绝对位置编码的BERT，layernorm用来自身通道归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
#放置多头注意力后，因为在于多头注意力使用的矩阵乘法为线性变换，后面跟上由全连接网络构成的FeedForward增加非线性结构
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim,dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)
#多头注意力层，多个自注意力连起来。使用qkv计算
class Attention(nn.Module):
    def __init__(self, dim, heads=8,dropout_rate=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim/heads) ** -0.5#原来写法dim**-0.5

        self.dropout = nn.Dropout(p=dropout_rate)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):#, mask = None
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)#(b,n,3*dim), n=hw=sequence
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)#(3,  b,h,n,d) for q,k,v, d=dim/h

        qk_dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale #(b,h,n,n)

        attn = qk_dots.softmax(dim=-1)#(b,h,n,n)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)#(b,h,n,d)
        out = rearrange(out, 'b h n d -> b n (h d)')#(b, n, dim),hd=dim
        out = self.to_out(out)
        out = self.dropout(out)

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout_rate=dropout_rate))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate=dropout_rate)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
        
#将图像切割成一个个图像块,组成序列化的数据输入Transformer执行图像分类任务。
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=2, dropout_rate=0.1, token_class=128):
        super().__init__()

        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)#是否可以改成卷积
        # self.conv_patch_to_embedding = nn.Conv2d(
        #                                         in_channels = channels,
        #                                         out_channels = dim,
        #                                         kernel_size = patch_size,
        #                                         stride = patch_size
        # )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout_rate=dropout_rate)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

        self.mlp_seq = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, token_class)
        )
        
        self.to_cls_token = nn.Identity()

    def Mask_sequence(self, seq, p=0.2):
    
        _,s,d=seq.shape
        masked_seq=seq.clone()

        masked_index=torch.bernoulli(torch.ones(s)*p)
        index = []
        for i, (bernoulli) in enumerate(masked_index):
            if bernoulli == 1:
                masked_seq[:,i,:]=1.
                index.append(i)

        return masked_seq.to(device), index

    def forward(self, img, mask=True,classification=False):

        if mask:
            p = self.patch_size

            # x = self.conv_patch_to_embedding(img)#(b, dim, h,w)
            # x = rearrange(x, 'b dim h w -> b (h w) dim')#(b, hw,dim)

            x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p) #(b,hw,patch_dim)  #hw个sequence
            
            x, mask_index = self.Mask_sequence(x,p=0.2)

            x = self.patch_to_embedding(x)#(b,hw,dim),   patch_dim->dim

            cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)#(b, 1, dim)
            x = torch.cat((cls_tokens, x), dim=1)#(b, hw+1, dim)
            x += self.pos_embedding#(b, hw+1, dim)
            x = self.transformer(x)#(b, hw+1,dim)
            x = self.to_cls_token(x[:, 1:])#(b,hw,dim)

            x = self.to_cls_token(x[:, mask_index,:])#(b,int(hw*p),dim)
            return self.mlp_seq(x), torch.tensor(mask_index)#(b,int(hw*p),num_token)# the tokenizer from Transformer



        elif classification:
            p = self.patch_size
            x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p) #(b,hw,patch_dim)  #hw个sequence
            x = self.patch_to_embedding(x)#(b,hw,dim),   patch_dim->dim

            # x = self.conv_patch_to_embedding(img)#(b, dim, h,w)
            # x = rearrange(x, 'b dim h w -> b (h w) dim')#(b, hw,dim)

            cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)#(b, 1, dim)
            x = torch.cat((cls_tokens, x), dim=1)#(b, hw+1, dim)
            x += self.pos_embedding#(b, hw+1, dim)

            x = self.transformer(x)#(b, hw+1,dim)
            x = self.to_cls_token(x[:, 0])#(b,dim)
            return self.mlp_head(x)#(b,class_num)




def pre_train(model, vae_model,optimizer, data_loader, loss_history,epoch):
    # total_samples = len(data_loader)*data_loader.batch_size#len(data_loader.dataset)
    model.train()

    The_loss = 0

    for img,_ in data_loader:

        img.to(device)

        optimizer.zero_grad()

        pre_train=model.forward(img.clone())

        pre_index, pre_seq=pre_train[1], pre_train[0]

        VAE_token=vae_model.encode(img.clone().cuda()).detach()
        # VAE_token=OneHotCategorical(VAE_token[:,pre_index.type(torch.long),:]).sample()
        VAE_token=VAE_token[:,pre_index.type(torch.long),:].argmax(dim=-1)

        Loss_func=nn.CrossEntropyLoss()
        # loss = Loss_func(pre_seq, VAE_token.detach())
        loss = Loss_func(pre_seq.reshape(-1,pre_seq.size(-1)), VAE_token.detach().reshape(-1))


        The_loss += loss.detach().item()

        loss.backward()
        optimizer.step()

    loss_history.append(The_loss/len(data_loader))

    # if epoch % 10==0 or epoch<=5:
    print('Epoch {}: Average pre-train loss : {:.5f}'.format(epoch,The_loss/len(data_loader)))
    

def pre_evaluate(model, vae_model, data_loader, loss_history, epoch):
    # total_samples = len(data_loader)*data_loader.batch_size#len(data_loader.dataset)
    model.eval()

    The_loss = 0
    with torch.no_grad():

        for img,_ in data_loader:

            img.to(device)

            pre_train=model.forward(img.clone())

            pre_index, pre_seq=pre_train[1], pre_train[0]

            VAE_token=vae_model.encode(img.clone().cuda()).detach()
            # VAE_token=OneHotCategorical(VAE_token[:,pre_index.type(torch.long),:]).sample()
            VAE_token=VAE_token[:,pre_index.type(torch.long),:].argmax(dim=-1)

            Loss_func=nn.CrossEntropyLoss()
            # loss = Loss_func(pre_seq, VAE_token.detach())
            loss = Loss_func(pre_seq.reshape(-1,pre_seq.size(-1)), VAE_token.detach().reshape(-1))


            The_loss += loss.detach().item()


    loss_history.append(The_loss/len(data_loader))

    # if epoch % 10==0 or epoch<=5:
    print('Epoch {}: Average pre-evaluate loss : {:.5f}'.format(epoch,The_loss/len(data_loader)) + '\n')





def fine_tune_train(model, optimizer, data_loader, loss_history,epoch):
    total_samples = len(data_loader)*data_loader.batch_size#len(data_loader.dataset)
    model.train()

    The_loss=0

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data=data.to(device)
        target=target.to(device)

        # output = F.log_softmax(model(data), dim=1)
        # loss = F.nll_loss(output, target)
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(model.forward(data, mask=False,classification=True), target)

        The_loss+=loss.detach().item()

        loss.backward()
        optimizer.step()


        # if i % 50 == 0:
        #     print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
        #           ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
        #           '{:6.4f}'.format(loss.item()))
    loss_history.append(The_loss/len(data_loader))#loss.item()

    if epoch % 10==0 or epoch<=5:
        print('Epoch {}: Average train loss : {:.5f}'.format(epoch,The_loss/len(data_loader)))


def fine_tune_evaluate(model, data_loader, loss_history,Accuracy_rate,epoch):
    model.eval()

    total_samples = len(data_loader)*data_loader.batch_size#len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    # T_error=[]

    with torch.no_grad():
        for data, target in data_loader:
            data=data.to(device)
            target=target.to(device)
            ###
            # output = F.log_softmax(model(data), dim=1)
            # loss = F.nll_loss(output, target, reduction='sum')
            loss_f = nn.CrossEntropyLoss()
            output = model.forward(data, mask=False,classification=True)
            loss = loss_f(output,target)
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

            # if predicted.eq(target.argmax(axis=1)).sum().item() < target.size(0):
            #     cha=abs(predicted-target.argmax(axis=1))
            #     for i in cha*T_tensor:
            #             if i!=0:
            #                 T_error.append(i.item())

    avg_loss = total_loss / len(data_loader)#total_samples
    Accuracy = (100.0 * correct_samples / total_samples).item()
    loss_history.append(avg_loss)
    Accuracy_rate.append(Accuracy)
    
    if epoch % 10 ==0 or epoch<=5:

        print('Epoch {}: '.format(epoch)+'Average test loss: ' + '{:.5f}'.format(avg_loss) +
            '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
            '{:5}'.format(total_samples) + ' (' +
            '{:4.2f}'.format(Accuracy) + '%)\n')
        # print(T_error)