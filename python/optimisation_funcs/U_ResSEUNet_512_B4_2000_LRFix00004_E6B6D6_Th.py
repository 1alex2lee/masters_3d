import torch.nn as nn
import matplotlib as mpl
import torch.utils.data as data
import matplotlib.pyplot as plt
#package import
import time
import numpy as np
import torch
import random
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
# from visdom import Visdom
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    
# Hyper parameters
# random_seed = 37
# learning_rate = 0.0004
# batch_size = 4
# num_epochs= 2000
# device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# num_features = 2500
# ratio = 1
# num_channel = np.array([4,8,16,32,64,128,256,512])*ratio
# test_ratio = 0.85
# torch.set_num_threads(8)

# load data (expensive operation)
# path_node_inputs = 'Inputs.npy'
# path_node_target = 'TargetImages_Thinning.npy'
# samplenum = 603

# Inputs = np.load(path_node_inputs)[0:samplenum,:]
# Targets = np.load(path_node_target)[0:samplenum,:] 

# # Data
# random.seed(random_seed)
# torch.random.manual_seed(random_seed)
# np.random.seed(random_seed)
# index = [i for i in range(len(Targets))]
# random.shuffle(index)

# Inputs = Inputs[index]
# Targets = Targets[index]

# Inputs_train=Inputs[:int(len(Targets)*test_ratio),:,:,:]
# Inputs_test=Inputs[int(len(Targets)*test_ratio):,:,:,:]

# Targets_train=Targets[:int(len(Targets)*test_ratio),:,:][:,None,:,:] #only 1st channel kept (thinning image)
# Targets_test=Targets[int(len(Targets)*test_ratio):,:][:,None,:,:]

# class FormDataset(data.Dataset):
#     def __init__(self,Inputs,Targets):
#         self.Inputs=Inputs
#         self.Targets=Targets

#     def __getitem__(self,index):
#         Input, Target=self.Inputs[index,:,:,:], self.Targets[index,:,:]

#         Input=torch.from_numpy(Input).float().to(device)
#         Target=torch.from_numpy(Target).float().to(device)

#         a = torch.zeros([6,256,512])
#         a[:]=Input[:,:,:]
#         b = torch.zeros([1,256,512])
#         b[0] = Target
#         return a, b

#     def __len__(self):
#         return self.Inputs.shape[0]
# print('Inputs training dimensions:', Inputs_train.shape)
# print('Inputs testing dimensions:', Inputs_test.shape)
# print('Targets training dimensions:', Targets_train.shape)
# print('Targets testing dimensions:', Targets_test.shape)

# #Input and target is same
# Formdataset_train=FormDataset(Inputs_train,Targets_train)
# Formdataset_test=FormDataset(Inputs_test,Targets_test)

# train_loader=torch.utils.data.DataLoader(dataset=Formdataset_train,
#                                         batch_size=batch_size,
#                                         shuffle=True)

# test_loader=torch.utils.data.DataLoader(dataset=Formdataset_test,
#                                        batch_size=batch_size,
#                                        shuffle=False)

##########################
### MODEL
##########################
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1) # Squeeze
        w = self.fc(w)
        w, b = w.split(w.data.size(1) // 2, dim=1) # Excitation
        w = torch.sigmoid(w)

        return x * w + b # Scale and add bias


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se_block = SEBlock(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        out = self.se_block(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResUNet(torch.nn.Module):

    def __init__(self,num_channel,batch_size):
        super(ResUNet, self).__init__()

        ### ENCODER

        self.downscale=nn.Sequential(
            torch.nn.Conv2d(in_channels=6,
                           out_channels=num_channel[0],
                           kernel_size=(9,9),
                           stride=(1,1),
                           padding=4),

            nn.BatchNorm2d(num_channel[0]),

            nn.ReLU()
        )

        self.encoder_layer1=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[0],
                            out_channels=num_channel[1],
                            kernel_size=(8, 8),
                            stride=(2, 2),
                            padding=3),

            nn.ReLU(),

            nn.BatchNorm2d(num_channel[1])
        )

        self.encoder_layer2=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[1],
                            out_channels=num_channel[2],
                            kernel_size=(8, 8),
                            stride=(2, 2),
                            padding=(3, 3)),

            nn.BatchNorm2d(num_channel[2]),

            nn.ReLU()
        )

        self.encoder_layer3=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[2],
                            out_channels=num_channel[3],
                            kernel_size=(6, 6),
                            stride=(2, 2),
                            padding=(2, 2)),

            nn.BatchNorm2d(num_channel[3]),

            nn.ReLU()
        )

        self.encoder_layer4=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[3],
                            out_channels=num_channel[4],
                            kernel_size=(5, 6),
                            stride=(1, 2),
                            padding=2),

            nn.BatchNorm2d(num_channel[4]),

            nn.ReLU()
        )

        self.encoder_layer5=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[4],
                            out_channels=num_channel[5],
                            kernel_size=(4, 4),
                            stride=(2, 2),
                            padding=(1, 1)),

            nn.BatchNorm2d(num_channel[5]),

            nn.ReLU()
        )

        self.encoder_layer6=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[5],
                            out_channels=num_channel[6],
                            kernel_size=(4, 4),
                            stride=(2, 2),
                            padding=(1, 1)),

            nn.BatchNorm2d(num_channel[6]),

            nn.ReLU()
        )

        ### RESBLOCK
        self.res_layer = self._make_layer(BasicBlock, num_channel[6], num_channel[6], 6)

        ### DECODER

        self.decoder_layer6=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[7],
                            out_channels=num_channel[5],
                            kernel_size=(4, 4),
                            stride=(2, 2),
                            padding=(1, 1)),

            nn.BatchNorm2d(num_channel[5]),

            nn.ReLU()
        )

        self.decoder_layer5=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[6],
                            out_channels=num_channel[4],
                            kernel_size=(4, 4),
                            stride=(2, 2),
                            padding=(1, 1)),

            nn.BatchNorm2d(num_channel[4]),

            nn.ReLU()
        )

        self.decoder_layer4=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[5],
                            out_channels=num_channel[3],
                            kernel_size=(5, 6),
                            stride=(1, 2),
                            padding=2),

            nn.BatchNorm2d(num_channel[3]),

            nn.ReLU()
        )

        self.decoder_layer3=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[4],
                            out_channels=num_channel[2],
                            kernel_size=(6, 6),
                            stride=(2, 2),
                            padding=(2, 2)),

            nn.BatchNorm2d(num_channel[2]),

            nn.ReLU()
        )

        self.decoder_layer2=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[3],
                            out_channels=num_channel[1],
                            kernel_size=(8, 8),
                            stride=(2, 2),
                            padding=3),

            nn.BatchNorm2d(num_channel[1]),

            nn.ReLU()
        )

        self.decoder_layer1=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[2],
                            out_channels=num_channel[0],
                            kernel_size=(8, 8),
                            stride=(2, 2),
                            padding=(3, 3)),

            nn.ReLU(),

            nn.BatchNorm2d(num_channel[0])
        )

        self.updown=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[1],
                                     out_channels=2,
                                     kernel_size=(9, 9),
                                     stride=(1, 1),
                                     padding=4),

            nn.BatchNorm2d(2),

            nn.ReLU(),

            torch.nn.Conv2d(in_channels=2,
                                     out_channels=1,
                                     kernel_size=(9, 9),
                                     stride=(1, 1),
                                     padding=4)
        )



    def _make_layer(self, block, inplanes, planes, blocks, stride=1):

        layers = []

        for i in range(0, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        ### ENCODER
        encoded0 = self.downscale(x)

        encoded1 = self.encoder_layer1(encoded0)
        encoded2 = self.encoder_layer2(encoded1)
        encoded3 = self.encoder_layer3(encoded2)
        encoded4 = self.encoder_layer4(encoded3)
        encoded5 = self.encoder_layer5(encoded4)
        encoded6 = self.encoder_layer6(encoded5)

        ### RESBLOCKS

        res=self.res_layer(encoded6)

        ### DECODER
        decoded = torch.cat((res,encoded6),1)

        decoded = self.decoder_layer6(decoded)

        decoded = torch.cat((decoded,encoded5),1)

        decoded = self.decoder_layer5(decoded)

        decoded = torch.cat((decoded,encoded4),1)

        decoded = self.decoder_layer4(decoded)

        decoded = torch.cat((decoded,encoded3),1)

        decoded = self.decoder_layer3(decoded)

        decoded = torch.cat((decoded,encoded2),1)

        decoded = self.decoder_layer2(decoded)

        decoded = torch.cat((decoded,encoded1),1)

        decoded = self.decoder_layer1(decoded)

        decoded = torch.cat((decoded,encoded0),1)

        decoded = self.updown(decoded)

        return decoded


# torch.manual_seed(random_seed)
# model = ResUNet(num_channel,batch_size)
# model = model.to(device)
# model.train()
# def my_loss(decoded, target):
#     loss = (decoded * target).sum() / ((decoded * decoded).sum().sqrt() * (target * target).sum().sqrt())
#     return loss.mean()

# loss = nn.MSELoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# model.load_state_dict(torch.load('U_ResSEUNet_512_B4_2000_LRFix0.0004_E6B6D6_Th.pkl',map_location=torch.device('cpu')))