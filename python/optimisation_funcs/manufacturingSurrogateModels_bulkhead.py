import torch.nn as nn
import torch
import torch.nn.functional as F

##########################
### Res-SE-U-Net from HFQ-ML paper
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
            torch.nn.Conv2d(in_channels=1,
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
                            kernel_size=(6, 6),
                            stride=(2, 2),
                            padding=2),
            
            nn.BatchNorm2d(num_channel[2]),
            
            nn.ReLU()
        )
        
        self.encoder_layer3=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[2],
                            out_channels=num_channel[3],
                            kernel_size=(4, 4),
                            stride=(2, 2),
                            padding=1),
            
            nn.BatchNorm2d(num_channel[3]),
            
            nn.ReLU()
        )
        
        ### RESBLOCK
        self.res_layer = self._make_layer(BasicBlock, num_channel[3], num_channel[3], 6)
        
        ### DECODER
        
        self.decoder_layer3=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[4],
                            out_channels=num_channel[2],
                            kernel_size=(4,4),
                            stride=(2,2),
                            padding=1),
            
            nn.BatchNorm2d(num_channel[2]),
            
            nn.ReLU()
        )
        
        self.decoder_layer2=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[3],
                            out_channels=num_channel[1],
                            kernel_size=(6,6),
                            stride=(2,2),
                            padding=2),
            
            nn.BatchNorm2d(num_channel[1]),
            
            nn.ReLU()
        )
        
        self.decoder_layer1=nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=num_channel[2],
                            out_channels=num_channel[0],
                            kernel_size=(8,8),
                            stride=(2,2),
                            padding=3),
            
            nn.BatchNorm2d(num_channel[0]),
            
            nn.ReLU()
        )
        
        self.updown=nn.Sequential(
            torch.nn.Conv2d(in_channels=num_channel[1],
                                     out_channels=int(num_channel[1]/4),
                                     kernel_size=(5, 5),
                                     stride=(1, 1),
                                     padding=2),
            
            nn.BatchNorm2d(int(num_channel[1]/4)),
            
            nn.ReLU(),
            
            torch.nn.Conv2d(in_channels=int(num_channel[1]/4),
                                     out_channels=1,
                                     kernel_size=(5, 5),
                                     stride=(1, 1),
                                     padding=2)
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
        
        ### RESBLOCKS
        
        res=self.res_layer(encoded3)
        
        ### DECODER
        decoded = torch.cat((res,encoded3),1)
        
        decoded = self.decoder_layer3(decoded)
        
        decoded = torch.cat((decoded,encoded2),1)
        
        decoded = self.decoder_layer2(decoded)
        
        decoded = torch.cat((decoded,encoded1),1)
        
        decoded = self.decoder_layer1(decoded)
        
        decoded = torch.cat((decoded,encoded0),1)
        
        decoded = self.updown(decoded)
        
        return decoded