import torch.nn as nn
import torch

# Encoder modeule
class encoder1(nn.Module):
    def __init__(self,vgg1):
        ## Convolution Layer 1
        super(encoder1,self).__init__()
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.conv1.bias = torch.nn.Parameter(vgg1.get(0).bias.float())
        self.conv1.weight = torch.nn.Parameter(vgg1.get(0).weight.float())
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        
        ## Convolution Layer 2
        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.conv2.bias = torch.nn.Parameter(vgg1.get(2).bias.float())
        self.conv2.weight = torch.nn.Parameter(vgg1.get(2).weight.float())
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        ## Forward Propogation
        out = self.relu(self.conv2(self.reflecPad1(self.conv1(x))))
        return out

#Decoder module
class decoder1(nn.Module):
    def __init__(self,d1):
        ## Convolution Layer
        super(decoder1,self).__init__()
        self.reflecPad2 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,3,3,1,0)
        self.conv3.bias = torch.nn.Parameter(d1.get(1).bias.float())
        self.conv3.weight = torch.nn.Parameter(d1.get(1).weight.float())

    def forward(self,x):
        ## Forward Propogation
        out = self.conv3(self.reflecPad2(x))
        return out