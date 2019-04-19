from __future__ import division
import torch
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
from model import decoder1
from model import encoder1
import torch.nn as nn

class WCT(nn.Module):
    def __init__(self,args):
        super(WCT, self).__init__()
        # Load pre-trained network
        vgg1 = load_lua(args.vgg1)
        decoder1_torch = load_lua(args.decoder1)

        self.e1 = encoder1(vgg1)
        self.d1 = decoder1(decoder1_torch)

