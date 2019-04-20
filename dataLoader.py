from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.utils as vutils
import torch
import os
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def isimg(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

class Dataset(data.Dataset):
    def __init__(self,cp,sp,fineSize):
        super(Dataset,self).__init__()
        self.image_list = [x for x in os.listdir(cp) if isimg(x)]
        self.fineSize = fineSize
        self.stylePath = sp
        self.contentPath = cp
        self.prep = transforms.Compose([
                    transforms.Scale(self.fineSize),
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return len(self.image_list)