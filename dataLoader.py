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

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        styleImgPath = os.path.join(self.stylePath,self.image_list[index])
        contentImg = Image.open(contentImgPath).convert('RGB')
        styleImg = Image.open(styleImgPath).convert('RGB')

        fz = self.fineSize
        w,h = contentImg.size
        if(fz == 0 and h != fz):
            contentImg = contentImg.resize((int(w*newh/h),fz))
            styleImg = styleImg.resize((int(w*newh/h),fz))
        elif(fz != 0 and w>h and w != fz):
            contentImg = contentImg.resize((fz,int(h*neww/w)))
            styleImg = styleImg.resize((fz,int(h*neww/w)))

        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        return contentImg.squeeze(0),styleImg.squeeze(0),self.image_list[index]