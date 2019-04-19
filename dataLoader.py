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
